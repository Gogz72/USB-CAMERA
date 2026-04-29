import os
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image
import cv2
from pygrabber.dshow_graph import FilterGraph


class UsbCamera:
    _BACKENDS = (
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    )

    _camera_names_cache = {}
    _camera_names_cache_ts = 0.0
    _camera_names_cache_lock = threading.Lock()
    _available_cameras_cache = {}
    _available_cameras_cache_ts = 0.0
    _available_cameras_cache_lock = threading.Lock()

    def __init__(self, camera_index=None, logger=None, auto_reconnect=True, image_save_path=None, video_save_path=None):
        """
        Initialize USB camera wrapper (OpenCV / VideoCapture).

        - camera_index: None = auto-detect first available, or an int index (0,1,...)
        - logger: optional logger with .debug/.warning/.error methods
        - auto_reconnect: if True the capture thread will try to reopen camera on failures
        """
        self.logger = logger
        self.auto_reconnect = auto_reconnect

        self._capture_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._record_stop = threading.Event()
        self._started_event = threading.Event()
        self._stopped_event = threading.Event()

        self._state = "IDLE"

        self.camera_index = camera_index if camera_index is not None else self._auto_detect_camera()
        self._log(f"UsbCamera: Initialized with index '{self.camera_index}'.")

        self._cap = None
        self._graph = None
        self._capture_mode = None
        self._active_backend = None
        self._last_frame = None
        self._last_image_path = None
        self._frame_timestamp = None

        self.image_save_path = image_save_path or tempfile.gettempdir()
        self.video_save_path = video_save_path or tempfile.gettempdir()

        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.video_save_path, exist_ok=True)

    # ------------------------------------------------------------
    def _log(self, msg, level="debug"):
        """Internal safe logger wrapper."""
        if self.logger:
            try:
                if level == "error":
                    self.logger.error(msg)
                elif level == "warning":
                    self.logger.warning(msg)
                else:
                    self.logger.debug(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    def _set_state(self, state: str):
        with self._state_lock:
            self._state = state

    def get_state(self) -> str:
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------
    # Core camera functions
    # ------------------------------------------------------------
    @staticmethod
    def _read_test_frame(cap, timeout: float = 2.0, sleep_interval: float = 0.05) -> Tuple[bool, Any]:
        """
        Try to read a valid frame, allowing slow DirectShow/MSMF devices to warm up.
        """
        deadline = time.time() + timeout
        last_frame = None
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                return True, frame
            last_frame = frame
            time.sleep(sleep_interval)
        return False, last_frame

    @classmethod
    def _open_cv_capture_candidates(cls, camera_ref):
        """
        Yield (backend_name, backend_id, open_ref, cap) for index/name variants.

        Numeric OpenCV indices are not always aligned with DirectShow device order
        on hardened Windows images. DirectShow can also accept "video=<device name>"
        for some builds, so named candidates are worth trying.
        """
        refs = [camera_ref]
        if isinstance(camera_ref, str) and not camera_ref.lower().startswith("video="):
            refs.append(f"video={camera_ref}")

        for open_ref in refs:
            for backend_name, backend_id in cls._BACKENDS:
                cap = cv2.VideoCapture(open_ref, backend_id)
                yield backend_name, backend_id, open_ref, cap

    @classmethod
    def list_cameras(cls, max_test: int = 8, ttl_seconds: float = 10.0) -> list:
        """
        Return list of available camera indices up to max_test.
        Only includes numeric devices that can deliver a valid frame through OpenCV.
        """
        now = time.time()
        cache_key = int(max_test)
        with cls._available_cameras_cache_lock:
            cached = cls._available_cameras_cache.get(cache_key)
            if cached is not None and (now - cls._available_cameras_cache_ts) < ttl_seconds:
                return list(cached)

        available = []
        for i in range(max_test):
            for backend_name, _backend_id, _open_ref, cap in cls._open_cv_capture_candidates(i):
                try:
                    if not cap.isOpened():
                        continue
                    ok, _frame = cls._read_test_frame(cap, timeout=1.0)
                    if ok:
                        available.append(i)
                        break
                finally:
                    try:
                        cap.release()
                    except Exception:
                        pass
        with cls._available_cameras_cache_lock:
            cls._available_cameras_cache[cache_key] = list(available)
            cls._available_cameras_cache_ts = time.time()
        return available

    @classmethod
    def list_camera_details(cls, max_test: int = 16, read_timeout: float = 1.5) -> dict:
        """
        Return a diagnostic camera report.

        - directshow_devices: devices exposed to DirectShow via pygrabber.
        - windows_pnp_devices: camera-like devices visible in Device Manager/PnP.
        - opencv_probes: per-index backend open/read results.

        This is intentionally broader than list_cameras() so ATM builds can show
        the difference between "Windows can see it" and "OpenCV can read it".
        """
        directshow_devices = cls._get_camera_names_map(ttl_seconds=0.0)
        report = {
            "directshow_devices": [
                {"index": index, "name": name}
                for index, name in sorted(directshow_devices.items())
            ],
            "windows_pnp_devices": cls._list_windows_pnp_camera_devices(),
            "opencv_probes": [],
        }

        candidate_refs = list(range(max_test))
        for name in directshow_devices.values():
            if name not in candidate_refs:
                candidate_refs.append(name)

        seen = set()
        for camera_ref in candidate_refs:
            if camera_ref in seen:
                continue
            seen.add(camera_ref)

            ref_result = {"ref": camera_ref, "attempts": []}
            for backend_name, _backend_id, open_ref, cap in cls._open_cv_capture_candidates(camera_ref):
                attempt = {
                    "backend": backend_name,
                    "open_ref": open_ref,
                    "opened": False,
                    "frame": False,
                    "resolution": None,
                    "error": None,
                }
                try:
                    attempt["opened"] = bool(cap.isOpened())
                    if attempt["opened"]:
                        ok, frame = cls._read_test_frame(cap, timeout=read_timeout)
                        attempt["frame"] = bool(ok)
                        if ok and frame is not None:
                            h, w = frame.shape[:2]
                            attempt["resolution"] = (w, h)
                except Exception as exc:
                    attempt["error"] = str(exc)
                finally:
                    try:
                        cap.release()
                    except Exception:
                        pass
                ref_result["attempts"].append(attempt)
            report["opencv_probes"].append(ref_result)

        return report

    @staticmethod
    def _list_windows_pnp_camera_devices() -> list:
        """
        Best-effort Device Manager style enumeration using built-in PowerShell/WMI.
        """
        script = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { "
            "$_.PNPClass -in @('Camera','Image','MEDIA') -or "
            "$_.Name -match 'camera|webcam|video|imaging' "
            "} | "
            "Select-Object Name,PNPClass,DeviceID,Status,Manufacturer | "
            "ConvertTo-Json -Compress"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                return []

            parsed = json.loads(completed.stdout)
            if isinstance(parsed, dict):
                parsed = [parsed]
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @staticmethod
    def _auto_detect_camera(max_test: int = 8) -> int:
        """Try to find first available camera index; return 0 if none found (conservative)."""
        cams = UsbCamera.list_cameras(max_test=max_test)
        return cams[0] if cams else 0

    def set_camera(self, index):
        """Change target camera index/name. Will restart capture if running."""
        with self._capture_lock:
            self._log(f"UsbCamera: Changing camera index to {index}.")
            running = self.is_capturing()

        if running:
            self.stop_capture(join=True, timeout=5.0)

        with self._capture_lock:
            self.camera_index = index
            self._last_frame = None
            self._frame_timestamp = None

        if running:
            self.start_capture(wait_ready=True, timeout=5.0)

    def get_status(self) -> dict:
        with self._capture_lock:
            opened = bool(self._cap and self._cap.isOpened())
            if self._graph is not None:
                opened = True
            state = self.get_state()
            res = {
                "Index": self.camera_index,
                "Opened": opened,
                "State": state,
                "CameraName": self._get_camera_name(self.camera_index),
                "Backend": self._active_backend,
                "CaptureMode": self._capture_mode,
            }

            if self._cap and self._cap.isOpened():
                try:
                    w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or None
                    res.update({"Resolution": (w, h), "FPS": fps})
                except Exception:
                    res.update({"Resolution": None, "FPS": None})
            elif self._last_frame is not None:
                h, w = self._last_frame.shape[:2]
                res.update({"Resolution": (w, h), "FPS": None})

            res["LastFrameTime"] = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._frame_timestamp))
                if self._frame_timestamp else None
            )

            self._log(f"UsbCamera: Status = {res}")
            return res

    @classmethod
    def _get_camera_names_map(cls, ttl_seconds: float = 10.0):
        """
        Enumerate DirectShow video devices with short-lived cache.
        Returns: {index: "Camera Name"}
        """
        now = time.time()
        with cls._camera_names_cache_lock:
            if cls._camera_names_cache and (now - cls._camera_names_cache_ts) < ttl_seconds:
                return dict(cls._camera_names_cache)

            try:
                devices = FilterGraph().get_input_devices()
                cls._camera_names_cache = {i: name for i, name in enumerate(devices)}
                cls._camera_names_cache_ts = now
            except Exception:
                cls._camera_names_cache = {}
                cls._camera_names_cache_ts = now

            return dict(cls._camera_names_cache)

    def _get_camera_name(self, index) -> str:
        if isinstance(index, str):
            return index
        names = self._get_camera_names_map()
        return names.get(index, "Unknown Camera")

    # ------------------------------------------------------------
    # Capture thread + helpers
    # ------------------------------------------------------------
    def _open_capture(self) -> bool:
        """Try to open cv2.VideoCapture for self.camera_index with backend/name fallback."""
        try:
            last_failure = None
            for backend_name, _backend_id, open_ref, cap in self._open_cv_capture_candidates(self.camera_index):
                try:
                    if not cap.isOpened():
                        last_failure = f"{backend_name} could not open {open_ref!r}"
                        continue

                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    with self._capture_lock:
                        self._cap = cap
                        self._graph = None
                        self._capture_mode = "opencv"
                        self._active_backend = backend_name

                    self._log(
                        f"UsbCamera: Camera {self.camera_index!r} opened successfully "
                        f"with {backend_name} ({open_ref!r})."
                    )
                    return True
                except Exception as exc:
                    last_failure = f"{backend_name} exception for {open_ref!r}: {exc}"
                finally:
                    if self._cap is not cap:
                        try:
                            cap.release()
                        except Exception:
                            pass

            self._log(
                f"UsbCamera: Failed to open camera {self.camera_index!r}. Last failure: {last_failure}",
                "warning",
            )
            return self._open_pygrabber_capture()

        except Exception as e:
            self._log(f"UsbCamera: Exception opening VideoCapture - {e}", "error")
            return False

    def _resolve_directshow_index(self):
        if isinstance(self.camera_index, int):
            return self.camera_index

        names = self._get_camera_names_map(ttl_seconds=0.0)
        wanted = str(self.camera_index)
        if wanted.lower().startswith("video="):
            wanted = wanted[6:]

        for index, name in names.items():
            if name == wanted:
                return index

        wanted_lower = wanted.lower()
        for index, name in names.items():
            if wanted_lower in name.lower():
                return index

        return None

    def _open_pygrabber_capture(self) -> bool:
        """Fallback to a raw DirectShow graph when OpenCV cannot open the device."""
        device_index = self._resolve_directshow_index()
        if device_index is None:
            self._log(f"UsbCamera: No DirectShow device matches {self.camera_index!r}.", "error")
            return False

        try:
            graph = FilterGraph()

            def on_frame(frame):
                try:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr_frame = frame

                with self._capture_lock:
                    self._last_frame = bgr_frame
                    self._frame_timestamp = time.time()

                if not self._started_event.is_set():
                    self._started_event.set()
                    self._set_state("RUNNING")
                    self._log("UsbCamera: First DirectShow frame received; capture is ready.")

            graph.add_video_input_device(device_index)
            graph.add_sample_grabber(on_frame)
            graph.add_null_render()
            graph.prepare_preview_graph()
            graph.run()

            with self._capture_lock:
                self._cap = None
                self._graph = graph
                self._capture_mode = "pygrabber_dshow"
                self._active_backend = "PYGRABBER_DSHOW"

            self._log(
                f"UsbCamera: Camera {self.camera_index!r} opened with DirectShow graph "
                f"(device index {device_index})."
            )
            return True
        except Exception as exc:
            self._log(f"UsbCamera: DirectShow graph fallback failed - {exc}", "error")
            return False

    def _is_capture_open(self) -> bool:
        with self._capture_lock:
            return bool((self._cap and self._cap.isOpened()) or self._graph is not None)

    def _close_capture(self):
        """Release the capture device gracefully."""
        with self._capture_lock:
            try:
                if self._cap:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                if self._graph:
                    try:
                        self._graph.stop()
                    except Exception:
                        pass
                self._cap = None
                self._graph = None
                self._capture_mode = None
                self._active_backend = None
                self._log("UsbCamera: VideoCapture released.")
            except Exception as e:
                self._log(f"UsbCamera: Exception releasing capture - {e}", "warning")

    def _capture_loop(self, read_interval: float = 0.01):
        """Background thread reading frames into self._last_frame."""
        self._log("UsbCamera: Capture thread starting.")
        retry_counter = 0

        try:
            while not self._stop_event.is_set():
                need_open = not self._is_capture_open()

                if need_open:
                    opened = self._open_capture()
                    if not opened:
                        if self.auto_reconnect and not self._stop_event.is_set():
                            self._log("UsbCamera: Capture not open, retrying in 0.5s.", "warning")
                            self._stop_event.wait(0.5)
                            continue
                        self._log("UsbCamera: Capture not open and auto_reconnect disabled.", "warning")
                        break

                with self._capture_lock:
                    cap_ref = self._cap
                    graph_ref = self._graph

                if cap_ref is None and graph_ref is None:
                    self._stop_event.wait(read_interval)
                    continue

                if graph_ref is not None:
                    try:
                        graph_ref.grab_frame()
                    except Exception as e:
                        self._log(f"UsbCamera: Exception during DirectShow frame grab - {e}", "warning")
                        self._close_capture()
                        if not self.auto_reconnect:
                            break
                    self._stop_event.wait(read_interval)
                    continue

                try:
                    ret, frame = cap_ref.read()
                    if not ret or frame is None:
                        retry_counter += 1
                        self._log("UsbCamera: Frame read returned empty, attempting reopen.", "warning")
                        self._close_capture()

                        if not self.auto_reconnect:
                            break

                        self._stop_event.wait(0.1)
                        if retry_counter > 10:
                            self._stop_event.wait(0.5)
                            retry_counter = 0
                        continue

                    retry_counter = 0
                    with self._capture_lock:
                        self._last_frame = frame
                        self._frame_timestamp = time.time()

                    if not self._started_event.is_set():
                        self._started_event.set()
                        self._set_state("RUNNING")
                        self._log("UsbCamera: First frame received; capture is ready.")

                except Exception as e:
                    self._log(f"UsbCamera: Exception during frame read - {e}", "warning")
                    self._close_capture()
                    if not self.auto_reconnect:
                        break

                self._stop_event.wait(read_interval)
        finally:
            self._close_capture()
            self._set_state("IDLE")
            self._stopped_event.set()
            self._log("UsbCamera: Capture thread stopped.")

    # ------------------------------------------------------------
    # Public capture control
    # ------------------------------------------------------------
    def start_capture(self, background: bool = True, wait_ready: bool = False, timeout: float = 12.0):
        """
        Start capture thread. Optionally wait for first frame readiness.
        Returns True when start request accepted.
        """
        del background  # kept for backward compatibility

        with self._capture_lock:
            if self._thread and self._thread.is_alive() and not self._stop_event.is_set():
                self._log("UsbCamera: Capture already running, start_capture no-op.")
                if wait_ready:
                    if not self._started_event.wait(timeout=timeout):
                        raise RuntimeError("Camera capture thread is running but not ready.")
                return True

            self._stop_event.clear()
            self._record_stop.clear()
            self._started_event.clear()
            self._stopped_event.clear()
            self._set_state("STARTING")

            self._thread = threading.Thread(target=self._capture_loop, name="UsbCamera-Capture", daemon=True)
            self._thread.start()

        self._log("UsbCamera: Capture started.")

        if wait_ready:
            if not self._started_event.wait(timeout=timeout):
                raise RuntimeError("Timeout waiting for first camera frame.")

        return True

    def stop_capture(self, join: bool = True, timeout: float = 5.0):
        """Stop capture thread and release device. Returns True when fully stopped."""
        self._log("UsbCamera: Stopping capture...")

        self._set_state("STOPPING")
        self._record_stop.set()
        self._stop_event.set()

        with self._capture_lock:
            thread_ref = self._thread

        if thread_ref and join:
            thread_ref.join(timeout=timeout)

        with self._capture_lock:
            thread_alive = bool(self._thread and self._thread.is_alive())

        if thread_alive:
            self._log("UsbCamera: Capture thread did not stop within timeout.", "warning")
            # Best-effort release path.
            self._close_capture()
            return False

        self._close_capture()
        self._stopped_event.set()

        with self._capture_lock:
            self._thread = None

        self._set_state("IDLE")
        self._log("UsbCamera: Capture stopped.")
        return True

    def is_capturing(self) -> bool:
        """Return True if capture thread is alive and stop event not set."""
        with self._capture_lock:
            return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    # ------------------------------------------------------------
    # Image operations
    # ------------------------------------------------------------
    def capture_image(self, save_path: Optional[str] = None, convert_rgb: bool = True) -> str:
        """
        Save the latest captured frame to disk and return the file path.
        If save_path is a directory, it will automatically append a filename.
        """
        with self._capture_lock:
            if self._last_frame is None:
                raise RuntimeError("No frame available to capture (camera not started or no frame yet).")
            frame = self._last_frame.copy()

        if convert_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        if not save_path:
            save_path = os.path.join(self.image_save_path, f"usb_{self.camera_index}_{self._timestamp()}.jpg")
        elif os.path.isdir(save_path):
            filename = f"usb_{self.camera_index}_{self._timestamp()}.jpg"
            save_path = os.path.join(save_path, filename)
        elif not Path(save_path).suffix:
            save_path = f"{save_path}.jpg"

        img.save(save_path, quality=90)
        self._last_image_path = save_path
        self._log(f"UsbCamera: Captured image saved to {save_path}.")
        return save_path

    def get_last_image_path(self) -> Optional[str]:
        """Return path of last saved image (if any)."""
        return self._last_image_path

    def get_last_frame(self) -> Optional[Tuple[bytes, float]]:
        """
        Return a tuple (numpy_bgr_frame, timestamp) of last frame if any,
        or None if no frame.
        """
        with self._capture_lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy(), self._frame_timestamp

    # ------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------
    def record_video(
        self,
        output_path: Optional[str] = None,
        duration: Optional[float] = None,
        fps: Optional[float] = None,
        codec: str = "mp4v",
    ):
        """
        Record video from live frames for given duration (seconds)
        or until stop_recording() is called.
        """

        if not self.is_capturing():
            raise RuntimeError("Camera is not capturing. Call start_capture() first.")

        if self._last_frame is None:
            raise RuntimeError("No frame available for recording yet.")

        self._record_stop.clear()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if output_path is None:
            output_path = os.path.join(self.video_save_path, f"usb_{self.camera_index}_{timestamp}.mp4")
        else:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"usb_{self.camera_index}_{timestamp}.mp4")
            elif not Path(output_path).suffix:
                output_path = f"{output_path}_{timestamp}.mp4"

        with self._capture_lock:
            if self._cap and self._cap.isOpened():
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_fps = self._cap.get(cv2.CAP_PROP_FPS) or 20.0
            else:
                h, w = self._last_frame.shape[:2]
                cap_fps = 20.0

        fps = fps or cap_fps or 20.0
        frame_interval = 1.0 / fps

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self._log(f"UsbCamera: Recording video to {output_path} ({w}x{h}@{fps:.1f})")

        start_time = time.time()
        next_frame_time = start_time

        try:
            while True:
                if duration and (time.time() - start_time) >= duration:
                    break

                if self._record_stop.is_set() or self._stop_event.is_set():
                    break

                with self._capture_lock:
                    frame = self._last_frame.copy() if self._last_frame is not None else None

                if frame is not None:
                    out.write(frame)

                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    if self._record_stop.wait(timeout=sleep_time):
                        break

        finally:
            out.release()
            self._log("UsbCamera: Video recording finished.")

        return output_path

    def stop_recording(self):
        """Immediately stop current video recording."""
        self._log("UsbCamera: Stopping recording...")
        self._record_stop.set()

    # ------------------------------------------------------------
    def stream_frames(self, width: int = 1280, height: int = 720, fps: float = 30):
        """Simple generator that streams frames for FastAPI StreamingResponse."""
        if self.is_capturing():
            try:
                start = time.time()
                while self._last_frame is None:
                    if time.time() - start > 8:
                        raise RuntimeError(f"UsbCamera: Timeout waiting for live frame on camera {self.camera_index}")
                    time.sleep(0.1)

                self._log(f"UsbCamera: Stream using existing capture (index {self.camera_index}).")
                while True:
                    with self._capture_lock:
                        frame = self._last_frame.copy() if self._last_frame is not None else None
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    ok, jpeg = cv2.imencode(".jpg", frame)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                    time.sleep(1.0 / fps if fps > 0 else 0.03)
            except GeneratorExit:
                self._log(f"UsbCamera: Stream stopped (existing capture, index {self.camera_index}).")
            except Exception as e:
                self._log(f"UsbCamera: Stream error - {e}", "error")
            return

        cap = None
        backend_name = None
        try:
            for candidate_backend, _backend_id, open_ref, candidate_cap in self._open_cv_capture_candidates(self.camera_index):
                if candidate_cap.isOpened():
                    cap = candidate_cap
                    backend_name = candidate_backend
                    self._log(
                        f"UsbCamera: Stream opened camera {self.camera_index!r} "
                        f"with {candidate_backend} ({open_ref!r})."
                    )
                    break
                try:
                    candidate_cap.release()
                except Exception:
                    pass

            if not cap or not cap.isOpened():
                raise RuntimeError(f"Unable to open camera {self.camera_index!r}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            start = time.time()
            while True:
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                if time.time() - start > 8:
                    raise RuntimeError(f"UsbCamera: Timeout waiting for first frame on camera {self.camera_index}")
                time.sleep(0.1)

            self._log(f"UsbCamera: Stream started (index {self.camera_index}, backend {backend_name}).")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
                ok, jpeg = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(1.0 / fps if fps > 0 else 0.03)

        except GeneratorExit:
            self._log(f"UsbCamera: Stream stopped (index {self.camera_index}).")
        except Exception as e:
            self._log(f"UsbCamera: Stream error - {e}", "error")
        finally:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._log(f"UsbCamera: Camera {self.camera_index} released after stream.")

    # ------------------------------------------------------------
    # Utility / properties
    # ------------------------------------------------------------
    def set_resolution(self, width: int, height: int):
        """Attempt to set capture resolution on the device."""
        with self._capture_lock:
            if self._graph is not None and not self._cap:
                self._log("UsbCamera: Resolution changes are not supported for DirectShow graph fallback.", "warning")
                return False
            if not self._cap or not self._cap.isOpened():
                self._log("UsbCamera: Capture not opened - attempting to open to set resolution.")
        if not self._cap or not self._cap.isOpened():
            opened = self._open_capture()
            if not opened:
                raise RuntimeError("Unable to open camera to set resolution.")

        with self._capture_lock:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        time.sleep(0.05)
        self._log(f"UsbCamera: Requested resolution set to {width}x{height}.")
        return True

    def get_resolution(self) -> Optional[Tuple[int, int]]:
        """Return current capture resolution or None if not available."""
        with self._capture_lock:
            if not self._cap or not self._cap.isOpened():
                if self._last_frame is not None:
                    h, w = self._last_frame.shape[:2]
                    return (w, h)
                return None
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

    def set_property(self, prop_id: int, value: float):
        """Set arbitrary OpenCV property on the capture (e.g., cv2.CAP_PROP_EXPOSURE)."""
        with self._capture_lock:
            if not self._cap or not self._cap.isOpened():
                raise RuntimeError("Camera capture not opened.")
            ok = self._cap.set(prop_id, value)
            if not ok:
                self._log(f"UsbCamera: Failed to set property {prop_id} -> {value}.", "warning")
            return ok

    def get_property(self, prop_id: int) -> Optional[float]:
        """Get arbitrary OpenCV property from the capture."""
        with self._capture_lock:
            if not self._cap or not self._cap.isOpened():
                return None
            return float(self._cap.get(prop_id))

    def _timestamp(self):
        return time.strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    def release(self):
        """Release resources (stop thread + release device)."""
        self._log("UsbCamera: Releasing resources...")
        self.stop_capture(join=True, timeout=5.0)
        with self._capture_lock:
            self._last_frame = None
            self._last_image_path = None
        self._log("UsbCamera: Released.")

    # ------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------
    def __enter__(self):
        self.start_capture(wait_ready=True, timeout=5.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

