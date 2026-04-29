import cv2
import time
from usbcamera import UsbCamera  # adjust the import path if needed

def main():
    print("🎥 Testing direct streaming from UsbCamera wrapper...\n")

    cam = UsbCamera(0)  # <-- test with your real camera index
    print(f"Initialized camera at index {cam.camera_index}")

    try:
        frame_gen = cam.stream_frames(fps=20)

        # Open OpenCV window
        cv2.namedWindow("USB Camera Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("USB Camera Test", 800, 600)

        for i, chunk in enumerate(frame_gen):
            # Decode the JPEG chunk (only frame bytes)
            try:
                # Extract raw JPEG bytes from multipart chunk
                start = chunk.find(b'\r\n\r\n') + 4
                frame_bytes = chunk[start:-2]
                frame = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                cv2.imshow("USB Camera Test", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Stream stopped by user.")
                    break

            except Exception as e:
                print(f"Decode error: {e}")
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted manually.")
    except Exception as e:
        print(f"❌ Stream error: {e}")
    finally:
        print("Releasing camera...")
        cv2.destroyAllWindows()
        cam.release()
        print("✅ Done.")

if __name__ == "__main__":
    import numpy as np
    main()
