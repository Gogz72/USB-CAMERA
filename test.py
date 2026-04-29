# -------------------------
# Example usage (commented)
# -------------------------
from usbcamera import UsbCamera
import time
if __name__ == "__main__":
     cam = UsbCamera()                 # auto-detect index
     cam.start_capture()
     time.sleep(0.5)                   # wait for frames
     print(cam.get_status())
     path = cam.capture_image(r"D:\Work\Finway local\Developed tools\USB Camera")        # saves to temp file
     print("Saved image:", path)
     cam.record_video("test.mp4", duration=5, fps=20)
     cam.release()
