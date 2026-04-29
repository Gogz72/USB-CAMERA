from usbcamera import UsbCamera   # <-- replace with actual file name

def test_camera_names(max_test=10):
    print("🎥 Testing UsbCamera._get_camera_name()\n")

    cam = UsbCamera(camera_index=0, logger=None)

    print("Detected camera names:")
    for i in range(max_test):
        name = cam._get_camera_name(i)
        print(f"[{i}] {name}")

    print("\n✅ Test finished.")

if __name__ == "__main__":
    test_camera_names()
