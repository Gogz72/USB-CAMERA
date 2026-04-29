import argparse
import json

from usbcamera import UsbCamera


def main():
    parser = argparse.ArgumentParser(description="Diagnose Windows/DirectShow/OpenCV camera visibility.")
    parser.add_argument("--max-test", type=int, default=16, help="Highest numeric OpenCV index count to probe.")
    parser.add_argument("--timeout", type=float, default=1.5, help="Seconds to wait for a test frame per attempt.")
    args = parser.parse_args()

    report = UsbCamera.list_camera_details(max_test=args.max_test, read_timeout=args.timeout)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
