import cv2

for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW]:
    cap = cv2.VideoCapture(0, backend)
    print(f"Trying backend {backend}...")
    ok, frame = cap.read()
    print("Opened:", cap.isOpened(), "Frame:", ok, "Shape:", None if frame is None else frame.shape)
    cap.release()
