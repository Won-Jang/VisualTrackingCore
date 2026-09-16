import cv2

for index in range(10):
    cap = cv2.VideoCapture(index)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            h, w = frame.shape[:2]
            print(f"Camera index {index}: WORKING - {w}x{h}")
        else:
            print(f"Camera index {index}: opened, but no frame")

    cap.release()