import math
import time
import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.8
resolution = (680, 520)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Camera not opened.")
    cap.set(3, resolution[0])
    cap.set(4, resolution[1])
except Exception as e:
    print(f"Error: {e}")
    exit()

model = YOLO("../models/l_version_1_300.pt")
classNames = ["fake", "real"]

prev_frame_time = 0

while True:
    start_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to grab a frame from the camera.")
        break

    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf >= confidence:
                color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
