import math
import time

import cv2
import cvzone
from ultralytics import YOLO

# Set confidence threshold
confidence = 0.6

# Capture from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1000)  # Set width
cap.set(4, 800)  # Set height
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For video file input

# Load YOLO model
model = YOLO("../models/yolov8n.pt")

# List of class names for the YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair dryer", "toothbrush"]

prev_frame_time = 0
new_frame_time = 0

while True:
    # Calculate FPS
    new_frame_time = time.time()

    # Read frame from webcam
    success, img = cap.read()
    if not success:
        break

    # Perform detection
    results = model(img, stream=True, verbose = False)

    # Iterate through results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Draw rectangle with corners using cvzone.cornerRect()
            cvzone.cornerRect(img, (x1, y1, w, h), l=30, rt=2)  # Remove the color argument

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            className = classNames[cls]

            # Display text
            cvzone.putTextRect(img, f"{className} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Display image
    cv2.imshow("Image", img)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
