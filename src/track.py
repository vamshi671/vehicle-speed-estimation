from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

# Dictionary: track_id → list of (x, y) points
track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        conf=0.4
    )

    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id)
        x1, y1, x2, y2 = box.xyxy[0]

        # Center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        # Draw center point
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    annotated = results[0].plot()
    cv2.imshow("Tracking + Center Points", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
