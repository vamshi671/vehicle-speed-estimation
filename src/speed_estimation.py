from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

# Approximate scale (we will improve later)
METERS_PER_PIXEL = 0.05

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

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        # Calculate speed if enough points
        if len(track_history[track_id]) >= 2:
            (x_prev, y_prev) = track_history[track_id][-2]
            (x_curr, y_curr) = track_history[track_id][-1]

            pixel_dist = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
            speed_mps = pixel_dist * METERS_PER_PIXEL * fps
            speed_kmph = speed_mps * 3.6

            cv2.putText(
                frame,
                f"{int(speed_kmph)} km/h",
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    annotated = results[0].plot()
    cv2.imshow("Vehicle Speed Estimation", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
