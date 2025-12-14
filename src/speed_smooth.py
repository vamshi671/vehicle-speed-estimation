from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
METERS_PER_PIXEL = 0.05  # approximate scale
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "outputs/speed_output.mp4",
    fourcc,
    fps,
    (int(cap.get(3)), int(cap.get(4)))
)


# Track history
track_history = {}

# Speed buffer for smoothing
speed_buffer = {}   # track_id -> deque
BUFFER_SIZE = 5     # last 5 frames

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
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if track_id not in track_history:
            track_history[track_id] = []
            speed_buffer[track_id] = deque(maxlen=BUFFER_SIZE)

        track_history[track_id].append((cx, cy))

        if len(track_history[track_id]) >= 2:
            (x_prev, y_prev) = track_history[track_id][-2]
            (x_curr, y_curr) = track_history[track_id][-1]

            pixel_dist = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
            speed_mps = pixel_dist * METERS_PER_PIXEL * fps
            speed_kmph = speed_mps * 3.6

            speed_buffer[track_id].append(speed_kmph)
            smooth_speed = int(np.mean(speed_buffer[track_id]))

            cv2.putText(
                frame,
                f"{smooth_speed} km/h",
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    annotated = results[0].plot()
    out.write(annotated)

    cv2.imshow("Smoothed Speed Estimation", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
