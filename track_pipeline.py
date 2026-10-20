"""
Multi-object tracking with YOLOv11 + ByteTrack.
Tracks persons AND jewellery with persistent IDs.
"""

import cv2
from ultralytics import YOLO
from collections import defaultdict
import json

# ---- CONFIG ----
MODEL_PATH = "best.pt"          # your trained model
VIDEO_SOURCE = 0                # 0 = webcam, or "video.mp4"
CONF_THRESHOLD = 0.35
TRACKER = "bytetrack.yaml"

# Your model's class names — adjust based on your dataset
# Common jewellery datasets: {0: "jewellery"} or {0: "person", 1: "jewellery"}
CLASS_NAMES = {0: "jewellery"}   # UPDATE THIS

# ---- INIT ----
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Track history: {track_id: [(frame, bbox, class, conf), ...]}
track_history = defaultdict(list)

# ---- MAIN LOOP ----
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, tid, cls, conf in zip(boxes, ids, clss, confs):
            x1, y1, x2, y2 = box
            label = f"{CLASS_NAMES.get(cls, cls)} ID:{tid} {conf:.2f}"

            # Draw
            color = (0, 255, 0) if CLASS_NAMES.get(cls) == "person" else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Store history
            track_history[tid].append({
                "frame": frame_idx,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class": CLASS_NAMES.get(cls, str(cls)),
                "conf": float(conf)
            })

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# Save track history
with open("track_history.json", "w") as f:
    json.dump({str(k): v for k, v in track_history.items()}, f, indent=2)

print(f"Processed {frame_idx} frames. Tracked {len(track_history)} objects.")