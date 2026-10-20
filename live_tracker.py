import cv2
import numpy as np
from ultralytics import YOLO
from math import hypot

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
# Load standard COCO model for Person Detection (Class 0)
person_model = YOLO("yolo11n.pt") 

# Load custom model fine-tuned on 45k jewellery images
# Replace with path to your trained weights
jewellery_model = YOLO("best.pt") 

# Distance threshold in pixels to consider "Handling"
INTERACTION_DISTANCE_PX = 150 

class ArticleState:
    DISPLAY = "DISPLAY"
    HANDLED = "HANDLED"
    ALERT = "ALERT (UNACCOUNTED)"

# State repository for tracked jewellery: { jewellery_id: {"state": ..., "holder": ...} }
article_states = {}

def get_box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# ==========================================
# 2. WEBCAM EXECUTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------------------------
    # STEP A: MULTI-OBJECT TRACKING (ByteTrack)
    # --------------------------------------
    # 1. Track Persons at standard resolution (640)
    person_results = person_model.track(
        frame, 
        persist=True, 
        classes=[0], 
        tracker="bytetrack.yaml", 
        imgsz=640, 
        verbose=False
    )
    
    # 2. Track Jewellery at HIGH resolution (1024) to capture small objects
    jewellery_results = jewellery_model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        imgsz=1024, 
        conf=0.30,  # Lower confidence threshold slightly for small items
        verbose=False
    )

    tracked_persons = {}
    tracked_jewellery = {}

    # Parse tracked person bounding boxes and centroids
    if person_results[0].boxes is not None and person_results[0].boxes.id is not None:
        boxes = person_results[0].boxes.xyxy.cpu().numpy()
        track_ids = person_results[0].boxes.id.int().cpu().numpy()
        for box, track_id in zip(boxes, track_ids):
            tracked_persons[track_id] = {
                "box": box,
                "center": get_box_center(box)
            }

    # Parse tracked jewellery bounding boxes and centroids
    if jewellery_results[0].boxes is not None and jewellery_results[0].boxes.id is not None:
        boxes = jewellery_results[0].boxes.xyxy.cpu().numpy()
        track_ids = jewellery_results[0].boxes.id.int().cpu().numpy()
        for box, track_id in zip(boxes, track_ids):
            tracked_jewellery[track_id] = {
                "box": box,
                "center": get_box_center(box)
            }

    # --------------------------------------
    # STEP B: HUMAN-OBJECT INTERACTION (HOI) & STATE MACHINE
    # --------------------------------------
    for j_id, j_data in tracked_jewellery.items():
        j_center = j_data["center"]
        closest_person_id = None
        min_distance = float("inf")

        # Find nearest person to the jewellery item
        for p_id, p_data in tracked_persons.items():
            p_center = p_data["center"]
            dist = hypot(j_center[0] - p_center[0], j_center[1] - p_center[1])
            if dist < min_distance:
                min_distance = dist
                closest_person_id = p_id

        # Update State Machine Logic
        if closest_person_id is not None and min_distance <= INTERACTION_DISTANCE_PX:
            state = ArticleState.HANDLED
            holder = f"Person #{closest_person_id}"
        else:
            state = ArticleState.DISPLAY
            holder = "None"

        article_states[j_id] = {
            "state": state,
            "holder": holder,
            "last_seen_center": j_center
        }

        # Draw connecting vector for active Human-Object Interaction
        if state == ArticleState.HANDLED and closest_person_id in tracked_persons:
            p_center = tracked_persons[closest_person_id]["center"]
            cv2.line(frame, j_center, p_center, (0, 255, 255), 2)  # Yellow link line

    # --------------------------------------
    # STEP C: OPENCV VISUALIZATION & OVERLAY
    # --------------------------------------
    # Draw Person Bounding Boxes
    for p_id, p_data in tracked_persons.items():
        x1, y1, x2, y2 = map(int, p_data["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, f"Person #{p_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Draw Jewellery Bounding Boxes and State Overlays
    for j_id, j_data in tracked_jewellery.items():
        x1, y1, x2, y2 = map(int, j_data["box"])
        state_info = article_states.get(j_id, {"state": ArticleState.DISPLAY, "holder": "None"})
        
        color = (0, 255, 0) if state_info["state"] == ArticleState.DISPLAY else (0, 165, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Jewellery #{j_id} [{state_info['state']}]", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Heads-Up Display (HUD) Dashboard
    cv2.rectangle(frame, (10, 10), (380, 40 + len(article_states) * 25), (0, 0, 0), -1)
    cv2.putText(frame, "LIVE ARTICLE STATE TRACKER", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset = 55
    for j_id, info in article_states.items():
        hud_text = f"Item #{j_id}: {info['state']} | Holder: {info['holder']}"
        cv2.putText(frame, hud_text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 25

    cv2.imshow("CCTV Theft Prevention - MOT & Interaction Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()