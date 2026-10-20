import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from math import hypot
import time

# =====================================================================
# 1. VISUAL EMBEDDING EXTRACTOR (Re-ID Feature Engine)
# =====================================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def forward(self, crop):
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        tensor = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            x = self.features(tensor)
            x = self.pool(x)
            embedding = torch.flatten(x, 1).cpu().numpy()[0]
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

# Initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = FeatureExtractor().to(device)
jewellery_model = YOLO("best.pt")

# =====================================================================
# 2. STABLE HYBRID TRACKER & RE-ID MANAGER
# =====================================================================
class StableJewelleryTracker:
    def __init__(self, match_threshold=0.55, max_lost_time=5.0):
        self.match_threshold = match_threshold
        self.max_lost_time = max_lost_time
        
        # { raw_byte_track_id: persistent_id }
        self.raw_to_persistent = {}
        
        # Persistent Gallery:
        # { persistent_id: {"embeddings": [...], "last_center": (x,y), "last_seen": timestamp, "active": bool} }
        self.gallery = {}
        self.next_persistent_id = 1

    def update(self, current_tracks, frame_time):
        """
        current_tracks: dict of { raw_id: {"box": (x1,y1,x2,y2), "crop": np_array, "center": (cx,cy)} }
        """
        current_raw_ids = set(current_tracks.keys())
        active_persistent_ids = set()
        resolved_frame = {}

        # STEP 1: Process raw tracks that are ALREADY known (Continuous Tracking)
        unmatched_raw_ids = []
        
        for raw_id, data in current_tracks.items():
            if raw_id in self.raw_to_persistent:
                # STABLE TRACK: Lock to existing persistent ID without re-matching!
                p_id = self.raw_to_persistent[raw_id]
                active_persistent_ids.add(p_id)
                
                # Update gallery feature history
                emb = encoder.forward(data["crop"])
                if emb is not None:
                    self.gallery[p_id]["embeddings"].append(emb)
                    if len(self.gallery[p_id]["embeddings"]) > 10:
                        self.gallery[p_id]["embeddings"].pop(0)
                        
                self.gallery[p_id]["last_center"] = data["center"]
                self.gallery[p_id]["last_seen"] = frame_time
                self.gallery[p_id]["active"] = True
                
                resolved_frame[p_id] = data["box"]
            else:
                unmatched_raw_ids.append(raw_id)

        # STEP 2: Handle NEW raw tracks (Re-ID Matching against Lost/Inactive items)
        for raw_id in unmatched_raw_ids:
            data = current_tracks[raw_id]
            crop_emb = encoder.forward(data["crop"])
            
            best_match_id = None
            highest_score = -1.0

            if crop_emb is not None:
                # Check against all INACTIVE items in gallery lost within `max_lost_time`
                for p_id, p_data in self.gallery.items():
                    if p_id in active_persistent_ids:
                        continue # Cannot match an item currently active in the frame!
                        
                    time_diff = frame_time - p_data["last_seen"]
                    if time_diff > self.max_lost_time:
                        continue # Expired track

                    # 1. Cosine Visual Similarity
                    visual_sims = [np.dot(crop_emb, v) for v in p_data["embeddings"]]
                    avg_visual_sim = np.mean(visual_sims) if visual_sims else 0.0

                    # 2. Spatial Center Proximity Score
                    dist = hypot(data["center"][0] - p_data["last_center"][0], 
                                 data["center"][1] - p_data["last_center"][1])
                    spatial_score = max(0.0, 1.0 - (dist / 300.0)) # 300px max drop distance

                    # Composite Score (60% Visual, 40% Spatial Location)
                    composite_score = (0.6 * avg_visual_sim) + (0.4 * spatial_score)

                    if composite_score > highest_score:
                        highest_score = composite_score
                        best_match_id = p_id

            # Decide whether to restore ID or create a new ID
            if best_match_id is not None and highest_score >= self.match_threshold:
                # Restore Lost Persistent ID
                assigned_id = best_match_id
                self.raw_to_persistent[raw_id] = assigned_id
                
                if crop_emb is not None:
                    self.gallery[assigned_id]["embeddings"].append(crop_emb)
                    
                self.gallery[assigned_id]["last_center"] = data["center"]
                self.gallery[assigned_id]["last_seen"] = frame_time
                self.gallery[assigned_id]["active"] = True
            else:
                # Assign Brand New Persistent ID
                assigned_id = self.next_persistent_id
                self.next_persistent_id += 1
                
                self.raw_to_persistent[raw_id] = assigned_id
                self.gallery[assigned_id] = {
                    "embeddings": [crop_emb] if crop_emb is not None else [],
                    "last_center": data["center"],
                    "last_seen": frame_time,
                    "active": True
                }

            active_persistent_ids.add(assigned_id)
            resolved_frame[assigned_id] = data["box"]

        # STEP 3: Mark persistent IDs not visible in this frame as INACTIVE
        for p_id in self.gallery:
            if p_id not in active_persistent_ids:
                self.gallery[p_id]["active"] = False

        return resolved_frame

# =====================================================================
# 3. LIVE WEBCAM PIPELINE
# =====================================================================
tracker = StableJewelleryTracker(match_threshold=0.55, max_lost_time=5.0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_time = time.time()

    # Step A: Run Detection + Spatial ByteTrack
    results = jewellery_model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        conf=0.25, 
        imgsz=1024, 
        verbose=False
    )[0]

    current_tracks = {}

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        raw_ids = results.boxes.id.int().cpu().numpy()

        for box, r_id in zip(boxes, raw_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            current_tracks[r_id] = {
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "crop": crop
            }

    # Step B: Update Stable Tracker Engine
    resolved_items = tracker.update(current_tracks, frame_time)

    # Step C: Visualization
    for p_id, box in resolved_items.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Jewellery ID #{p_id}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imshow("Stabilized Real-Time Jewellery Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()