import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

# =====================================================================
# 1. DINOv2 TRANSFORMER EMBEDDING ENGINE
# =====================================================================
class DINOv2FeatureExtractor:
    """
    Uses Meta's DINOv2 (ViT-S/14) for shape-agnostic, fine-grained visual Re-ID.
    Captures intrinsic material texture and link structures instead of shape outlines.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[INFO] Loading DINOv2 Model on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, crop):
        if crop is None or crop.shape[0] < 15 or crop.shape[1] < 15:
            return None

        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy()[0]
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

# =====================================================================
# 2. MULTI-VIEW PERSISTENT GALLERY MANAGER (SAFE DINOv2 MATCHING)
# =====================================================================
class PersistentJewelleryGallery:
    def __init__(self, reid_threshold=0.60, max_snapshots=20):
        self.reid_threshold = reid_threshold
        self.max_snapshots = max_snapshots
        self.extractor = DINOv2FeatureExtractor()
        
        # { master_id: {"snapshots": [emb1, emb2, ...], "last_seen": timestamp, "active": bool} }
        self.master_gallery = {}
        # { raw_track_id: master_id }
        self.raw_to_master = {}
        self.next_master_id = 1

    def update(self, current_detections, current_time):
        current_raw_ids = set(current_detections.keys())
        active_master_ids = set()
        resolved_frame = {}
        unmatched_raw_ids = []

        # -------------------------------------------------------------
        # STEP 1: Process Currently Tracked Items (Frame Continuity)
        # -------------------------------------------------------------
        for raw_id, det in current_detections.items():
            if raw_id in self.raw_to_master:
                master_id = self.raw_to_master[raw_id]
                active_master_ids.add(master_id)
                resolved_frame[master_id] = det["box"]

                # Append snapshots continuously while visible
                emb = self.extractor.extract(det["crop"])
                if emb is not None:
                    gallery_item = self.master_gallery[master_id]
                    gallery_item["snapshots"].append(emb)
                    if len(gallery_item["snapshots"]) > self.max_snapshots:
                        gallery_item["snapshots"].pop(0)
                    gallery_item["last_seen"] = current_time
                    gallery_item["active"] = True
            else:
                unmatched_raw_ids.append(raw_id)

        # -------------------------------------------------------------
        # STEP 2: Match New / Re-appearing Items Against Memory Bank
        # -------------------------------------------------------------
        if unmatched_raw_ids:
            candidate_raws = []
            candidate_embs = []

            for r_id in unmatched_raw_ids:
                emb = self.extractor.extract(current_detections[r_id]["crop"])
                if emb is not None:
                    candidate_raws.append(r_id)
                    candidate_embs.append(emb)

            # Retrieve master IDs that have valid snapshots in gallery
            inactive_master_ids = [
                m_id for m_id, data in self.master_gallery.items() 
                if m_id not in active_master_ids and len(data["snapshots"]) > 0
            ]

            if candidate_embs and inactive_master_ids:
                candidate_embs = np.array(candidate_embs)
                sim_matrix = np.zeros((len(candidate_raws), len(inactive_master_ids)))

                for i, c_emb in enumerate(candidate_embs):
                    for j, m_id in enumerate(inactive_master_ids):
                        snapshots = self.master_gallery[m_id]["snapshots"]
                        if not snapshots:
                            sim_matrix[i, j] = 0.0
                            continue
                        
                        snapshots_arr = np.array(snapshots)
                        similarities = np.dot(snapshots_arr, c_emb)
                        sim_matrix[i, j] = np.max(similarities)

                # Solve global bipartite assignment via Hungarian Algorithm
                cost_matrix = 1.0 - sim_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_candidate_indices = set()

                for r_idx, c_idx in zip(row_ind, col_ind):
                    sim = sim_matrix[r_idx, c_idx]
                    
                    if sim >= self.reid_threshold:
                        raw_id = candidate_raws[r_idx]
                        master_id = inactive_master_ids[c_idx]

                        self.raw_to_master[raw_id] = master_id
                        active_master_ids.add(master_id)
                        resolved_frame[master_id] = current_detections[raw_id]["box"]

                        self.master_gallery[master_id]["snapshots"].append(candidate_embs[r_idx])
                        self.master_gallery[master_id]["last_seen"] = current_time
                        self.master_gallery[master_id]["active"] = True

                        matched_candidate_indices.add(r_idx)

                # Register unmatched candidates as new Master IDs
                for i, r_id in enumerate(candidate_raws):
                    if i not in matched_candidate_indices:
                        master_id = self.next_master_id
                        self.next_master_id += 1

                        self.raw_to_master[r_id] = master_id
                        active_master_ids.add(master_id)
                        resolved_frame[master_id] = current_detections[r_id]["box"]

                        self.master_gallery[master_id] = {
                            "snapshots": [candidate_embs[i]],
                            "last_seen": current_time,
                            "active": True
                        }

            else:
                # First time seeing items or no valid historical gallery
                for r_id in unmatched_raw_ids:
                    emb = self.extractor.extract(current_detections[r_id]["crop"])
                    master_id = self.next_master_id
                    self.next_master_id += 1

                    self.raw_to_master[r_id] = master_id
                    active_master_ids.add(master_id)
                    resolved_frame[master_id] = current_detections[r_id]["box"]

                    self.master_gallery[master_id] = {
                        "snapshots": [emb] if emb is not None else [],
                        "last_seen": current_time,
                        "active": True
                    }

        # Update activity state
        for m_id in self.master_gallery:
            if m_id not in active_master_ids:
                self.master_gallery[m_id]["active"] = False

        return resolved_frame

# =====================================================================
# 3. WEBCAM EXECUTION LOOP
# =====================================================================
yolo_model = YOLO("best.pt")
gallery = PersistentJewelleryGallery(reid_threshold=0.60)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    results = yolo_model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        conf=0.25, 
        imgsz=1024, 
        verbose=False
    )[0]

    current_detections = {}

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        raw_ids = results.boxes.id.int().cpu().numpy()

        for box, r_id in zip(boxes, raw_ids):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            current_detections[r_id] = {
                "box": (x1, y1, x2, y2),
                "crop": crop
            }

    active_master_items = gallery.update(current_detections, current_time)

    for master_id, box in active_master_items.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Jewellery #{master_id}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.rectangle(frame, (10, 10), (380, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Master IDs in Memory: {len(gallery.master_gallery)}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("DINOv2 Re-ID Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()