import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import time

# =====================================================================
# 1. OPTIMIZED DINOv2 TRANSFORMER ENGINE (FP16 Support)
# =====================================================================
class DINOv2FeatureExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[INFO] Loading DINOv2 Model on {self.device} (Optimized)...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.model.eval()
        
        # Enable half-precision (FP16) if on GPU for major speed boost
        if self.device == "cuda":
            self.model.half()
            print("[INFO] Enabled FP16 Half-Precision for DINOv2.")

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
        
        if self.device == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            embedding = self.model(tensor).float().cpu().numpy()[0]
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

# =====================================================================
# 2. PERSISTENT GALLERY MANAGER
# =====================================================================
class PersistentJewelleryGallery:
    def __init__(self, reid_threshold=0.55, max_snapshots=15):
        self.reid_threshold = reid_threshold
        self.max_snapshots = max_snapshots
        self.extractor = DINOv2FeatureExtractor()
        
        self.master_gallery = {}
        self.raw_to_master = {}
        self.next_master_id = 1

    def update(self, current_detections, current_time):
        current_raw_ids = set(current_detections.keys())
        active_master_ids = set()
        resolved_frame = {}
        unmatched_raw_ids = []

        for raw_id, det in current_detections.items():
            if raw_id in self.raw_to_master:
                master_id = self.raw_to_master[raw_id]
                active_master_ids.add(master_id)
                resolved_frame[master_id] = det["box"]

                box_area = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
                if box_area > 1500:
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

        if unmatched_raw_ids:
            candidate_raws = []
            candidate_embs = []

            for r_id in unmatched_raw_ids:
                emb = self.extractor.extract(current_detections[r_id]["crop"])
                if emb is not None:
                    candidate_raws.append(r_id)
                    candidate_embs.append(emb)

            inactive_master_ids = [
                m_id for m_id, data in self.master_gallery.items() 
                if m_id not in active_master_ids and data["anchor"] is not None
            ]

            if candidate_embs and inactive_master_ids:
                candidate_embs = np.array(candidate_embs)
                sim_matrix = np.zeros((len(candidate_raws), len(inactive_master_ids)))

                for i, c_emb in enumerate(candidate_embs):
                    for j, m_id in enumerate(inactive_master_ids):
                        data = self.master_gallery[m_id]
                        anchor = data["anchor"]
                        snapshots = data["snapshots"]

                        anchor_sim = np.dot(anchor, c_emb)
                        snap_sims = [np.dot(s, c_emb) for s in snapshots] if snapshots else [anchor_sim]
                        
                        max_sim = (0.7 * anchor_sim) + (0.3 * max(snap_sims))
                        sim_matrix[i, j] = max_sim

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

                for i, r_id in enumerate(candidate_raws):
                    if i not in matched_candidate_indices:
                        master_id = self.next_master_id
                        self.next_master_id += 1

                        emb = candidate_embs[i]
                        self.raw_to_master[r_id] = master_id
                        active_master_ids.add(master_id)
                        resolved_frame[master_id] = current_detections[r_id]["box"]

                        self.master_gallery[master_id] = {
                            "anchor": emb,
                            "snapshots": [emb],
                            "last_seen": current_time,
                            "active": True
                        }
            else:
                for r_id in unmatched_raw_ids:
                    emb = self.extractor.extract(current_detections[r_id]["crop"])
                    master_id = self.next_master_id
                    self.next_master_id += 1

                    self.raw_to_master[r_id] = master_id
                    active_master_ids.add(master_id)
                    resolved_frame[master_id] = current_detections[r_id]["box"]

                    self.master_gallery[master_id] = {
                        "anchor": emb,
                        "snapshots": [emb] if emb is not None else [],
                        "last_seen": current_time,
                        "active": True
                    }

        for m_id in self.master_gallery:
            if m_id not in active_master_ids:
                self.master_gallery[m_id]["active"] = False

        return resolved_frame

# =====================================================================
# 3. WEBCAM EXECUTION LOOP (Optimized Resolution + FPS Counter)
# =====================================================================
yolo_model = YOLO("best.pt")
gallery = PersistentJewelleryGallery(reid_threshold=0.55)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Optimized resolution (imgsz=960) provides a huge speed boost over 1280
    results = yolo_model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        conf=0.20,       
        imgsz=960,       # Optimized balance between earring detection speed & accuracy
        verbose=False
    )[0]

    current_detections = {}

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        raw_ids = results.boxes.id.int().cpu().numpy()

        for box, r_id in zip(boxes, raw_ids):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            if w < 15 or h < 15 or w > 600 or h > 600:
                continue
            
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio > 10.0:
                continue

            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            current_detections[r_id] = {
                "box": (x1, y1, x2, y2),
                "crop": crop
            }

    active_master_items = gallery.update(current_detections, current_time)

    # Calculate real-time FPS
    fps = 1.0 / (time.time() - prev_time) if prev_time > 0 else 0.0
    prev_time = time.time()

    for master_id, box in active_master_items.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Jewellery #{master_id}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # HUD Overlay with FPS counter and Memory Stats
    cv2.rectangle(frame, (10, 10), (380, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Master IDs in Memory: {len(gallery.master_gallery)}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Optimized High-FPS DINOv2 Re-ID Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()