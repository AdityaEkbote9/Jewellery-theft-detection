import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

# =====================================================================
# 1. COMPOSITE FINGERPRINT EXTRACTOR (Color + Texture + Deep Features)
# =====================================================================
class JewelleryFingerprintExtractor(nn.Module):
    """
    Creates a unique visual signature invariant to folding, deformation, 
    and shape morphing by combining Deep Neural Features with HSV Color Histograms.
    """
    def __init__(self):
        super(JewelleryFingerprintExtractor, self).__init__()
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_hsv_histogram(self, crop):
        """Extracts color/reflection fingerprint invariant to shape/folding."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Compute 2D Hue-Saturation Histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def forward(self, crop):
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        # 1. Deep Feature Embedding
        tensor = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            x = self.features(tensor)
            x = self.pool(x)
            deep_emb = torch.flatten(x, 1).cpu().numpy()[0]
            deep_norm = np.linalg.norm(deep_emb)
            deep_emb = deep_emb / deep_norm if deep_norm > 0 else deep_emb

        # 2. Color/Metallic Histogram
        color_emb = self.get_hsv_histogram(crop)

        # Concatenate into unified 544-D Fingerprint
        composite_fingerprint = np.hstack((deep_emb * 0.5, color_emb * 0.5))
        norm = np.linalg.norm(composite_fingerprint)
        return composite_fingerprint / norm if norm > 0 else composite_fingerprint

# Initialize Extractor & Detection Model
device = "cuda" if torch.cuda.is_available() else "cpu"
fingerprint_engine = JewelleryFingerprintExtractor().to(device)
jewellery_model = YOLO("best.pt")

# =====================================================================
# 2. PERMANENT RE-ID GALLERY & STATE REGISTRY
# =====================================================================
class PermanentJewelleryGallery:
    def __init__(self, match_threshold=0.68):
        self.match_threshold = match_threshold
        # { persistent_id: {"gallery": [fingerprints], "last_seen": timestamp, "active": bool} }
        self.registered_gallery = {}
        # Mapping from active spatial tracker ID -> Persistent Master ID
        self.active_tracker_map = {}
        self.next_master_id = 1

    def process_frame(self, current_detections):
        """
        current_detections: dict of { raw_track_id: {"crop": np_array, "box": (x1,y1,x2,y2)} }
        """
        resolved_outputs = {}
        unmatched_raw_ids = []
        unmatched_fingerprints = []

        # STEP 1: Process items currently tracked frame-to-frame by ByteTrack
        for raw_id, det in current_detections.items():
            if raw_id in self.active_tracker_map:
                # Lock onto already assigned Master ID (ZERO ID FLICKERING WHILE VISIBLE)
                master_id = self.active_tracker_map[raw_id]
                resolved_outputs[master_id] = det["box"]
                
                # Update fingerprint gallery for this item
                fp = fingerprint_engine.forward(det["crop"])
                if fp is not None:
                    self.registered_gallery[master_id]["gallery"].append(fp)
                    if len(self.registered_gallery[master_id]["gallery"]) > 15:
                        self.registered_gallery[master_id]["gallery"].pop(0)
            else:
                unmatched_raw_ids.append(raw_id)
                fp = fingerprint_engine.forward(det["crop"])
                unmatched_fingerprints.append(fp)

        # STEP 2: If new/reappearing items are present, match against ALL registered IDs
        if unmatched_raw_ids and self.registered_gallery:
            valid_indices = [i for i, fp in enumerate(unmatched_fingerprints) if fp is not None]
            
            if valid_indices:
                registered_ids = list(self.registered_gallery.keys())
                
                # Build mean fingerprint vector for every registered item
                registered_means = np.array([
                    np.mean(self.registered_gallery[mid]["gallery"], axis=0) 
                    for mid in registered_ids
                ])
                
                candidate_fps = np.array([unmatched_fingerprints[i] for i in valid_indices])

                # Distance matrix calculation (1.0 - Cosine Similarity)
                cost_matrix = cdist(candidate_fps, registered_means, metric='cosine')

                # Solve global 1-to-1 match using Hungarian Algorithm
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_candidate_set = set()

                for r_idx, c_idx in zip(row_ind, col_ind):
                    cost = cost_matrix[r_idx, c_idx]
                    similarity = 1.0 - cost
                    
                    if similarity >= self.match_threshold:
                        # RE-IDENTIFIED OLD ITEM (Restoring previous Master ID!)
                        orig_candidate_idx = valid_indices[r_idx]
                        raw_id = unmatched_raw_ids[orig_candidate_idx]
                        master_id = registered_ids[c_idx]

                        self.active_tracker_map[raw_id] = master_id
                        resolved_outputs[master_id] = current_detections[raw_id]["box"]
                        
                        self.registered_gallery[master_id]["gallery"].append(candidate_fps[r_idx])
                        matched_candidate_set.add(orig_candidate_idx)

                # Register remaining unmatched items as BRAND NEW MASTER IDs
                for i, raw_id in enumerate(unmatched_raw_ids):
                    if i not in matched_candidate_set:
                        master_id = self.next_master_id
                        self.next_master_id += 1
                        
                        self.active_tracker_map[raw_id] = master_id
                        resolved_outputs[master_id] = current_detections[raw_id]["box"]
                        
                        fp = unmatched_fingerprints[i]
                        self.registered_gallery[master_id] = {
                            "gallery": [fp] if fp is not None else [],
                            "registered_at": time.time()
                        }
        elif unmatched_raw_ids:
            # First time running: Register initial detections
            for i, raw_id in enumerate(unmatched_raw_ids):
                master_id = self.next_master_id
                self.next_master_id += 1
                
                self.active_tracker_map[raw_id] = master_id
                resolved_outputs[master_id] = current_detections[raw_id]["box"]
                
                fp = unmatched_fingerprints[i]
                self.registered_gallery[master_id] = {
                    "gallery": [fp] if fp is not None else [],
                    "registered_at": time.time()
                }

        return resolved_outputs

# =====================================================================
# 3. WEBCAM EXECUTION LOOP
# =====================================================================
gallery_manager = PermanentJewelleryGallery(match_threshold=0.68)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Layer 1: Run ByteTrack for frame-level spatial continuity
    results = jewellery_model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        conf=0.45, 
        imgsz=1024, 
        verbose=False
    )[0]

    current_detections = {}

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        raw_ids = results.boxes.id.int().cpu().numpy()

        for box, r_id in zip(boxes, raw_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop image region safely inside frame boundaries
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            current_detections[r_id] = {
                "box": (x1, y1, x2, y2),
                "crop": crop
            }

    # Layer 2: Resolve Master Permanent IDs
    active_master_items = gallery_manager.process_frame(current_detections)

    # Visualization
    for master_id, box in active_master_items.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Jewellery ID #{master_id}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display HUD showing total unique registered items in memory
    cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Registered Gallery Items: {len(gallery_manager.registered_gallery)}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Permanent Jewellery Visual Fingerprint Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()