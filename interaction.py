"""
Human-Object Interaction via proximity.
Associates jewellery track IDs with person track IDs.
No hand detection yet — pure spatial reasoning.
"""

import numpy as np

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def bbox_iou(boxA, boxB):
    """IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def point_in_bbox(point, bbox, margin=0):
    """Check if point is inside bbox with optional margin."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - margin <= x <= x2 + margin and
            y1 - margin <= y <= y2 + margin)

def associate_jewellery_to_person(
    jewellery_bbox,
    person_bboxes,
    person_ids,
    proximity_margin=50,
    iou_threshold=0.01
):
    """
    Find the person most likely holding this jewellery.

    Strategy:
    1. If jewellery center is inside person bbox → strong match.
    2. If jewellery bbox overlaps person bbox (IoU > threshold) → match.
    3. If jewellery center is within `proximity_margin` of person bbox → weak match.

    Returns: (person_id, confidence) or (None, 0)
    """
    jcx, jcy = bbox_center(jewellery_bbox)

    best_id = None
    best_score = 0

    for pid, pbbox in zip(person_ids, person_bboxes):
        # Priority 1: center inside person bbox
        if point_in_bbox((jcx, jcy), pbbox):
            return pid, 1.0

        # Priority 2: IoU
        iou = bbox_iou(jewellery_bbox, pbbox)
        if iou > iou_threshold:
            score = 0.5 + iou
            if score > best_score:
                best_score = score
                best_id = pid

        # Priority 3: proximity to bbox
        # Expand person bbox by margin
        px1, py1, px2, py2 = pbbox
        expanded = (px1 - proximity_margin, py1 - proximity_margin,
                    px2 + proximity_margin, py2 + proximity_margin)
        if point_in_bbox((jcx, jcy), expanded):
            # Distance to person center
            pcx, pcy = bbox_center(pbbox)
            dist = np.sqrt((jcx - pcx)**2 + (jcy - pcy)**2)
            score = 1.0 / (1.0 + dist / 100.0)  # closer = higher
            if score > best_score:
                best_score = score
                best_id = pid

    if best_id is not None and best_score > 0.3:
        return best_id, best_score

    return None, 0