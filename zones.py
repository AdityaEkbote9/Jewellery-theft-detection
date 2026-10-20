"""
Zone definition and point-in-polygon checking.
Zones are manually defined once per camera view.
"""

import cv2
import numpy as np
import json

class ZoneManager:
    def __init__(self, zones_config):
        """
        zones_config: dict like:
        {
          "DISPLAY": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
          "CUSTOMER": [[...]],
          "BILLING": [[...]],
          "EXIT": [[...]]
        }
        """
        self.zones = {name: np.array(pts, dtype=np.int32)
                      for name, pts in zones_config.items()}

    def get_zone(self, point):
        """Return zone name for a point (x, y), or None."""
        for name, poly in self.zones.items():
            if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
                return name
        return None

    def get_zone_from_bbox(self, bbox, use_center=True):
        """Get zone from bounding box. Default: use center point."""
        x1, y1, x2, y2 = bbox
        if use_center:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return self.get_zone((cx, cy))
        return None

# ---- EXAMPLE ----
if __name__ == "__main__":
    zones = {
        "DISPLAY": [[100, 400], [500, 400], [500, 600], [100, 600]],
        "CUSTOMER": [[500, 300], [900, 300], [900, 600], [500, 600]],
        "BILLING": [[300, 200], [600, 200], [600, 300], [300, 300]],
        "EXIT": [[700, 100], [1000, 100], [1000, 200], [700, 200]]
    }

    zm = ZoneManager(zones)
    print(zm.get_zone((300, 500)))    # Should be DISPLAY
    print(zm.get_zone((700, 400)))    # Should be CUSTOMER
    print(zm.get_zone((800, 150)))    # Should be EXIT