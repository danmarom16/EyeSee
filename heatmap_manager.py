import numpy as np
import cv2

class HeatmapManager:

    def __init__(self, CFG):
        self.heatmap = None
        self.initialized = False
        self.colormap = cv2.COLORMAP_PARULA if self.CFG["colormap"] is None else self.CFG["colormap"]
    def initialize_heatmap(self, frame):
        if not self.initialized:
            self.heatmap = np.zeros_like(frame, dtype=np.float32)
            self.initialized = True

    def apply_heatmap_effect(self, box):
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # Calculate squared distances from the center
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # Create a mask of points within the radius
        within_radius = dist_squared <= radius_squared

        # Update only the values within the bounding box in a single vectorized operation
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def normalize_heatmap(self, im0):
        return cv2.addWeighted(
            im0,
            0.5,
            cv2.applyColorMap(
                cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), self.colormap
            ),
            0.5,
            0,
        )

    def overlay_heatmap(self, frame, colormap):
        # Logic for normalizing and overlaying the heatmap
        return self.normalize_heatmap(frame, self.heatmap, colormap)


