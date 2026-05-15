import numpy as np
import cv2


class MOG2Detector:
    """Background-subtraction person detector using MOG2."""

    def __init__(
        self,
        history: int = 200,
        varThreshold: int = 40,
        min_area: int = 1500,
        max_area: int = 80000,
        morph_kernel_size: int = 5,
        aspect_ratio_range: tuple = (0.2, 4.0),
    ):
        self.history = history
        self.varThreshold = varThreshold
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel_size = morph_kernel_size
        self.aspect_ratio_range = aspect_ratio_range
        self._create_bg_subtractor()

    def _create_bg_subtractor(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=True,
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect person-like blobs via background subtraction.
        Returns [{"bbox": [x1,y1,x2,y2], "conf": float, "area": float}, ...]
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold to remove shadows (shadow pixels are 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
                continue

            x1, y1, x2, y2 = x, y, x + w, y + h
            conf = min(area / 10000.0, 1.0)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "area": float(area),
            })

        return detections

    def reset(self):
        """Re-create the background model (useful for new video)."""
        self._create_bg_subtractor()

    def warmup(self, frames: list[np.ndarray]) -> None:
        """Feed first N frames to build initial background model (no detection)."""
        for frame in frames:
            self.bg_subtractor.apply(frame)
