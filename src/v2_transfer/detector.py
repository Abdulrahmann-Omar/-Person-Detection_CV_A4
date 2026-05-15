import numpy as np


class YOLOv8Detector:
    """Wraps ultralytics YOLOv8n, filters to person class (class_id=0).

    ultralytics and torch are imported lazily in _load_model() so that
    importing this module never raises ImportError when those packages
    are not installed locally (they are present on Modal).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = None,
    ):
        try:
            import torch
            _default_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _default_device = "cpu"
        self.device = device if device is not None else _default_device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            self._model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 on frame, filter to person class.
        Returns [{"bbox": [x1,y1,x2,y2], "conf": float}, ...]
        """
        self._load_model()
        results = self._model(
            frame,
            classes=[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                detections.append({
                    "bbox": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                    "conf": conf,
                })

        return detections

    def warmup(self) -> None:
        """Run one dummy inference on blank 640x640 frame to load weights."""
        self._load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
