import numpy as np


class ByteTrackWrapper:
    """Wraps ultralytics built-in ByteTrack through the YOLO tracking API.

    ultralytics and torch are imported lazily so that importing this module
    never raises ImportError when those packages are not installed locally.
    They are present inside the Modal container image.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        device: str = None,
    ):
        try:
            import torch
            _default_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _default_device = "cpu"
        self.device = device if device is not None else _default_device
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            self._model.to(self.device)

    def track_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Track persons in a single frame using ByteTrack.
        Returns [{"id": int, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
        """
        self._load_model()
        results = self._model.track(
            frame,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            verbose=False,
        )

        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            if boxes.id is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                track_id = int(boxes.id[i].cpu().numpy())
                tracks.append({
                    "id": track_id,
                    "bbox": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                    "conf": conf,
                })

        return tracks

    def reset(self) -> None:
        """Re-instantiate model to clear track state."""
        self._model = None
