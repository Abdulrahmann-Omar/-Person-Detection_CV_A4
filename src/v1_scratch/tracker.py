import numpy as np
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _compute_iou(boxA, boxB):
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_b = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


class CentroidTracker:
    """Pure-numpy IoU + centroid distance tracker."""

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3, max_distance: float = 100.0):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.next_id = 0
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Update tracker with new detections.
        Returns list of active tracks with "id" key added.
        """
        if not self.tracks:
            for det in detections:
                self._register(det)
            return self._active_tracks()

        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return self._active_tracks()

        track_ids = list(self.tracks.keys())
        matched, unmatched_tracks, unmatched_dets = self._match(track_ids, detections)

        for tid, det_idx in matched:
            det = detections[det_idx]
            self.tracks[tid]["bbox"] = det["bbox"]
            self.tracks[tid]["centroid"] = _centroid(det["bbox"])
            self.tracks[tid]["conf"] = det.get("conf", 0.0)
            self.tracks[tid]["disappeared"] = 0
            self.tracks[tid]["history"].append(det["bbox"])

        for tid in unmatched_tracks:
            self.tracks[tid]["disappeared"] += 1
            if self.tracks[tid]["disappeared"] > self.max_disappeared:
                del self.tracks[tid]

        for det_idx in unmatched_dets:
            self._register(detections[det_idx])

        return self._active_tracks()

    def _register(self, detection: dict):
        tid = self.next_id
        self.next_id += 1
        bbox = detection["bbox"]
        self.tracks[tid] = {
            "id": tid,
            "bbox": bbox,
            "centroid": _centroid(bbox),
            "conf": detection.get("conf", 0.0),
            "disappeared": 0,
            "history": deque([bbox], maxlen=30),
        }

    def _active_tracks(self) -> list[dict]:
        result = []
        for tid, t in self.tracks.items():
            if t["disappeared"] == 0:
                result.append({
                    "id": t["id"],
                    "bbox": t["bbox"],
                    "conf": t["conf"],
                })
        return result

    def _match(self, track_ids, detections):
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Build IoU cost matrix
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = _compute_iou(self.tracks[tid]["bbox"], det["bbox"])

        # Build centroid distance matrix
        dist_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
        for i, tid in enumerate(track_ids):
            tc = self.tracks[tid]["centroid"]
            for j, det in enumerate(detections):
                dc = _centroid(det["bbox"])
                dist_matrix[i, j] = np.linalg.norm(tc - dc)

        # Combined cost: prefer IoU, use distance as fallback
        # Cost = 1 - IoU (lower is better). If IoU is 0, use normalized distance.
        cost_matrix = np.where(
            iou_matrix > 0,
            1.0 - iou_matrix,
            dist_matrix / max(self.max_distance, 1.0),
        )

        if HAS_SCIPY:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = self._greedy_assign(cost_matrix)

        matched = []
        unmatched_tracks = set(range(n_tracks))
        unmatched_dets = set(range(n_dets))

        for r, c in zip(row_indices, col_indices):
            iou_val = iou_matrix[r, c]
            dist_val = dist_matrix[r, c]
            if iou_val >= self.iou_threshold or dist_val <= self.max_distance:
                matched.append((track_ids[r], c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        unmatched_det_indices = list(unmatched_dets)

        return matched, unmatched_track_ids, unmatched_det_indices

    def _greedy_assign(self, cost_matrix):
        """Greedy fallback when scipy is not available."""
        rows, cols = [], []
        used_rows = set()
        used_cols = set()
        n_rows, n_cols = cost_matrix.shape

        flat_indices = np.argsort(cost_matrix, axis=None)
        for idx in flat_indices:
            r = idx // n_cols
            c = idx % n_cols
            if r not in used_rows and c not in used_cols:
                rows.append(r)
                cols.append(c)
                used_rows.add(r)
                used_cols.add(c)
            if len(rows) == min(n_rows, n_cols):
                break

        return np.array(rows), np.array(cols)
