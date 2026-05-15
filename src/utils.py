import time
import numpy as np
import cv2


def read_video_frames(path: str) -> tuple[list[np.ndarray], dict]:
    """Returns (frames_list, metadata_dict with fps/width/height/total_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    metadata = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }
    return frames, metadata


def write_output_video(frames: list[np.ndarray], path: str, fps: float) -> None:
    """Write annotated frames to mp4 using cv2.VideoWriter (mp4v codec)."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def draw_tracks(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes + IDs on frame.
    tracks = [{"id": int, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
    Use distinct color per ID (hash ID to HSV then convert to BGR).
    Draw filled label above bbox.
    Return annotated frame copy.
    """
    out = frame.copy()
    for t in tracks:
        tid = t["id"]
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        conf = t.get("conf", 0.0)

        hue = int((tid * 47) % 180)
        hsv_color = np.array([[[hue, 255, 200]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tid} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness)

    return out


def compute_iou(boxA: list, boxB: list) -> float:
    """Standard IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_b = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def evaluate_tracking(gt_tracks: list[dict], pred_tracks: list[dict]) -> dict:
    """
    Simple MOTA-like metrics.
    Returns {"precision": float, "recall": float, "f1": float,
             "id_switches": int, "avg_iou": float}
    Uses IoU threshold 0.5 for TP matching.
    If gt_tracks is empty, return zeros (graceful no-GT mode).
    """
    if not gt_tracks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "id_switches": 0, "avg_iou": 0.0}

    tp = 0
    matched_ious = []
    gt_matched_ids = {}
    id_switches = 0

    gt_used = [False] * len(gt_tracks)
    for pred in pred_tracks:
        best_iou = 0.0
        best_idx = -1
        for gi, gt in enumerate(gt_tracks):
            if gt_used[gi]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= 0.5 and best_idx >= 0:
            tp += 1
            gt_used[best_idx] = True
            matched_ious.append(best_iou)
            gt_id = gt_tracks[best_idx].get("id", best_idx)
            pred_id = pred.get("id", -1)
            if gt_id in gt_matched_ids and gt_matched_ids[gt_id] != pred_id:
                id_switches += 1
            gt_matched_ids[gt_id] = pred_id

    precision = tp / len(pred_tracks) if pred_tracks else 0.0
    recall = tp / len(gt_tracks) if gt_tracks else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "id_switches": id_switches,
        "avg_iou": avg_iou,
    }


def benchmark_speed(func, *args, n_runs: int = 5) -> dict:
    """Run func(*args) n_runs times, return {"mean_ms": float, "std_ms": float, "fps": float}."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}
