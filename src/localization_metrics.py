from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _valid_interval(start: float, end: float) -> bool:
    return start >= 0 and end > start


def best_segment_iou(
    predicted_segments: List[Dict[str, float]],
    true_start: float,
    true_end: float,
) -> float:
    """Best temporal intersection-over-union against the ground-truth fake region."""
    if not _valid_interval(float(true_start), float(true_end)):
        return float("nan")
    if not predicted_segments:
        return 0.0

    best = 0.0
    for seg in predicted_segments:
        ps = float(seg["start"])
        pe = float(seg["end"])
        inter = max(0.0, min(pe, true_end) - max(ps, true_start))
        union = max(pe, true_end) - min(ps, true_start)
        if union > 0:
            best = max(best, inter / union)
    return float(best)


def best_center_error_seconds(
    predicted_segments: List[Dict[str, float]],
    true_start: float,
    true_end: float,
) -> float:
    """Distance between the true fake-region center and the closest predicted segment center."""
    if not _valid_interval(float(true_start), float(true_end)):
        return float("nan")
    if not predicted_segments:
        return float("inf")

    true_center = (float(true_start) + float(true_end)) / 2.0
    return float(
        min(abs(((float(seg["start"]) + float(seg["end"])) / 2.0) - true_center) for seg in predicted_segments)
    )


def localization_hit(
    predicted_segments: List[Dict[str, float]],
    true_start: float,
    true_end: float,
    min_iou: float = 0.10,
) -> int:
    """Return 1 when any predicted segment overlaps the true region sufficiently."""
    iou = best_segment_iou(predicted_segments, true_start, true_end)
    if iou != iou:  # NaN check
        return 0
    return int(iou >= min_iou)

def _get_segment_bounds(segment):
    """
    Convert a segment into (start, end).

    Accepts:
    - tuple/list: (start, end)
    - dict: {"start": ..., "end": ...}
    """

    if isinstance(segment, dict):
        start = segment["start"]
        end = segment["end"]
    else:
        start = segment[0]
        end = segment[1]

    return float(start), float(end)


def segment_iou(pred_segment, true_segment):
    """
    Compute Intersection over Union between two time segments.

    Example:
    pred_segment = (2.0, 5.0)
    true_segment = (3.0, 6.0)

    Intersection = 3.0 to 5.0 = 2 seconds
    Union = 2.0 to 6.0 = 4 seconds
    IoU = 2 / 4 = 0.5
    """

    pred_start, pred_end = _get_segment_bounds(pred_segment)
    true_start, true_end = _get_segment_bounds(true_segment)

    if pred_end <= pred_start:
        return 0.0

    if true_end <= true_start:
        return 0.0

    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = max(0.0, union_end - union_start)

    if union == 0.0:
        return 0.0

    return intersection / union