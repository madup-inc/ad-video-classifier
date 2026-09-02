# -*- coding: utf-8 -*-
"""Extract model input frames from a video file."""
from __future__ import annotations

import cv2
import numpy as np

N_ANCHOR = 8
N_BURST = 4
BURST_FPS = 4.0
SIZE = 224


def read_frames(
    path: str,
    *,
    n_anchor: int = N_ANCHOR,
    n_burst: int = N_BURST,
    size: int = SIZE,
) -> tuple[np.ndarray, float]:
    """Read frames and the median inter-frame interval from a video.

    Samples n_burst consecutive frames at each of n_anchor points spread evenly
    across the video.

    Args:
        path: Path to the video file.
        n_anchor: Number of points spread evenly across the video.
        n_burst: Number of consecutive frames taken at each point.
        size: Square resize dimension.

    Returns:
        A tuple of uint8 frames shaped (T, size, size, 3) and the median
        inter-frame interval in seconds.

    Examples:
        >>> frames, gap = read_frames("ad.mp4")
        >>> frames.shape
        (32, 224, 224, 3)
    """
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if not 0 < fps <= 240:
        fps = 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        capture.release()
        raise ValueError(f"Cannot read frames from {path}")

    step = max(1, round(fps / BURST_FPS))
    wanted: list[int] = []
    for a in range(n_anchor):
        start = min(total - 1, max(0, round((a + 0.5) / n_anchor * total)))
        wanted += [min(total - 1, start + b * step) for b in range(n_burst)]

    # Decode sequentially — random seeking is unreliable across container formats
    need, grabbed, index = set(wanted), {}, 0
    while index <= max(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index in need:
            grabbed[index] = frame
        index += 1
    capture.release()
    if not grabbed:
        raise ValueError(f"No frames decoded from {path}")

    frames = []
    for i in wanted:
        frame = grabbed.get(i)
        if frame is None:
            frame = grabbed[min(grabbed, key=lambda k: abs(k - i))]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA))

    timestamps = np.array(wanted, dtype=np.float32) / fps
    gap = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    return np.stack(frames), max(gap, 1e-3)
