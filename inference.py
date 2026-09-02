# -*- coding: utf-8 -*-
"""Command-line entry point for advertising video production-format classification."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.model import CLASSES, load_pretrained
from src.video import read_frames


def classify(model, paths: list[str], *, batch_size: int = 2) -> list[dict]:
    """Classify a list of video paths.

    Args:
        model: Model returned by load_pretrained.
        paths: Video file paths.
        batch_size: Number of videos processed at once.

    Returns:
        One dict per path with keys path, label, confidence and scores.

    Examples:
        >>> results = classify(model, ["ad.mp4"])
        >>> results[0]["label"]
        'UGC'
    """
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        decoded = [read_frames(p) for p in chunk]
        frames = torch.from_numpy(np.stack([f for f, _ in decoded]))
        gaps = torch.tensor([g for _, g in decoded], dtype=torch.float32)

        with torch.no_grad():
            probs = model.predict_proba(frames.to(device), gaps.to(device)).cpu()

        for path, row in zip(chunk, probs):
            index = int(row.argmax())
            results.append(
                {
                    "path": path,
                    "label": CLASSES[index],
                    "confidence": round(float(row[index]), 4),
                    "scores": {c: round(float(v), 4) for c, v in zip(CLASSES, row)},
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify advertising videos by production format")
    parser.add_argument("videos", nargs="+", help="video file paths")
    parser.add_argument("--weights", default="adapter_model.bin", help="adapter weights path")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_pretrained(args.weights).eval().to(args.device)
    for item in classify(model, args.videos, batch_size=args.batch_size):
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
