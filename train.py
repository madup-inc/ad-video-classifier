# -*- coding: utf-8 -*-
"""광고 영상 제작형식 분류 모델 학습.

CSV 로 주어진 (영상 경로, 라벨) 목록으로 LoRA 어댑터와 분류 헤드를 학습한다.
"""
from __future__ import annotations

import argparse
import csv
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import CLASSES, VideoFormatClassifier
from src.video import read_frames


class VideoDataset(Dataset):
    """(영상 경로, 클래스) 쌍을 프레임 배치로 변환한다."""

    def __init__(self, rows: list[tuple[str, int]], *, size: int = 224) -> None:
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        path, label = self.rows[index]
        frames, gap = read_frames(path, size=self.size)
        return {
            "frames": torch.from_numpy(frames),
            "gap": torch.tensor(gap, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "gap": torch.stack([b["gap"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def load_rows(path: str) -> list[tuple[str, int]]:
    """video_path, label 컬럼을 가진 CSV 를 읽는다."""
    index = {name: i for i, name in enumerate(CLASSES)}
    rows = []
    for record in csv.DictReader(open(path, encoding="utf-8")):
        label = record.get("label")
        if label in index:
            rows.append((record["video_path"], index[label]))
    return rows


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Warmup 후 cosine 으로 감쇠하는 스케줄러를 만든다."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main() -> None:
    parser = argparse.ArgumentParser(description="제작형식 분류 모델 학습")
    parser.add_argument("--train-csv", required=True, help="video_path, label 컬럼 CSV")
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--output", default="adapter_model.bin")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_rows(args.train_csv)
    if not train_rows:
        raise ValueError(f"학습 데이터가 비어 있다: {args.train_csv}")
    loader = DataLoader(
        VideoDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    model = VideoFormatClassifier().to(args.device)
    model.model.gradient_checkpointing_enable()
    model.model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"학습 파라미터 {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader) // args.accumulate
    scheduler = build_scheduler(optimizer, total_steps, max(1, total_steps // args.epochs))
    criterion = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(loader, 1):
            logits = model(batch["frames"].to(args.device), batch["gap"].to(args.device))
            loss = criterion(logits.float(), batch["label"].to(args.device)) / args.accumulate
            loss.backward()
            running += loss.item() * args.accumulate

            # Step the optimizer once per accumulation window
            if i % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running / i:.4f}", flush=True)

        state = {k: v for k, v in model.state_dict().items()
                 if "lora_" in k or k.startswith(("proj.", "fuse.", "head."))}
        torch.save(state, args.output)
        print(f"epoch {epoch + 1} 완료 — {args.output} 저장", flush=True)


if __name__ == "__main__":
    main()
