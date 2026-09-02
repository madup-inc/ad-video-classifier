# -*- coding: utf-8 -*-
"""단일 영상 분류 예시."""
import sys

import torch

sys.path.insert(0, "..")

from src.model import CLASSES, load_pretrained
from src.video import read_frames

model = load_pretrained("adapter_model.bin").eval().cuda()

frames, gap = read_frames(sys.argv[1])
batch = torch.from_numpy(frames).unsqueeze(0).cuda()
gaps = torch.tensor([gap]).cuda()

with torch.no_grad():
    probs = model.predict_proba(batch, gaps)[0]

for name, value in zip(CLASSES, probs.tolist()):
    print(f"{name:16s} {value:.4f}")
print(f"\n→ {CLASSES[int(probs.argmax())]}")
