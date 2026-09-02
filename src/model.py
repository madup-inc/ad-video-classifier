# -*- coding: utf-8 -*-
"""광고 영상 제작형식 분류 모델.

Qwen2.5-VL 백본에 LoRA 어댑터와 분류 헤드를 얹은 구조.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

CLASSES = ["UGC", "MOTION_GRAPHICS", "LIVE"]

# Head layout as stored in the released weights
_HEAD_CLASSES = ["AI_GENERATED", "UGC", "MOTION_GRAPHICS", "LIVE_PRODUCTION", "LIVE_COMPOSITE"]

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "qkv", "proj", "fc1", "fc2",
]

PROMPT = (
    "이 광고 영상의 프레임들이다. 영상의 제작 방식(촬영 vs 그래픽 합성)과 "
    "시간에 따른 변화 양상을 분석하라."
)


class VideoFormatClassifier(nn.Module):
    """영상 프레임을 받아 제작형식 클래스 로짓을 반환한다."""

    def __init__(
        self,
        num_classes: int = len(_HEAD_CLASSES),
        *,
        dim: int = 512,
        lora_r: int = 128,
        lora_alpha: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype)

        # Attach LoRA adapters so the published weights load onto matching module shapes
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(backbone, config)

        hidden = backbone.config.text_config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.GELU())
        self.fuse = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.1))
        self.head = nn.Linear(dim, num_classes)

    def _encode(self, frames: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        """프레임 배치를 Qwen2.5-VL 비디오 입력으로 변환한다."""
        videos = [[f.numpy() for f in clip] for clip in frames.cpu()]
        message = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text=[text] * len(videos), videos=videos, return_tensors="pt")

        out = {}
        for key, value in encoded.items():
            if not torch.is_tensor(value):
                out[key] = value
            elif value.dtype.is_floating_point:
                out[key] = value.to(device, self.proj[1].weight.dtype)
            else:
                out[key] = value.to(device)
        return out

    def forward(self, frames: torch.Tensor, frame_gap: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            frames: (B, T, H, W, 3) uint8 프레임 배치.
            frame_gap: (B,) 프레임 간 실제 간격(초). 시간축 위치 인코딩에 사용된다.

        Returns:
            (B, num_classes) 로짓.
        """
        device = next(self.head.parameters()).device
        encoded = self._encode(frames, device)

        # Feed the measured inter-frame interval into the temporal position axis
        if frame_gap is not None and "second_per_grid_ts" in encoded:
            encoded["second_per_grid_ts"] = frame_gap.to(device, torch.float32) * 2.0

        output = self.model(**encoded, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[-1]

        # Mean-pool over non-padding positions
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

        feature = self.proj(pooled.to(self.proj[1].weight.dtype))
        return self.head(self.fuse(feature))

    def predict_proba(self, frames: torch.Tensor, frame_gap: torch.Tensor | None = None) -> torch.Tensor:
        """CLASSES 순서의 클래스 확률을 반환한다.

        Args:
            frames: (B, T, H, W, 3) uint8 프레임 배치.
            frame_gap: (B,) 프레임 간 실제 간격(초).

        Returns:
            (B, 3) 확률. 합이 1이 되도록 정규화된다.

        Examples:
            >>> probs = model.predict_proba(frames, gaps)
            >>> CLASSES[int(probs[0].argmax())]
            'UGC'
        """
        probs = self.forward(frames, frame_gap).float().softmax(-1)
        ugc = probs[:, _HEAD_CLASSES.index("UGC")]
        motion = probs[:, _HEAD_CLASSES.index("MOTION_GRAPHICS")]
        live = probs[:, _HEAD_CLASSES.index("LIVE_PRODUCTION")] + probs[:, _HEAD_CLASSES.index("LIVE_COMPOSITE")]
        merged = torch.stack([ugc, motion, live], dim=-1)
        return merged / merged.sum(-1, keepdim=True).clamp(min=1e-8)


def load_pretrained(weights_path: str, **kwargs) -> VideoFormatClassifier:
    """공개된 어댑터 가중치를 로드한다.

    Examples:
        >>> model = load_pretrained("adapter_model.bin")
        >>> model.eval().cuda()
    """
    model = VideoFormatClassifier(**kwargs)
    state = torch.load(weights_path, map_location="cpu")

    # Strip the training-time module prefixes
    remapped = {}
    for key, value in state.items():
        name = key.replace("net.vlm.model.", "model.").replace("net.vlm.proj.", "proj.")
        name = name.replace("net.fuse.", "fuse.").replace("net.heads.format5.", "head.")
        remapped[name] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    trained = [k for k in missing if "lora_" in k or k.startswith(("proj.", "fuse.", "head."))]
    if trained:
        raise RuntimeError(f"학습된 가중치 {len(trained)}개가 로드되지 않았다: {trained[:5]}")
    return model
