from __future__ import annotations

from pathlib import Path

import torch

from blackjackgpt.model import GPT, GPTConfig


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[GPT, GPTConfig]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**payload["gpt_config"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, gpt_cfg


def resolve_checkpoint(run_dir: str | Path, prefer: str = "best") -> Path:
    root = Path(run_dir)
    best, last = root / "ckpt_best.pt", root / "ckpt_last.pt"
    if prefer == "best" and best.exists():
        return best
    if last.exists():
        return last
    if best.exists():
        return best
    raise FileNotFoundError(f"no checkpoint found under {root}")
