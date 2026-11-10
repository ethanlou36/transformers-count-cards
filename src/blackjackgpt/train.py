from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from blackjackgpt.config import ExperimentConfig, load_config
from blackjackgpt.dataset import (
    ShoeDataset,
    load_stream,
    max_shoe_length,
    split_by_shoe,
)
from blackjackgpt.model import GPT, GPTConfig, load_vocabulary
from blackjackgpt.provenance import dataset_manifest, git_commit
from blackjackgpt.tokens import TokenClass, TokenTables

_TRACKED_CLASSES = (
    TokenClass.BET,
    TokenClass.CARD,
    TokenClass.ACTION,
    TokenClass.OUTCOME,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and (
            torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bet_unigram_cross_entropy(
    tokens: np.ndarray,
    offsets: np.ndarray,
    train_shoe_ids: np.ndarray,
    tables: TokenTables,
) -> float:
    bet_ids = np.array(sorted(tables.ids_in_class(TokenClass.BET)), dtype=np.int64)
    is_bet = np.zeros(len(tables.texts), dtype=bool)
    is_bet[bet_ids] = True
    counts = np.zeros(len(tables.texts), dtype=np.int64)
    for shoe_id in train_shoe_ids:
        seq = tokens[offsets[shoe_id] : offsets[shoe_id + 1]]
        bet_toks = seq[is_bet[seq]]
        np.add.at(counts, bet_toks, 1)
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts[bet_ids] / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def lr_at_step(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def _infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def build_gpt_config(exp: ExperimentConfig, offsets, vocab, pad_id) -> GPTConfig:
    max_len = max_shoe_length(offsets)
    context_len = exp.model.context_len
    if context_len is None:
        context_len = max_len
    elif context_len < max_len:
        raise ValueError(
            f"model.context_len ({context_len}) < max shoe length ({max_len})"
        )
    return GPTConfig(
        vocab_size=len(vocab),
        context_len=context_len,
        n_layer=exp.model.n_layer,
        n_head=exp.model.n_head,
        d_model=exp.model.d_model,
        dropout=exp.model.dropout,
        pad_token_id=pad_id,
    )


@torch.no_grad()
def evaluate(
    model: GPT,
    loader: DataLoader,
    tables: TokenTables,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    class_index = {cls: i for i, cls in enumerate(_TRACKED_CLASSES)}
    id_to_class = torch.full((len(tables.texts),), -1, dtype=torch.long)
    for tid, cls in enumerate(tables.token_class):
        if cls in class_index:
            id_to_class[tid] = class_index[cls]

    total_loss = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    class_loss = torch.zeros(len(_TRACKED_CLASSES), device=device)
    class_count = torch.zeros(len(_TRACKED_CLASSES), device=device)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        mask = batch["loss_mask"].to(device)
        out = model(x, targets=y, loss_mask=mask)
        per_pos = out["per_position_loss"]
        valid = mask
        losses = per_pos[valid]
        total_loss += losses.sum()
        total_count += valid.sum()
        target_class = id_to_class.to(device)[y[valid]]
        for ci in range(len(_TRACKED_CLASSES)):
            sel = target_class == ci
            class_loss[ci] += losses[sel].sum()
            class_count[ci] += sel.sum()

    model.train()
    metrics = {"val_loss": float((total_loss / total_count.clamp_min(1)).item())}
    for cls, ci in class_index.items():
        count = class_count[ci].item()
        mean = (class_loss[ci] / class_count[ci].clamp_min(1)).item()
        metrics[f"val_loss_{cls.value.lower()}"] = (
            float(mean) if count else float("nan")
        )
    return metrics


def save_checkpoint(
    path: Path, model: GPT, gpt_cfg: GPTConfig, step: int, metrics: dict
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "gpt_config": asdict(gpt_cfg),
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def train(exp: ExperimentConfig) -> Path:
    cfg = exp.train
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)

    vocab, pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, cfg.val_frac, cfg.split_seed)
    gpt_cfg = build_gpt_config(exp, offsets, vocab, pad_id)

    train_ds = ShoeDataset(tokens, offsets, split.train, gpt_cfg.context_len, pad_id)
    val_ds = ShoeDataset(tokens, offsets, split.val, gpt_cfg.context_len, pad_id)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if cfg.epochs:
        steps_per_epoch = math.ceil(len(split.train) / cfg.batch_size)
        cfg.max_steps = cfg.epochs * steps_per_epoch
        if cfg.warmup_steps >= cfg.max_steps:
            cfg.warmup_steps = max(1, cfg.max_steps // 20)

    bet_unigram_ce = bet_unigram_cross_entropy(tokens, offsets, split.train, tables)

    checkpoint_steps = {
        max(0, int(cfg.max_steps * frac) - 1) for frac in cfg.checkpoint_fracs
    }

    model = GPT(gpt_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    use_amp = device.type == "cuda"

    run_dir = exp.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    exp.save(run_dir / "config.yaml")
    metadata = {
        "name": exp.name,
        "seed": cfg.seed,
        "device": str(device),
        "git": git_commit(),
        "dataset_manifest": dataset_manifest(exp.data.dataset_dir),
        "gpt_config": asdict(gpt_cfg),
        "num_parameters": model.num_parameters(),
        "num_train_shoes": len(split.train),
        "num_val_shoes": len(split.val),
        "max_steps": cfg.max_steps,
        "epochs": cfg.epochs,
        "bet_unigram_cross_entropy": bet_unigram_ce,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    metrics_path = run_dir / "metrics.csv"
    fieldnames = (
        ["step", "lr", "train_loss", "val_loss"]
        + [f"val_loss_{c.value.lower()}" for c in _TRACKED_CLASSES]
        + ["bet_unigram_ce"]
    )
    metrics_file = metrics_path.open("w", newline="")
    writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
    writer.writeheader()

    data_iter = _infinite(train_loader)
    best_val = float("inf")
    print(
        f"[train] {exp.name}: {model.num_parameters():,} params, "
        f"ctx={gpt_cfg.context_len}, device={device}, "
        f"{len(split.train)} train / {len(split.val)} val shoes, "
        f"{cfg.max_steps} steps, bet_unigram_ce={bet_unigram_ce:.4f}"
    )

    model.train()
    for step in range(cfg.max_steps):
        lr = lr_at_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = next(data_iter)
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        mask = batch["loss_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x, targets=y, loss_mask=mask)
            loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            print(f"  step {step:5d} | lr {lr:.2e} | train_loss {loss.item():.4f}")

        is_last = step == cfg.max_steps - 1
        if step % cfg.eval_interval == 0 or is_last:
            val_metrics = evaluate(
                model, val_loader, tables, device, cfg.eval_max_batches
            )
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.item()),
                "bet_unigram_ce": bet_unigram_ce,
                **val_metrics,
            }
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            metrics_file.flush()
            bet = val_metrics.get("val_loss_bet", float("nan"))
            print(
                f"  step {step:5d} | val_loss {val_metrics['val_loss']:.4f} "
                f"| val_loss_bet {bet:.4f} (unigram {bet_unigram_ce:.4f})"
            )
            save_checkpoint(run_dir / "ckpt_last.pt", model, gpt_cfg, step, val_metrics)
            if val_metrics["val_loss"] < best_val:
                best_val = val_metrics["val_loss"]
                save_checkpoint(
                    run_dir / "ckpt_best.pt", model, gpt_cfg, step, val_metrics
                )

        if step in checkpoint_steps:
            save_checkpoint(
                run_dir / f"ckpt_step_{step:06d}.pt", model, gpt_cfg, step, {}
            )

    metrics_file.close()
    print(f"[train] done. best val_loss={best_val:.4f}. run dir: {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the blackjack GPT.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    if args.seed is not None:
        exp.train.seed = args.seed
    if args.name is not None:
        exp.name = args.name
    train(exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
