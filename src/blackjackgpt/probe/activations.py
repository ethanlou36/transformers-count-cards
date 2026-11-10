from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.model import GPT
from blackjackgpt.tokens import TokenTables

from .alignment import Positions, positions_for_kind


@dataclass(frozen=True)
class ActivationCache:
    activations: np.memmap
    positions: Positions
    n_layer: int
    d_model: int
    kind: str

    def layer(self, index: int) -> np.ndarray:
        return np.asarray(self.activations[:, index, :])


@torch.no_grad()
def build_activation_cache(
    model: GPT,
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    tables: TokenTables,
    out_dir: str | Path,
    device: torch.device,
    kind: str = "card",
    batch_size: int = 16,
    include_embedding: bool = False,
    positions: Positions | None = None,
) -> ActivationCache:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()

    if positions is None:
        positions = positions_for_kind(tokens, offsets, shoe_ids, tables, kind)
    n = len(positions)
    n_layer = model.config.n_layer + (1 if include_embedding else 0)
    d_model = model.config.d_model
    context_len = model.config.context_len
    pad_id = model.config.pad_token_id

    mmap_path = out / "activations.dat"
    activations = np.memmap(
        mmap_path, dtype=np.float32, mode="w+", shape=(n, n_layer, d_model)
    )

    select = np.zeros(len(tables.texts), dtype=bool)
    select_ids = np.unique(positions.token_id) if n else np.empty(0, dtype=np.int64)
    select[select_ids] = True

    cursor = 0
    shoe_ids = np.asarray(shoe_ids, dtype=np.int64)
    for batch_start in range(0, len(shoe_ids), batch_size):
        batch = shoe_ids[batch_start : batch_start + batch_size]
        seqs = [tokens[offsets[s] : offsets[s + 1]] for s in batch]
        max_len = max(len(s) for s in seqs)
        if max_len > context_len:
            raise ValueError("shoe longer than model context; cannot cache")
        idx = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        for i, seq in enumerate(seqs):
            idx[i, : len(seq)] = seq
        idx_t = torch.from_numpy(idx).to(device)
        out_dict = model(idx_t, return_activations=True)
        acts = list(out_dict["activations"])
        if include_embedding:
            pos = torch.arange(idx_t.shape[1], device=device)
            emb = model.tok_emb(idx_t) + model.pos_emb(pos)
            acts = [emb, *acts]
        stacked = torch.stack(acts, dim=2).to(torch.float32).cpu().numpy()
        for i, seq in enumerate(seqs):
            local = np.nonzero(select[seq])[0]
            k = len(local)
            if k:
                activations[cursor : cursor + k] = stacked[i, local]
                cursor += k

    if cursor != n:
        raise AssertionError(f"cached {cursor} rows but expected {n}")
    activations.flush()

    np.savez(
        out / "positions.npz",
        shoe_index=positions.shoe_index,
        within_pos=positions.within_pos,
        global_offset=positions.global_offset,
        token_id=positions.token_id,
    )
    (out / "activations_meta.json").write_text(
        json.dumps({"n": n, "n_layer": n_layer, "d_model": d_model, "kind": kind}),
        encoding="utf-8",
    )
    return ActivationCache(
        activations=activations,
        positions=positions,
        n_layer=n_layer,
        d_model=d_model,
        kind=kind,
    )
