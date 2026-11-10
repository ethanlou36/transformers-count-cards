from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.intervene.dose_response import _round_start_positions
from blackjackgpt.model import load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import fit_ridge
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables

from .decision import AblationSpec


class AblationContext:
    def __init__(
        self, exp, run_dir: Path, device: torch.device, max_shoes: int = 2000
    ) -> None:
        vocab, _pad = load_vocabulary(exp.data.dataset_dir)
        tables = TokenTables(vocab)
        tokens, offsets = load_stream(exp.data.dataset_dir)
        sidecar = load_sidecar(exp.data.dataset_dir)
        split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
        shoe_ids = split.val[:max_shoes]
        model, _ = load_model(resolve_checkpoint(run_dir), device)
        n_block = model.config.n_layer

        positions = _round_start_positions(tokens, offsets, shoe_ids, vocab)
        cache = build_activation_cache(
            model,
            tokens,
            offsets,
            shoe_ids,
            tables,
            run_dir / "play" / "ablation_cache",
            device,
            "bet",
            positions=positions,
        )
        labels = load_labels(cache.positions, sidecar, "before")
        rc = labels.running_count
        comp = labels.remaining_composition
        sp = split_positions_by_shoe(
            cache.positions.shoe_index,
            exp.probe.probe_eval_frac,
            exp.probe.probe_split_seed,
        )
        tr, ev = sp.train_mask, sp.eval_mask
        alphas = exp.probe.alphas

        per_layer = [
            fit_ridge(cache.layer(i)[tr], rc[tr], cache.layer(i)[ev], rc[ev], alphas).r2
            for i in range(n_block)
        ]
        self.best_layer = int(np.nanargmax(per_layer))
        self.running_count_r2 = float(per_layer[self.best_layer])
        self.feats = cache.layer(self.best_layer)
        self.d = self.feats.shape[1]
        self.running_count = rc
        self.composition = comp
        self.train_mask, self.eval_mask = tr, ev
        self.alphas = alphas

        hi = rc >= np.median(rc)
        self.count_dir = self.feats[hi].mean(0) - self.feats[~hi].mean(0)

        self.W = fit_ridge(
            self.feats[tr], comp[tr], self.feats[ev], comp[ev], alphas
        ).coef
        self.composition_r2 = float(
            fit_ridge(self.feats[tr], comp[tr], self.feats[ev], comp[ev], alphas).r2
        )

    def _spec(self, directions: np.ndarray) -> AblationSpec:
        q, _ = np.linalg.qr(directions)
        q = q[:, : directions.shape[1]]
        mean_proj = (self.feats @ q).mean(axis=0)
        return AblationSpec(
            layer_block=self.best_layer,
            directions=q.astype(np.float64),
            mean_projection=mean_proj.astype(np.float64),
        )

    def count_ablation(self) -> AblationSpec:
        return self._spec(self.count_dir[:, None])

    def composition_ablation(self, k: int) -> AblationSpec:
        _, _, vt = np.linalg.svd(self.W, full_matrices=False)
        return self._spec(vt[:k].T)

    def random_ablation(self, k: int, seed: int = 0) -> AblationSpec:
        rng = np.random.default_rng(seed)
        return self._spec(rng.standard_normal((self.d, k)))

    def composition_variance_ablation(self, k: int) -> AblationSpec:
        fc = self.feats - self.feats.mean(0)
        m = fc.T @ (fc @ self.W.T)
        u_var, _, _ = np.linalg.svd(m, full_matrices=False)
        return self._spec(u_var[:, :k])

    def all_specs(self) -> dict[str, AblationSpec]:
        return {
            "count": self.count_ablation(),
            "comp_k1": self.composition_ablation(1),
            "comp_k2": self.composition_ablation(2),
            "comp_k5": self.composition_ablation(5),
            "comp_k10": self.composition_ablation(10),
            "random_k1": self.random_ablation(1),
            "random_k10": self.random_ablation(10),
            "compvar_k1": self.composition_variance_ablation(1),
        }


def save_specs(specs: dict[str, AblationSpec], path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        arrays[f"{name}__dir"] = spec.directions
        arrays[f"{name}__mean"] = spec.mean_projection
        arrays[f"{name}__layer"] = np.array([spec.layer_block])
    path.parent.mkdir(parents=True, exist_ok=True)
    cast(Any, np.savez)(str(path), **arrays)


def load_specs(path: Path) -> dict[str, AblationSpec]:
    data = np.load(path)
    names = sorted({k.split("__")[0] for k in data.files})
    return {
        name: AblationSpec(
            layer_block=int(data[f"{name}__layer"][0]),
            directions=data[f"{name}__dir"],
            mean_projection=data[f"{name}__mean"],
        )
        for name in names
    }
