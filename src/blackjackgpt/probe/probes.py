from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from .activations import ActivationCache
from .alignment import ProbeLabels
from .regression import held_out_r2


@dataclass(frozen=True)
class ProbeSplit:
    train_mask: np.ndarray
    eval_mask: np.ndarray


def split_positions_by_shoe(
    shoe_index: np.ndarray, eval_frac: float, seed: int
) -> ProbeSplit:
    shoes = np.unique(shoe_index)
    perm = np.random.default_rng(seed).permutation(shoes)
    n_eval = max(1, round(len(shoes) * eval_frac)) if len(shoes) > 1 else 0
    eval_shoes = set(perm[:n_eval].tolist())
    eval_mask = np.array([s in eval_shoes for s in shoe_index])
    return ProbeSplit(train_mask=~eval_mask, eval_mask=eval_mask)


def _residualize(
    control_train: np.ndarray,
    control_eval: np.ndarray,
    target_train: np.ndarray,
    target_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reg = LinearRegression().fit(control_train, target_train)
    res_train = target_train - reg.predict(control_train)
    res_eval = target_eval - reg.predict(control_eval)
    return res_train, res_eval


@dataclass
class ProbeReport:
    payload: dict

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8")


def run_probe_class(
    cache: ActivationCache,
    labels: ProbeLabels,
    alphas: list[float],
    eval_frac: float,
    split_seed: int,
    depth_bands: list[list[int]],
) -> dict:
    positions = cache.positions
    split = split_positions_by_shoe(positions.shoe_index, eval_frac, split_seed)
    tr, ev = split.train_mask, split.eval_mask
    n_layer = cache.n_layer
    targets = labels.as_targets()
    dealt = labels.cards_dealt.reshape(-1, 1)

    r2_by_layer: dict[str, list[float]] = {name: [] for name in targets}
    partial: list[float] = []
    band_names = [f"{lo}_{hi}" for lo, hi in depth_bands]
    within_bands: dict[str, list[float]] = {name: [] for name in band_names}

    for layer in range(n_layer):
        feats = cache.layer(layer)
        for name, y in targets.items():
            r2_by_layer[name].append(
                held_out_r2(feats[tr], y[tr], feats[ev], y[ev], alphas)
            )

        x_res_tr, x_res_ev = _residualize(dealt[tr], dealt[ev], feats[tr], feats[ev])
        y = labels.running_count
        y_res_tr, y_res_ev = _residualize(dealt[tr], dealt[ev], y[tr], y[ev])
        partial.append(held_out_r2(x_res_tr, y_res_tr, x_res_ev, y_res_ev, alphas))

        for (lo, hi), band_name in zip(depth_bands, band_names):
            in_band = (labels.cards_dealt >= lo) & (labels.cards_dealt < hi)
            btr, bev = tr & in_band, ev & in_band
            if btr.sum() >= 5 and bev.sum() >= 2:
                within_bands[band_name].append(
                    held_out_r2(feats[btr], y[btr], feats[bev], y[bev], alphas)
                )
            else:
                within_bands[band_name].append(float("nan"))

    def best(name: str) -> dict:
        arr = np.array(r2_by_layer[name])
        idx = int(np.nanargmax(arr)) if np.isfinite(arr).any() else -1
        return {"layer": idx + 1, "r2": float(arr[idx]) if idx >= 0 else None}

    return {
        "kind": cache.kind,
        "targets": list(targets),
        "n_layer": n_layer,
        "n_positions": len(positions),
        "n_train_positions": int(tr.sum()),
        "n_eval_positions": int(ev.sum()),
        "n_train_shoes": len(np.unique(positions.shoe_index[tr])),
        "n_eval_shoes": len(np.unique(positions.shoe_index[ev])),
        "alphas": alphas,
        "layers": [f"layer_{i + 1}" for i in range(n_layer)],
        "r2_by_layer": r2_by_layer,
        "controls": {
            "running_count_partial_out_cards_dealt": partial,
            "running_count_within_depth_bands": within_bands,
            "depth_bands": depth_bands,
        },
        "best_layer": {name: best(name) for name in targets},
    }


def combine_report(class_payloads: dict[str, dict], meta: dict) -> ProbeReport:
    payload = dict(meta)
    payload["position_classes"] = class_payloads
    return ProbeReport(payload)


def run_probes(
    cache: ActivationCache,
    labels: ProbeLabels,
    alphas: list[float],
    eval_frac: float,
    split_seed: int,
    depth_bands: list[list[int]],
) -> ProbeReport:
    return ProbeReport(
        run_probe_class(cache, labels, alphas, eval_frac, split_seed, depth_bands)
    )


def _plot_panel(ax, panel: dict, title: str) -> None:
    n_layer = panel["n_layer"]
    layers = list(range(1, n_layer + 1))
    for name in panel["targets"]:
        ax.plot(layers, panel["r2_by_layer"][name], marker="o", label=name)
    partial = panel["controls"]["running_count_partial_out_cards_dealt"]
    ax.plot(
        layers,
        partial,
        marker="s",
        linestyle="--",
        color="black",
        label="running_count | depth (partial)",
    )
    ax.set_xlabel("layer (post-block residual stream)")
    ax.set_ylabel("held-out R²")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def plot_report(report: ProbeReport, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = report.payload.get("position_classes")
    if classes is None:
        classes = {report.payload.get("kind", "positions"): report.payload}

    names = list(classes)
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        _plot_panel(ax, classes[name], f"{name} positions")
    fig.suptitle("Linear decodability of count vs layer")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
