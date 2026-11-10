from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_dir: str = "data/fixtures/blackjack-v1-100-shoes"


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.0

    context_len: int | None = None


@dataclass
class TrainConfig:
    val_frac: float = 0.05
    split_seed: int = 0
    batch_size: int = 32

    epochs: int = 0
    max_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_max_batches: int = 40
    log_interval: int = 20

    checkpoint_fracs: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])

    device: str = "auto"
    seed: int = 0
    num_workers: int = 0


@dataclass
class ProbeConfig:
    alphas: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0, 1000.0])

    probe_eval_frac: float = 0.4
    probe_split_seed: int = 0

    depth_bands: list[list[int]] = field(
        default_factory=lambda: [[0, 13], [13, 26], [26, 40]]
    )

    max_shoes: int = 0


@dataclass
class EvalConfig:
    max_shoes: int = 0

    samples_per_position: int = 8
    sample_temperature: float = 1.0
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "blackjackgpt-run"
    output_root: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )


def _build_section(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    top = {k: v for k, v in raw.items() if k not in _SECTION_TYPES}
    unknown_top = set(top) - _TOP_KEYS
    if unknown_top:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown_top)}")
    sections = {
        name: _build_section(cls, raw.get(name)) for name, cls in _SECTION_TYPES.items()
    }
    return ExperimentConfig(**top, **sections)


_SECTION_TYPES: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "probe": ProbeConfig,
    "eval": EvalConfig,
}
_TOP_KEYS = {"name", "output_root"}
