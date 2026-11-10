from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeFit:
    r2: float
    coef: np.ndarray


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    alphas: Sequence[float],
) -> RidgeFit:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    eval_scaled = scaler.transform(x_eval)
    model = RidgeCV(alphas=alphas).fit(train_scaled, y_train)
    prediction = model.predict(eval_scaled)
    coef = np.asarray(model.coef_) / scaler.scale_
    return RidgeFit(
        r2=float(r2_score(y_eval, prediction, multioutput="variance_weighted")),
        coef=coef,
    )


def held_out_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    alphas: Sequence[float],
) -> float:
    return fit_ridge(x_train, y_train, x_eval, y_eval, alphas).r2


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))
