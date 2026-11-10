from __future__ import annotations

from typing import Final

from .policy import BetRampEntry, ExpertPolicy, ExpertPolicyConfig

COUNT_BLIND_BET_RAMP: Final[tuple[BetRampEntry, ...]] = (BetRampEntry(None, 1),)


COUNT_BLIND_POLICY_CONFIG: Final[ExpertPolicyConfig] = ExpertPolicyConfig(
    bet_ramp=COUNT_BLIND_BET_RAMP,
    deviations=(),
)


def count_blind_policy() -> ExpertPolicy:
    return ExpertPolicy(COUNT_BLIND_POLICY_CONFIG)
