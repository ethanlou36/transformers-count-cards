from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from blackjackgpt.blackjack.state import Action
from blackjackgpt.expert.observation import ActionObservation, BettingObservation
from blackjackgpt.expert.policy import ExpertPolicy
from blackjackgpt.model import GPT
from blackjackgpt.play.harness import Context, Decision


class ScriptedDecisionMaker:
    def __init__(self, policy: ExpertPolicy) -> None:
        self.policy = policy

    def decide(self, contexts: list[Context]) -> list[Decision]:
        out = []
        for c in contexts:
            if c.is_bet:
                bet_obs = BettingObservation(
                    running_count=c.running_count,
                    physical_cards_remaining=c.cards_remaining,
                )
                out.append(
                    Decision(bet_units=self.policy.choose_bet(bet_obs).bet_units)
                )
            else:
                assert c.dealer_upcard is not None
                act_obs = ActionObservation(
                    player_cards=c.player_ranks,
                    dealer_upcard=c.dealer_upcard,
                    running_count=c.running_count,
                    physical_cards_remaining=c.cards_remaining,
                    legal_actions=c.legal_actions,
                )
                out.append(
                    Decision(action=self.policy.choose_action(act_obs).selected_action)
                )
        return out


@dataclass
class AblationSpec:
    layer_block: int
    directions: np.ndarray
    mean_projection: np.ndarray


class ModelDecisionMaker:
    def __init__(
        self,
        model: GPT,
        vocab,
        device: torch.device,
        ablation: AblationSpec | None = None,
        audit_policy: ExpertPolicy | None = None,
    ) -> None:
        self.model = model.eval()
        self.vocab = vocab
        self.device = device
        self.pad_id = model.config.pad_token_id
        self.ctx = model.config.context_len
        self.ablation = ablation
        self.audit = audit_policy
        self.bet_unit_of = {vocab.encode(f"BET_{u}"): u for u in (1, 2, 4, 6, 8)}
        self.action_of = {
            vocab.encode(f"ACTION:{a.value}"): a
            for a in (Action.HIT, Action.STAND, Action.DOUBLE)
        }
        self.raw_legal = 0
        self.decisions = 0

        self.bet_total = self.bet_match = 0
        self.action_total = self.action_match = 0
        self.dev_total = self.dev_match = 0
        self.nondev_total = self.nondev_match = 0
        self._handle: RemovableHandle | None = None

    def _install_hook(self) -> None:
        if self.ablation is None:
            return
        U = torch.tensor(
            self.ablation.directions, dtype=torch.float32, device=self.device
        )
        m = torch.tensor(
            self.ablation.mean_projection, dtype=torch.float32, device=self.device
        )

        def hook(_module, _inp, out):
            coord = out @ U
            return out + (m - coord) @ U.T

        self._handle = self.model.blocks[
            self.ablation.layer_block
        ].register_forward_hook(hook)

    @torch.no_grad()
    def decide(self, contexts: list[Context]) -> list[Decision]:
        lengths = [min(len(c.transcript), self.ctx) for c in contexts]
        mx = max(lengths)
        idx = np.full((len(contexts), mx), self.pad_id, dtype=np.int64)
        for i, c in enumerate(contexts):
            seq = c.transcript[-lengths[i] :]
            idx[i, : lengths[i]] = seq
        idx_t = torch.from_numpy(idx).to(self.device)
        self._install_hook()
        try:
            logits = self.model(idx_t)["logits"]
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
        out: list[Decision] = []
        for i, c in enumerate(contexts):
            row = logits[i, lengths[i] - 1]
            raw_arg = int(torch.argmax(row).item())
            legal = c.legal_token_ids
            if raw_arg in legal:
                self.raw_legal += 1
            self.decisions += 1
            legal_t = torch.tensor(legal, device=self.device)
            choice = int(legal_t[torch.argmax(row[legal_t])].item())
            if c.is_bet:
                d = Decision(bet_units=self.bet_unit_of[choice])
            else:
                d = Decision(action=self.action_of[choice])
            if self.audit is not None:
                self._audit(c, d)
            out.append(d)
        return out

    def _audit(self, c: Context, d: Decision) -> None:
        assert self.audit is not None
        if c.is_bet:
            expert = self.audit.choose_bet(
                BettingObservation(
                    running_count=c.running_count,
                    physical_cards_remaining=c.cards_remaining,
                )
            ).bet_units
            self.bet_total += 1
            self.bet_match += int(expert == d.bet_units)
        else:
            assert c.dealer_upcard is not None
            dec = self.audit.choose_action(
                ActionObservation(
                    player_cards=c.player_ranks,
                    dealer_upcard=c.dealer_upcard,
                    running_count=c.running_count,
                    physical_cards_remaining=c.cards_remaining,
                    legal_actions=c.legal_actions,
                )
            )
            self.action_total += 1
            self.action_match += int(dec.selected_action == d.action)
            if dec.action_changed_from_basic:
                self.dev_total += 1
                self.dev_match += int(dec.selected_action == d.action)
            else:
                self.nondev_total += 1
                self.nondev_match += int(dec.selected_action == d.action)

    @property
    def in_play_legality(self) -> float:
        return self.raw_legal / max(1, self.decisions)
