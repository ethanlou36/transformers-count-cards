from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        q = q.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.proj = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.pos_emb.weight.numel()

        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        loss_mask: torch.Tensor | None = None,
        return_activations: bool = False,
    ) -> dict[str, object]:
        batch, seq = idx.shape
        if seq > self.config.context_len:
            raise ValueError(
                f"sequence length {seq} exceeds context_len {self.config.context_len}"
            )
        pos = torch.arange(seq, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        activations: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            if return_activations:
                activations.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        result: dict[str, object] = {"logits": logits}
        if return_activations:
            result["activations"] = activations

        if targets is not None:
            per_pos = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(batch, seq)
            if loss_mask is not None:
                mask = loss_mask.to(per_pos.dtype)
                denom = mask.sum().clamp_min(1.0)
                result["loss"] = (per_pos * mask).sum() / denom
                result["per_position_loss"] = per_pos.masked_fill(
                    ~loss_mask, float("nan")
                )
            else:
                result["loss"] = per_pos.mean()
                result["per_position_loss"] = per_pos
        return result
