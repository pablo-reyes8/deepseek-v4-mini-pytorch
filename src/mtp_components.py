# ============================================================
# Mini DeepSeek-V4 MTP / Multi-Token Prediction
# Auxiliary future-token prediction heads - canonical mini version
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """
    Minimal RMSNorm used by the MTP transform.

    Input:
        x: [..., dim]

    Output:
        normalized x with the same shape.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected x.shape[-1] == dim={self.dim}, got {x.shape[-1]}"
            )

        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)

        return x * self.weight.to(dtype=x.dtype)


# ============================================================
# MTP label construction
# ============================================================

def build_mtp_labels(
    input_ids: torch.Tensor,
    mtp_depth: int,
    ignore_index: int = -100,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Build labels for Multi-Token Prediction.

    Main causal LM head predicts:
        x_{t+1}

    MTP heads predict:
        depth 0 -> x_{t+2}
        depth 1 -> x_{t+3}
        ...
        depth k -> x_{t+k+2}

    Args:
        input_ids:
            Token ids with shape [B, T].

        mtp_depth:
            Number of MTP future-token heads.

        ignore_index:
            Label value ignored by cross_entropy.

        pad_token_id:
            Optional padding token id. If provided, positions whose future
            target is padding are also set to ignore_index.

    Returns:
        mtp_labels:
            [B, mtp_depth, T]
    """
    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must have shape [B,T], got {tuple(input_ids.shape)}"
        )

    if torch.is_floating_point(input_ids):
        raise TypeError("input_ids must be integer token ids, not floating point.")

    if mtp_depth <= 0:
        raise ValueError(f"mtp_depth must be > 0, got {mtp_depth}")

    B, T = input_ids.shape

    labels = torch.full(
        (B, mtp_depth, T),
        fill_value=int(ignore_index),
        device=input_ids.device,
        dtype=torch.long,
    )

    input_ids_long = input_ids.long()

    for k in range(mtp_depth):
        shift = k + 2

        if shift >= T:
            continue

        future = input_ids_long[:, shift:]  # [B, T-shift]
        labels[:, k, : T - shift] = future

        if pad_token_id is not None:
            is_pad = labels[:, k, :] == int(pad_token_id)
            labels[:, k, :] = torch.where(
                is_pad,
                torch.full_like(labels[:, k, :], int(ignore_index)),
                labels[:, k, :],
            )

    return labels


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MTPConfig:
    d_model: int
    vocab_size: int
    mtp_depth: int = 1

    hidden_dim: Optional[int] = None
    use_mtp_transform: bool = True
    activation: str = "silu"

    dropout: float = 0.0
    use_bias: bool = False
    init_std: float = 0.02

    tie_with_lm_head: bool = False
    mtp_loss_weight: float = 0.3

    # Explicit ignore index for CE. This is distinct from pad_token_id.
    ignore_index: int = -100

    # Optional tokenizer pad id. If provided, label validation allows it as
    # a token id, and build_mtp_labels can convert future-pad targets to
    # ignore_index.
    pad_token_id: Optional[int] = None

    # Optional per-depth loss weights. If None, all depths are averaged equally.
    # If provided, must have length mtp_depth and non-negative values.
    depth_loss_weights: Optional[Tuple[float, ...]] = None

    validate_label_range: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")

        if self.mtp_depth <= 0:
            raise ValueError(f"mtp_depth must be > 0, got {self.mtp_depth}")

        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0 when provided, got {self.hidden_dim}"
            )

        valid_activations = {"silu", "gelu", "relu", "identity"}
        if self.activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got {self.activation}"
            )

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {self.dropout}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.mtp_loss_weight < 0:
            raise ValueError(
                f"mtp_loss_weight must be >= 0, got {self.mtp_loss_weight}"
            )

        if self.pad_token_id is not None:
            if not (0 <= self.pad_token_id < self.vocab_size):
                raise ValueError(
                    "pad_token_id must satisfy 0 <= pad_token_id < vocab_size. "
                    f"Got pad_token_id={self.pad_token_id}, vocab_size={self.vocab_size}"
                )

        if self.depth_loss_weights is not None:
            if len(self.depth_loss_weights) != self.mtp_depth:
                raise ValueError(
                    "depth_loss_weights must have length mtp_depth. "
                    f"Got len={len(self.depth_loss_weights)}, mtp_depth={self.mtp_depth}"
                )

            if any(w < 0 for w in self.depth_loss_weights):
                raise ValueError("depth_loss_weights must be non-negative.")

            if sum(self.depth_loss_weights) <= 0:
                raise ValueError("At least one depth_loss_weight must be > 0.")


# ============================================================
# ACTIVATION
# ============================================================

class MTPActivation(nn.Module):
    def __init__(self, activation: str):
        super().__init__()

        valid_activations = {"silu", "gelu", "relu", "identity"}
        if activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got {activation}"
            )

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            return F.silu(x)

        if self.activation == "gelu":
            return F.gelu(x)

        if self.activation == "relu":
            return F.relu(x)

        if self.activation == "identity":
            return x

        raise RuntimeError(f"Unknown activation={self.activation}")


# ============================================================
# MTP TRANSFORM
# ============================================================

class MTPTransform(nn.Module):
    """
    Per-depth transformation before the MTP vocabulary head.

    If enabled:
        RMSNorm(d_model)
        Linear(d_model, hidden_dim)
        activation
        Linear(hidden_dim, d_model)
        Dropout

    Input:
        hidden_states: [B,T,D]

    Output:
        transformed: [B,T,D]
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        use_bias: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        if hidden_dim is None:
            hidden_dim = d_model

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")

        if init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {init_std}")

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.init_std = init_std

        self.norm = RMSNorm(dim=d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=use_bias)
        self.act = MTPActivation(activation)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [self.fc1, self.fc2]:
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"MTPTransform expects hidden_states [B,T,D], got {tuple(hidden_states.shape)}"
            )

        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden size {self.d_model}, got {hidden_states.shape[-1]}"
            )

        h = self.norm(hidden_states)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)

        return h