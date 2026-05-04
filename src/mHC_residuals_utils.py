# ============================================================
# Mini DeepSeek-V4 mHC utilities
# Manifold-Constrained Hyper-Connections - canonical mini version
# ============================================================

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Tuple, Union

import math
import torch
import torch.nn as nn


# ============================================================
# Sinkhorn projection
# ============================================================

def sinkhorn(
    logits: torch.Tensor,
    n_iters: int = 20,
    eps: float = 1e-6,
    fp32: bool = True,
) -> torch.Tensor:
    """
    Project logits to an approximately doubly stochastic matrix.

    This is the standard Sinkhorn-Knopp projection used for mHC:

        M^(0) = exp(logits)
        M <- row_normalize(column_normalize(M))

    The implementation stabilizes the exponentiation and optionally performs
    the normalization in fp32 for safer mixed-precision behavior.

    Args:
        logits:
            Tensor with shape [..., N, N].
        n_iters:
            Number of normalization iterations.
        eps:
            Numerical stabilizer.
        fp32:
            If True, computes the projection in fp32 and casts back.

    Returns:
        M:
            Tensor with shape [..., N, N]. Approximately:
                M >= 0
                M.sum(dim=-1) ~= 1
                M.sum(dim=-2) ~= 1
    """
    if logits.dim() < 2:
        raise ValueError(
            f"sinkhorn expects logits with at least 2 dims, got {tuple(logits.shape)}"
        )

    if logits.shape[-1] != logits.shape[-2]:
        raise ValueError(
            "sinkhorn expects square matrices in the last two dimensions, "
            f"got {tuple(logits.shape[-2:])}"
        )

    if n_iters <= 0:
        raise ValueError(f"n_iters must be > 0, got {n_iters}")

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    orig_dtype = logits.dtype
    work = logits.float() if fp32 else logits

    # Stabilize before exponentiation. Subtracting a shared max preserves
    # the Sinkhorn solution up to a global positive scale.
    max_val = work.amax(dim=(-1, -2), keepdim=True)
    M = torch.exp(work - max_val)

    # Paper notation applies column and row normalization iteratively.
    # Alternating row/column is equivalent up to iteration order; after enough
    # iterations both marginals approach one.
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)

    return M.to(dtype=orig_dtype)


def log_sinkhorn(
    logits: torch.Tensor,
    n_iters: int = 20) -> torch.Tensor:
    
    """
    Log-domain Sinkhorn projection.

    This is a numerically robust alternative to `sinkhorn`. It is useful for
    large logits or aggressive mixed-precision experiments. It returns a normal
    probability matrix, not log-probabilities.
    """
    if logits.dim() < 2:
        raise ValueError(
            f"log_sinkhorn expects logits with at least 2 dims, got {tuple(logits.shape)}"
        )

    if logits.shape[-1] != logits.shape[-2]:
        raise ValueError(
            "log_sinkhorn expects square matrices in the last two dimensions, "
            f"got {tuple(logits.shape[-2:])}"
        )

    if n_iters <= 0:
        raise ValueError(f"n_iters must be > 0, got {n_iters}")

    orig_dtype = logits.dtype
    log_M = logits.float()

    for _ in range(n_iters):
        log_M = log_M - torch.logsumexp(log_M, dim=-1, keepdim=True)
        log_M = log_M - torch.logsumexp(log_M, dim=-2, keepdim=True)

    return torch.exp(log_M).to(dtype=orig_dtype)


# ============================================================
# Expand / collapse residual streams
# ============================================================

def expand_residual_stream(
    x: torch.Tensor,
    n_hc: int,
    mode: str = "first",
) -> torch.Tensor:
    """
    Expand a standard residual stream into n_hc hyper-connection streams.

    Args:
        x: [B, T, D]
        n_hc: number of residual streams.
        mode:
            "first": stream 0 = x, streams 1..n_hc-1 = 0.
            "mean": every stream receives x / n_hc.
            "repeat": every stream receives x.

    Returns:
        X: [B, T, n_hc, D]
    """
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B,T,D], got {tuple(x.shape)}")

    if n_hc <= 0:
        raise ValueError(f"n_hc must be > 0, got {n_hc}")

    B, T, D = x.shape

    if mode == "first":
        X = x.new_zeros(B, T, n_hc, D)
        X[:, :, 0, :] = x
        return X

    if mode == "mean":
        return x[:, :, None, :].expand(B, T, n_hc, D) / n_hc

    if mode == "repeat":
        return x[:, :, None, :].expand(B, T, n_hc, D).clone()

    raise ValueError(f"Unknown expand mode: {mode}")


def collapse_residual_stream(
    X: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Collapse expanded residual stream back to [B,T,D].

    Args:
        X: [B, T, n_hc, D]
        mode:
            "mean": average over hyper streams.
            "first": take stream 0.
            "sum": sum over hyper streams.

    Returns:
        x: [B, T, D]
    """
    if X.dim() != 4:
        raise ValueError(f"X must have shape [B,T,n_hc,D], got {tuple(X.shape)}")

    if mode == "mean":
        return X.mean(dim=2)

    if mode == "first":
        return X[:, :, 0, :]

    if mode == "sum":
        return X.sum(dim=2)

    raise ValueError(f"Unknown collapse mode: {mode}")


class HyperConnectionReadout(nn.Module):
    """
    Learnable readout from [B,T,n_hc,D] to [B,T,D].
    """

    def __init__(self, n_hc: int, init: str = "mean"):
        super().__init__()
        if n_hc <= 0:
            raise ValueError(f"n_hc must be > 0, got {n_hc}")
        self.n_hc = n_hc
        self.logits = nn.Parameter(torch.empty(n_hc))
        self.reset_parameters(init=init)

    def reset_parameters(self, init: str = "mean") -> None:
        with torch.no_grad():
            if init == "mean":
                self.logits.zero_()
            elif init == "first":
                self.logits.fill_(-6.0)
                self.logits[0] = 6.0
            else:
                raise ValueError(f"Unknown readout init: {init}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 4:
            raise ValueError(f"X must have shape [B,T,n_hc,D], got {tuple(X.shape)}")
        if X.shape[2] != self.n_hc:
            raise ValueError(
                f"Expected X.shape[2] == n_hc={self.n_hc}, got {X.shape[2]}"
            )
        weights = torch.softmax(self.logits, dim=0).to(dtype=X.dtype)
        return torch.einsum("n,btnd->btd", weights, X)