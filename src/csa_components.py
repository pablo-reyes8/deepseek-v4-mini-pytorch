# ============================================================
# Mini DeepSeek-V4 Canonical CSA
# Compressed Sparse Attention with overlapped a/b compression
# ============================================================

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CSAConfig:
    d_model: int
    n_heads: int
    head_dim: Optional[int] = None

    compression_factor: int = 8
    top_k: int = 8
    window_size: int = 32

    indexer_dim: int = 32
    n_indexer_heads: int = 2
    query_compression_dim: Optional[int] = None

    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    use_bias: bool = False

    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None

    max_seq_len: int = 1024
    init_std: float = 0.02

    # More canonical / final CSA options
    use_attention_sink: bool = True
    use_grouped_output_projection: bool = True
    output_projection_groups: Optional[int] = None

    # Canonical default: indexer selects blocks, but does not directly bias
    # core attention logits. Enable for experimental/trainability purposes.
    use_indexer_score_bias: bool = False

    # Keep the local exact sliding-window KV branch conceptually separate from
    # the global compressed a/b KV construction.
    use_separate_local_kv: bool = True

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")

        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    "If head_dim is None, d_model must be divisible by n_heads. "
                    f"Got d_model={self.d_model}, n_heads={self.n_heads}"
                )
            head_dim = self.d_model // self.n_heads
        else:
            head_dim = self.head_dim

        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")

        if self.n_heads * head_dim != self.d_model:
            raise ValueError(
                "For CSA v1, n_heads * head_dim must equal d_model. "
                f"Got n_heads={self.n_heads}, head_dim={head_dim}, "
                f"d_model={self.d_model}"
            )

        if self.compression_factor <= 0:
            raise ValueError(
                f"compression_factor must be > 0, got {self.compression_factor}"
            )

        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")

        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        if self.indexer_dim <= 0:
            raise ValueError(f"indexer_dim must be > 0, got {self.indexer_dim}")

        if self.n_indexer_heads <= 0:
            raise ValueError(
                f"n_indexer_heads must be > 0, got {self.n_indexer_heads}"
            )

        if self.query_compression_dim is not None and self.query_compression_dim <= 0:
            raise ValueError(
                "query_compression_dim must be > 0 when provided, "
                f"got {self.query_compression_dim}"
            )

        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(
                "attention_dropout must satisfy 0 <= attention_dropout < 1, "
                f"got {self.attention_dropout}"
            )

        if not (0.0 <= self.residual_dropout < 1.0):
            raise ValueError(
                "residual_dropout must satisfy 0 <= residual_dropout < 1, "
                f"got {self.residual_dropout}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")

        if self.rotary_dim is not None:
            if self.rotary_dim <= 0:
                raise ValueError(
                    f"rotary_dim must be > 0 when provided, got {self.rotary_dim}"
                )

            if self.rotary_dim > head_dim:
                raise ValueError(
                    f"rotary_dim must be <= head_dim. "
                    f"Got rotary_dim={self.rotary_dim}, head_dim={head_dim}"
                )

            if self.rotary_dim % 2 != 0:
                raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")

        if self.output_projection_groups is not None:
            if self.output_projection_groups <= 0:
                raise ValueError(
                    "output_projection_groups must be > 0 when provided, "
                    f"got {self.output_projection_groups}"
                )

            if self.n_heads % self.output_projection_groups != 0:
                raise ValueError(
                    "n_heads must be divisible by output_projection_groups. "
                    f"Got n_heads={self.n_heads}, "
                    f"output_projection_groups={self.output_projection_groups}"
                )


# ============================================================
# SAFE MASKED SOFTMAX
# ============================================================

def safe_masked_softmax(
    scores: torch.Tensor,
    allowed_mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Safe masked softmax.

    Args:
        scores:
            Arbitrary score tensor.

        allowed_mask:
            Boolean tensor broadcastable to scores.
            True = allowed.
            False = masked.

        dim:
            Softmax dimension.

    Returns:
        weights:
            Same shape as scores.
            Rows with no allowed keys become exactly zero.
    """
    if allowed_mask.dtype != torch.bool:
        allowed_mask = allowed_mask.bool()

    mask_value = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~allowed_mask, mask_value)

    weights = F.softmax(masked_scores.float(), dim=dim).to(dtype=scores.dtype)
    weights = weights * allowed_mask.to(dtype=weights.dtype)

    denom = weights.sum(dim=dim, keepdim=True)

    weights = torch.where(
        denom > 0,
        weights / denom.clamp_min(torch.finfo(weights.dtype).tiny),
        torch.zeros_like(weights),
    )

    return weights


# ============================================================
# GROUPED OUTPUT PROJECTION
# ============================================================

class GroupedOutputProjection(nn.Module):
    """
    Simple PyTorch grouped output projection.

    Input:
        x: [B, T, H, Dh]

    Output:
        out: [B, T, H * Dh]

    This keeps the grouped-output-projection idea without custom kernels.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        num_groups: int,
        bias: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()

        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {num_groups}")
        if n_heads % num_groups != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by num_groups={num_groups}"
            )

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.heads_per_group = n_heads // num_groups
        self.group_dim = self.heads_per_group * head_dim
        self.init_std = init_std

        self.group_projs = nn.ModuleList(
            [
                nn.Linear(self.group_dim, self.group_dim, bias=bias)
                for _ in range(num_groups)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for proj in self.group_projs:
            nn.init.normal_(proj.weight, mean=0.0, std=self.init_std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B,T,H,Dh], got {tuple(x.shape)}")

        B, T, H, Dh = x.shape
        if H != self.n_heads:
            raise ValueError(f"Expected H={self.n_heads}, got {H}")
        if Dh != self.head_dim:
            raise ValueError(f"Expected Dh={self.head_dim}, got {Dh}")

        group_outputs = []
        for g, proj in enumerate(self.group_projs):
            h_start = g * self.heads_per_group
            h_end = (g + 1) * self.heads_per_group

            x_g = x[:, :, h_start:h_end, :].reshape(B, T, self.group_dim)
            y_g = proj(x_g)
            group_outputs.append(y_g)

        return torch.cat(group_outputs, dim=-1)


# ============================================================
# OVERLAPPED A/B COMPRESSOR
# ============================================================

class CSAOverlappedCompressor(nn.Module):
    """
    Canonical CSA overlapped a/b compressor.

    For compressed block i:
        A branch uses current block:
            [i*m, min((i+1)*m, T))

        B branch uses previous block:
            [(i-1)*m, i*m)

    For i = 0:
        B branch is empty/invalid.

    Input:
        C_a: [B, T, D]
        C_b: [B, T, D]
        Z_a: [B, T, D]
        Z_b: [B, T, D]

    Output:
        C_comp: [B, S, D]
        comp_valid_mask: [B, S]
        comp_position_ids: [S] or [B, S]
    """

    def __init__(
        self,
        compression_factor: int,
        dim: int,
        init_std: float = 0.02,
    ):
        super().__init__()

        if compression_factor <= 0:
            raise ValueError(
                f"compression_factor must be > 0, got {compression_factor}"
            )

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {init_std}")

        self.compression_factor = compression_factor
        self.dim = dim
        self.init_std = init_std

        self.bias_a = nn.Parameter(torch.empty(compression_factor, dim))
        self.bias_b = nn.Parameter(torch.empty(compression_factor, dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bias_a, mean=0.0, std=self.init_std)
        nn.init.normal_(self.bias_b, mean=0.0, std=self.init_std)

    def _safe_temporal_softmax(
        self,
        scores: torch.Tensor,
        valid: Optional[torch.Tensor],
        dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Safe softmax over temporal dimension.

        Args:
            scores: [B, L, D]
            valid:  None or [B, L]

        Returns:
            weights: [B, L, D]
            block_valid: [B]
        """
        B, L, _ = scores.shape

        if valid is None:
            block_valid = torch.ones(B, device=scores.device, dtype=torch.bool)
            weights = F.softmax(scores.float(), dim=dim).to(dtype=scores.dtype)
            return weights, block_valid

        if valid.shape != (B, L):
            raise ValueError(
                f"valid must have shape {(B, L)}, got {tuple(valid.shape)}"
            )

        valid_bool = valid.to(device=scores.device, dtype=torch.bool)
        block_valid = valid_bool.any(dim=1)

        weights = safe_masked_softmax(
            scores=scores,
            allowed_mask=valid_bool[:, :, None],
            dim=dim,
        )

        return weights, block_valid

    def _build_comp_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        num_blocks: int,
        device: torch.device,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Compressed block position = position of last token in current A block.
        """
        m = self.compression_factor

        if position_ids is None:
            positions = []
            for i in range(num_blocks):
                a_end = min((i + 1) * m, seq_len)
                positions.append(start_pos + a_end - 1)

            return torch.tensor(positions, device=device, dtype=torch.long)

        if position_ids.device != device:
            position_ids = position_ids.to(device)

        if position_ids.dim() == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError(
                    f"position_ids [T] must have length {seq_len}, "
                    f"got {position_ids.shape[0]}"
                )

            positions = []
            for i in range(num_blocks):
                a_end = min((i + 1) * m, seq_len)
                positions.append(position_ids[a_end - 1])

            return torch.stack(positions, dim=0)

        if position_ids.dim() == 2:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids [B,T] must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(position_ids.shape)}"
                )

            positions = []
            for i in range(num_blocks):
                a_end = min((i + 1) * m, seq_len)
                positions.append(position_ids[:, a_end - 1])

            return torch.stack(positions, dim=1)

        raise ValueError(
            "position_ids must be None, [T], or [B,T], "
            f"got {tuple(position_ids.shape)}"
        )

    def forward(
        self,
        C_a: torch.Tensor,
        C_b: torch.Tensor,
        Z_a: torch.Tensor,
        Z_b: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if C_a.dim() != 3:
            raise ValueError(f"C_a must have shape [B,T,D], got {tuple(C_a.shape)}")

        if C_b.shape != C_a.shape:
            raise ValueError("C_b must have same shape as C_a")

        if Z_a.shape != C_a.shape:
            raise ValueError("Z_a must have same shape as C_a")

        if Z_b.shape != C_a.shape:
            raise ValueError("Z_b must have same shape as C_a")

        B, T, D = C_a.shape

        if D != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {D}")

        if attention_mask is not None:
            if attention_mask.shape != (B, T):
                raise ValueError(
                    f"attention_mask must have shape {(B, T)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

        m = self.compression_factor
        S = math.ceil(T / m)

        comp_blocks = []
        valid_blocks = []

        for i in range(S):
            # A branch: current block
            a_start = i * m
            a_end = min((i + 1) * m, T)
            a_len = a_end - a_start

            A_tokens = C_a[:, a_start:a_end, :]
            A_scores = Z_a[:, a_start:a_end, :] + self.bias_a[:a_len, :].to(
                device=C_a.device,
                dtype=Z_a.dtype,
            )[None, :, :]

            if attention_mask is None:
                valid_a = None
            else:
                valid_a = attention_mask[:, a_start:a_end]

            # B branch: previous block
            if i == 0:
                B_tokens = C_b.new_zeros(B, 0, D)
                B_scores = Z_b.new_zeros(B, 0, D)
                valid_b = None if attention_mask is None else attention_mask.new_zeros(B, 0)
            else:
                b_start = (i - 1) * m
                b_end = i * m
                b_len = b_end - b_start

                B_tokens = C_b[:, b_start:b_end, :]
                B_scores = Z_b[:, b_start:b_end, :] + self.bias_b[:b_len, :].to(
                    device=C_b.device,
                    dtype=Z_b.dtype,
                )[None, :, :]

                if attention_mask is None:
                    valid_b = None
                else:
                    valid_b = attention_mask[:, b_start:b_end]

            tokens = torch.cat([A_tokens, B_tokens], dim=1)
            scores = torch.cat([A_scores, B_scores], dim=1)

            if attention_mask is None:
                valid = None
            else:
                valid = torch.cat([valid_a, valid_b], dim=1)

            weights, block_valid = self._safe_temporal_softmax(
                scores=scores,
                valid=valid,
                dim=1,
            )

            comp_i = (weights * tokens).sum(dim=1)

            comp_i = torch.where(
                block_valid[:, None],
                comp_i,
                torch.zeros_like(comp_i),
            )

            comp_blocks.append(comp_i)
            valid_blocks.append(block_valid)

        C_comp = torch.stack(comp_blocks, dim=1)
        comp_valid_mask = torch.stack(valid_blocks, dim=1)

        comp_position_ids = self._build_comp_position_ids(
            position_ids=position_ids,
            batch_size=B,
            seq_len=T,
            num_blocks=S,
            device=C_a.device,
            start_pos=start_pos,
        )

        return C_comp, comp_valid_mask, comp_position_ids