# ============================================================
# Mini DeepSeek-V4 HCAAttention
# Heavily Compressed Attention - more canonical version
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
class HCAConfig:
    d_model: int
    n_heads: int

    head_dim: Optional[int] = None

    compression_factor: int = 16
    window_size: int = 32

    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    use_bias: bool = False

    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None

    max_seq_len: int = 1024
    init_std: float = 0.02

    # More canonical details
    use_attention_sink: bool = True
    use_grouped_output_projection: bool = True
    output_projection_groups: Optional[int] = None

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

        inner_dim = self.n_heads * head_dim

        if inner_dim != self.d_model:
            raise ValueError(
                "For this HCA implementation, n_heads * head_dim must equal d_model. "
                f"Got n_heads={self.n_heads}, head_dim={head_dim}, "
                f"inner_dim={inner_dim}, d_model={self.d_model}"
            )

        if self.compression_factor <= 0:
            raise ValueError(
                f"compression_factor must be > 0, got {self.compression_factor}"
            )

        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0 for HCA, got {self.window_size}"
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
# TOKEN COMPRESSOR
# ============================================================

class HCATokenCompressor(nn.Module):
    """
    Token-level compressor for HCA.

    Given token-level KV entries C and compression logits Z, each block of
    compression_factor tokens is compressed into one shared KV entry.

    Input:
        C: [B, T, Dh]
        Z: [B, T, Dh]
        attention_mask: optional [B, T]

    Output:
        compressed_C: [B, S, Dh]
        compressed_valid_mask: [B, S]
        compressed_position_ids: [S]

    where:
        S = ceil(T / compression_factor)

    Compression rule:
        scores = Z_block + learned_position_bias
        weights = softmax(scores over block tokens)
        compressed = sum_token weights[token] * C[token]
    """

    def __init__(
        self,
        compression_factor: int,
        head_dim: int,
        init_std: float = 0.02,
    ):
        super().__init__()

        if compression_factor <= 0:
            raise ValueError(
                f"compression_factor must be > 0, got {compression_factor}"
            )

        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")

        if init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {init_std}")

        self.compression_factor = compression_factor
        self.head_dim = head_dim
        self.init_std = init_std

        self.compression_bias = nn.Parameter(
            torch.empty(compression_factor, head_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.compression_bias,
            mean=0.0,
            std=self.init_std,
        )

    def _safe_block_softmax(
        self,
        scores: torch.Tensor,
        valid_block: Optional[torch.Tensor],
        dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Safe softmax over block tokens.

        Args:
            scores:
                [B, block_len, Dh]

            valid_block:
                None or [B, block_len]
                1 = valid token
                0 = pad token

        Returns:
            weights:
                [B, block_len, Dh]

            block_valid:
                [B], bool
                True if the block contains at least one valid token.
        """
        B, block_len, _ = scores.shape

        if valid_block is None:
            block_valid = torch.ones(
                B,
                device=scores.device,
                dtype=torch.bool,
            )

            weights = F.softmax(scores.float(), dim=dim).to(dtype=scores.dtype)
            return weights, block_valid

        if valid_block.shape != (B, block_len):
            raise ValueError(
                f"valid_block must have shape {(B, block_len)}, "
                f"got {tuple(valid_block.shape)}"
            )

        valid_bool = valid_block.to(device=scores.device, dtype=torch.bool)
        block_valid = valid_bool.any(dim=1)

        mask_value = torch.finfo(scores.dtype).min

        masked_scores = scores.masked_fill(
            ~valid_bool[:, :, None],
            mask_value,
        )

        weights = F.softmax(masked_scores.float(), dim=dim).to(dtype=scores.dtype)

        # Remove probability mass from masked tokens.
        weights = weights * valid_bool[:, :, None].to(dtype=weights.dtype)

        denom = weights.sum(dim=dim, keepdim=True)

        weights = torch.where(
            denom > 0,
            weights / denom.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )

        return weights, block_valid

    def forward(
        self,
        C: torch.Tensor,
        Z: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if C.dim() != 3:
            raise ValueError(f"C must have shape [B, T, Dh], got {tuple(C.shape)}")

        if Z.shape != C.shape:
            raise ValueError(
                f"Z must have same shape as C. "
                f"Got Z={tuple(Z.shape)}, C={tuple(C.shape)}"
            )

        B, T, Dh = C.shape

        if Dh != self.head_dim:
            raise ValueError(
                f"Expected C.shape[-1] == head_dim={self.head_dim}, got {Dh}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (B, T):
                raise ValueError(
                    f"attention_mask must have shape {(B, T)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

        m = self.compression_factor
        S = math.ceil(T / m)

        compressed_blocks = []
        valid_blocks = []
        compressed_positions = []

        for i in range(S):
            start = i * m
            end = min((i + 1) * m, T)
            block_len = end - start

            C_block = C[:, start:end, :]  # [B, block_len, Dh]
            Z_block = Z[:, start:end, :]  # [B, block_len, Dh]

            bias_block = self.compression_bias[:block_len, :]
            bias_block = bias_block.to(device=C.device, dtype=Z.dtype)

            scores = Z_block + bias_block[None, :, :]

            if attention_mask is None:
                valid_block = None
            else:
                valid_block = attention_mask[:, start:end]

            weights, block_valid = self._safe_block_softmax(
                scores=scores,
                valid_block=valid_block,
                dim=1,
            )

            compressed_i = (weights * C_block).sum(dim=1)

            compressed_i = torch.where(
                block_valid[:, None],
                compressed_i,
                torch.zeros_like(compressed_i),
            )

            compressed_blocks.append(compressed_i)
            valid_blocks.append(block_valid)

            # Canonical convention for compressed RoPE:
            # the compressed block receives the position of its last token.
            compressed_positions.append(start_pos + end - 1)

        compressed_C = torch.stack(compressed_blocks, dim=1)
        compressed_valid_mask = torch.stack(valid_blocks, dim=1)

        compressed_position_ids = torch.tensor(
            compressed_positions,
            device=C.device,
            dtype=torch.long,
        )

        return compressed_C, compressed_valid_mask, compressed_position_ids


# ============================================================
# GROUPED OUTPUT PROJECTION
# ============================================================

class GroupedOutputProjection(nn.Module):
    """
    Simple PyTorch grouped output projection.

    Instead of using one dense projection over all heads jointly, this module
    splits the attention output by groups of heads and applies an independent
    projection inside each group.

    Input:
        x: [B, T, H, Dh]

    Output:
        out: [B, T, H * Dh]

    This is a clean mini version of grouped output projection. It keeps the
    architectural idea without requiring custom kernels.
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
            raise ValueError(f"x must have shape [B, T, H, Dh], got {tuple(x.shape)}")

        B, T, H, Dh = x.shape

        if H != self.n_heads:
            raise ValueError(f"Expected H={self.n_heads}, got {H}")

        if Dh != self.head_dim:
            raise ValueError(f"Expected Dh={self.head_dim}, got {Dh}")

        group_outputs = []

        for g, proj in enumerate(self.group_projs):
            h_start = g * self.heads_per_group
            h_end = (g + 1) * self.heads_per_group

            x_g = x[:, :, h_start:h_end, :]  # [B, T, Hg, Dh]
            x_g = x_g.reshape(B, T, self.group_dim)

            y_g = proj(x_g)                  # [B, T, group_dim]
            group_outputs.append(y_g)

        out = torch.cat(group_outputs, dim=-1)  # [B, T, H * Dh]
        return out