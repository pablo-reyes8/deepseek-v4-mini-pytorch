# ============================================================
# Mini DeepSeek-V4 LM configurable
# Final full-sequence forward wrapper
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.deepseek_csa_attention import * 
from src.deepseek_hca_attention import * 
from src.deepseek_moe import * 
from src.deepseek_mtp import *  
from src.mHC_residuals import * 
from src.transformer_modules.mha_baseline import *  

# ============================================================
# CONFIG
# ============================================================

@dataclass
class DeepSeekV4LMConfig:
    # Core
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int = 1024
    pad_token_id: Optional[int] = None
    ignore_index: int = -100

    # Loss semantics
    # labels_are_shifted=True:
    #   input_ids: [x0, x1, ..., xT-1]
    #   labels:    [x1, x2, ..., xT]
    #   loss uses logits[:, :, :] vs labels[:, :]
    #
    # labels_are_shifted=False:
    #   input_ids: [x0, x1, ..., xT-1]
    #   labels:    [x0, x1, ..., xT-1], usually labels=input_ids
    #   loss uses logits[:, :-1, :] vs labels[:, 1:]
    labels_are_shifted: bool = True
    ignore_pad_token_in_loss: bool = True


    # Embedding
    embedding_dropout: float = 0.0
    scale_embeddings: bool = False
    tie_word_embeddings: bool = True

    # Norm
    rms_norm_eps: float = 1e-6

    # Attention selection
    # "mha"    -> all layers use standard MHA
    # "hca"    -> all layers use HCA
    # "csa"    -> all layers use CSA
    # "hybrid" -> interleaves attention modules by layer according to attention_pattern
    attention_type: str = "hybrid"  # "mha", "hca", "csa", "hybrid"
    attention_pattern: Tuple[str, ...] = ("csa", "hca")
    n_heads: int = 4
    head_dim: Optional[int] = None
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    use_attention_bias: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None

    # HCA / CSA shared
    compression_factor: int = 8
    hca_compression_factor: int = 16
    window_size: int = 32

    # CSA
    top_k_blocks: int = 8
    indexer_dim: int = 32
    n_indexer_heads: int = 2
    query_compression_dim: Optional[int] = None

    # Canonical CSA extras
    use_attention_sink: bool = True
    use_grouped_output_projection: bool = True
    output_projection_groups: Optional[int] = None
    use_indexer_score_bias: bool = False
    use_separate_local_kv: bool = True

    # FFN selection
    ffn_type: str = "moe"  # "dense", "moe"

    # Dense MLP
    mlp_hidden_dim: Optional[int] = None
    mlp_expansion_factor: float = 4.0
    mlp_multiple_of: int = 1
    mlp_dropout: float = 0.0
    use_mlp_bias: bool = False

    # MoE
    num_experts: int = 8
    top_k_experts: int = 2
    expert_hidden_dim: Optional[int] = None
    expert_expansion_factor: float = 4.0
    expert_multiple_of: int = 1
    shared_experts: int = 1
    shared_hidden_dim: Optional[int] = None
    shared_expansion_factor: float = 4.0

    router_type: str = "learned"  # "learned", "hash"
    router_score_fn: str = "sqrt_softplus"
    normalize_topk_weights: bool = True
    topk_weight_scale: float = 1.0
    router_jitter_noise: float = 0.0
    hash_routing_stride: int = 1

    routed_scale: float = 1.0
    shared_scale: float = 1.0

    balance_loss_weight: float = 0.0
    sequence_balance_loss_weight: float = 0.0

    # mHC
    use_mhc: bool = False
    n_hc: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_eps: float = 1e-6
    mhc_dynamic: bool = True
    mhc_expand_mode: str = "first"
    mhc_collapse_mode: str = "readout"  # "mean", "first", "sum", "readout"

    # Optional canonical mHC extras
    mhc_use_log_sinkhorn: bool = False
    mhc_sinkhorn_fp32: bool = True
    mhc_init_alpha: float = 1e-3
    mhc_alpha_max: float = 1.0
    mhc_bounded_alpha: bool = True

    # MTP
    use_mtp: bool = False
    mtp_depth: int = 1
    mtp_hidden_dim: Optional[int] = None
    use_mtp_transform: bool = True
    mtp_activation: str = "silu"
    mtp_dropout: float = 0.0
    mtp_loss_weight: float = 0.3
    mtp_tie_with_lm_head: bool = False
    mtp_depth_loss_weights: Optional[Tuple[float, ...]] = None
    mtp_validate_label_range: bool = True

    # Init
    init_std: float = 0.02

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")

        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if self.pad_token_id is not None:
            if not (0 <= self.pad_token_id < self.vocab_size):
                raise ValueError(
                    "pad_token_id must satisfy 0 <= pad_token_id < vocab_size. "
                    f"Got pad_token_id={self.pad_token_id}, vocab_size={self.vocab_size}"
                )

        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be > 0, got {self.rms_norm_eps}")

        if self.attention_type not in {"mha", "hca", "csa", "hybrid"}:
            raise ValueError(
                f"attention_type must be one of {{'mha','hca','csa','hybrid'}}, "
                f"got {self.attention_type}"
            )

        if self.attention_type == "hybrid":
            if not isinstance(self.attention_pattern, tuple):
                raise ValueError(
                    f"attention_pattern must be a tuple, got {type(self.attention_pattern)}"
                )

            if len(self.attention_pattern) == 0:
                raise ValueError(
                    "attention_pattern must be non-empty when attention_type='hybrid'"
                )

            for attn_name in self.attention_pattern:
                if attn_name not in {"mha", "hca", "csa"}:
                    raise ValueError(
                        "Every element of attention_pattern must be one of "
                        f"{{'mha','hca','csa'}}, got {attn_name}"
                    )

        if self.ffn_type not in {"dense", "moe"}:
            raise ValueError(
                f"ffn_type must be one of {{'dense','moe'}}, got {self.ffn_type}"
            )

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.use_mhc:
            if self.n_hc < 2:
                raise ValueError(f"n_hc must be >= 2 when use_mhc=True, got {self.n_hc}")
            if self.mhc_collapse_mode not in {"mean", "first", "sum", "readout"}:
                raise ValueError(
                    "mhc_collapse_mode must be one of "
                    "{'mean','first','sum','readout'}, "
                    f"got {self.mhc_collapse_mode}"
                )

        if self.router_type not in {"learned", "hash"}:
            raise ValueError(
                f"router_type must be one of {'learned', 'hash'}, got {self.router_type}"
            )

        if self.hash_routing_stride <= 0:
            raise ValueError(
                f"hash_routing_stride must be > 0, got {self.hash_routing_stride}"
            )

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")

        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    "If head_dim is None, d_model must be divisible by n_heads. "
                    f"Got d_model={self.d_model}, n_heads={self.n_heads}"
                )
        else:
            if self.head_dim <= 0:
                raise ValueError(f"head_dim must be > 0, got {self.head_dim}")

        if self.rotary_dim is not None:
            effective_head_dim = (
                self.head_dim if self.head_dim is not None else self.d_model // self.n_heads
            )
            if self.rotary_dim <= 0:
                raise ValueError(f"rotary_dim must be > 0, got {self.rotary_dim}")
            if self.rotary_dim > effective_head_dim:
                raise ValueError(
                    f"rotary_dim must be <= head_dim. "
                    f"Got rotary_dim={self.rotary_dim}, head_dim={effective_head_dim}"
                )
            if self.rotary_dim % 2 != 0:
                raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")

        if self.use_mtp:
            if self.mtp_depth <= 0:
                raise ValueError(f"mtp_depth must be > 0, got {self.mtp_depth}")
            if self.mtp_loss_weight < 0:
                raise ValueError(
                    f"mtp_loss_weight must be >= 0, got {self.mtp_loss_weight}"
                )

        if not isinstance(self.labels_are_shifted, bool):
            raise ValueError(
                f"labels_are_shifted must be bool, got {type(self.labels_are_shifted)}"
            )

        if not isinstance(self.ignore_pad_token_in_loss, bool):
            raise ValueError(
                "ignore_pad_token_in_loss must be bool, "
                f"got {type(self.ignore_pad_token_in_loss)}"
            )

# ============================================================
# SMALL COMPAT HELPERS
# ============================================================

def _supports_kwarg(module_or_fn, name: str) -> bool:
    try:
        sig = inspect.signature(module_or_fn.forward if isinstance(module_or_fn, nn.Module) else module_or_fn)
        return name in sig.parameters
    except Exception:
        return False


def _get_token_embedding_weight(embedding: nn.Module) -> nn.Parameter:
    """
    Supports both:
        TokenEmbedding.weight
    and common internal names:
        TokenEmbedding.embedding.weight
        TokenEmbedding.token_embedding.weight
    """
    if hasattr(embedding, "weight"):
        return embedding.weight

    if hasattr(embedding, "embedding") and hasattr(embedding.embedding, "weight"):
        return embedding.embedding.weight

    if hasattr(embedding, "token_embedding") and hasattr(embedding.token_embedding, "weight"):
        return embedding.token_embedding.weight

    raise AttributeError(
        "Could not find token embedding weight. Expected embedding.weight, "
        "embedding.embedding.weight, or embedding.token_embedding.weight."
    )


# ============================================================
# FACTORIES
# ============================================================

def build_deepseek_attention(
    config: DeepSeekV4LMConfig,
    attention_type: Optional[str] = None,
) -> nn.Module:
    """
    Build one concrete attention module.

    Important:
        attention_type="hybrid" is a model-level scheduling mode, not a
        concrete attention module. Resolve it per layer with
        get_layer_attention_type(config, layer_idx) before calling this factory.
    """

    if attention_type is None:
        attention_type = config.attention_type

    if attention_type == "hybrid":
        raise ValueError(
            "build_deepseek_attention received attention_type='hybrid'. "
            "Resolve the concrete layer attention first with "
            "get_layer_attention_type(config, layer_idx)."
        )

    if attention_type == "mha":
        return CausalMultiHeadAttention(
            CausalMHAConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                attention_dropout=config.attention_dropout,
                residual_dropout=config.residual_dropout,
                use_bias=config.use_attention_bias,
                use_rope=config.use_rope,
                rope_theta=config.rope_theta,
                rotary_dim=config.rotary_dim,
                max_seq_len=config.max_seq_len,
                init_std=config.init_std,
            )
        )

    if attention_type == "hca":
        hca_kwargs = dict(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            compression_factor=config.hca_compression_factor,
            window_size=config.window_size,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            use_bias=config.use_attention_bias,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
            rotary_dim=config.rotary_dim,
            max_seq_len=config.max_seq_len,
            init_std=config.init_std,
        )

        # Canonical HCA extras. Add only when the installed HCAConfig supports them.
        optional_hca_kwargs = {
            "use_attention_sink": config.use_attention_sink,
            "use_grouped_output_projection": config.use_grouped_output_projection,
            "output_projection_groups": config.output_projection_groups,
        }

        try:
            hca_sig = inspect.signature(HCAConfig)
            for key, value in optional_hca_kwargs.items():
                if key in hca_sig.parameters:
                    hca_kwargs[key] = value
        except Exception:
            # If signature introspection fails, fall back to the base kwargs.
            pass

        return HCAAttention(HCAConfig(**hca_kwargs))

    if attention_type == "csa":
        return CSAAttention(
            CSAConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                compression_factor=config.compression_factor,
                top_k=config.top_k_blocks,
                window_size=config.window_size,
                indexer_dim=config.indexer_dim,
                n_indexer_heads=config.n_indexer_heads,
                query_compression_dim=config.query_compression_dim,
                attention_dropout=config.attention_dropout,
                residual_dropout=config.residual_dropout,
                use_bias=config.use_attention_bias,
                use_rope=config.use_rope,
                rope_theta=config.rope_theta,
                rotary_dim=config.rotary_dim,
                max_seq_len=config.max_seq_len,
                init_std=config.init_std,

                # Canonical CSA extras. These fields exist in your uploaded canonical CSA.
                use_attention_sink=config.use_attention_sink,
                use_grouped_output_projection=config.use_grouped_output_projection,
                output_projection_groups=config.output_projection_groups,
                use_indexer_score_bias=config.use_indexer_score_bias,
                use_separate_local_kv=config.use_separate_local_kv,
            )
        )

    raise RuntimeError(f"Unknown attention_type={attention_type}")


def get_layer_attention_type(config: DeepSeekV4LMConfig, layer_idx: int) -> str:
    """
    Resolve the concrete attention type for a given layer.

    If config.attention_type is:
        - "mha", "hca", or "csa": all layers use that attention.
        - "hybrid": layer attention is selected from attention_pattern cyclically.

    Example:
        attention_pattern=("csa", "hca")

        layer 0 -> csa
        layer 1 -> hca
        layer 2 -> csa
        layer 3 -> hca
    """

    if config.attention_type != "hybrid":
        return config.attention_type

    return config.attention_pattern[layer_idx % len(config.attention_pattern)]

def build_deepseek_ffn(config: DeepSeekV4LMConfig) -> nn.Module:
    if config.ffn_type == "dense":
        return SwiGLUMLP(
            SwiGLUMLPConfig(
                d_model=config.d_model,
                hidden_dim=config.mlp_hidden_dim,
                expansion_factor=config.mlp_expansion_factor,
                multiple_of=config.mlp_multiple_of,
                dropout=config.mlp_dropout,
                use_bias=config.use_mlp_bias,
                init_std=config.init_std,
            )
        )

    if config.ffn_type == "moe":
        return DeepSeekMoE(
            DeepSeekMoEConfig(
                d_model=config.d_model,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                expert_hidden_dim=config.expert_hidden_dim,
                expert_expansion_factor=config.expert_expansion_factor,
                expert_multiple_of=config.expert_multiple_of,
                shared_experts=config.shared_experts,
                shared_hidden_dim=config.shared_hidden_dim,
                shared_expansion_factor=config.shared_expansion_factor,

                router_type=config.router_type,
                router_score_fn=config.router_score_fn,
                normalize_topk_weights=config.normalize_topk_weights,
                topk_weight_scale=config.topk_weight_scale,
                router_jitter_noise=config.router_jitter_noise,
                hash_routing_stride=config.hash_routing_stride,

                routed_scale=config.routed_scale,
                shared_scale=config.shared_scale,

                dropout=config.mlp_dropout,
                use_bias=config.use_mlp_bias,
                init_std=config.init_std,

                balance_loss_weight=config.balance_loss_weight,
                sequence_balance_loss_weight=config.sequence_balance_loss_weight,
            )
        )

    raise RuntimeError(f"Unknown ffn_type={config.ffn_type}")


def build_deepseek_mhc(config: DeepSeekV4LMConfig) -> "ManifoldHyperConnection":
    return ManifoldHyperConnection(
        ManifoldHyperConnectionConfig(
            d_model=config.d_model,
            n_hc=config.n_hc,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
            eps=config.mhc_eps,
            use_log_sinkhorn=config.mhc_use_log_sinkhorn,
            sinkhorn_fp32=config.mhc_sinkhorn_fp32,
            dynamic=config.mhc_dynamic,
            init_alpha=config.mhc_init_alpha,
            alpha_max=config.mhc_alpha_max,
            bounded_alpha=config.mhc_bounded_alpha,
            init_std=config.init_std,
        )
    )
