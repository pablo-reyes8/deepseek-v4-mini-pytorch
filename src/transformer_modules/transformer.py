# ============================================================
# Mini Causal LM Baseline
# TokenEmbedding + TransformerBlock x L + Final RMSNorm + LM Head
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_modules.transformer_block import *
from data.data_utils import * 

# ============================================================
# CONFIG
# ============================================================

@dataclass
class MiniCausalLMConfig:
    vocab_size: int
    d_model: int
    n_layers: int

    pad_token_id: Optional[int] = None
    max_seq_len: int = 1024

    embedding_dropout: float = 0.0
    scale_embeddings: bool = False
    tie_word_embeddings: bool = True

    rms_norm_eps: float = 1e-6

    n_heads: int = 4
    head_dim: Optional[int] = None
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    use_attention_bias: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None

    mlp_hidden_dim: Optional[int] = None
    mlp_expansion_factor: float = 4.0
    mlp_multiple_of: int = 1
    mlp_dropout: float = 0.0
    use_mlp_bias: bool = False

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
                    "pad_token_id must satisfy 0 <= pad_token_id < vocab_size, "
                    f"got pad_token_id={self.pad_token_id}, "
                    f"vocab_size={self.vocab_size}"
                )

        if self.rms_norm_eps <= 0:
            raise ValueError(
                f"rms_norm_eps must be > 0, got {self.rms_norm_eps}"
            )

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        # Validate child configs too.
        self.to_embedding_config().validate()
        self.to_block_config().validate()

    def to_embedding_config(self) -> "EmbeddingConfig":
        return EmbeddingConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            pad_token_id=self.pad_token_id,
            max_seq_len=self.max_seq_len,
            embedding_dropout=self.embedding_dropout,
            scale_embeddings=self.scale_embeddings,
            init_std=self.init_std,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def to_block_config(self) -> "TransformerBlockConfig":
        return TransformerBlockConfig(
            d_model=self.d_model,
            rms_norm_eps=self.rms_norm_eps,

            n_heads=self.n_heads,
            head_dim=self.head_dim,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            use_attention_bias=self.use_attention_bias,
            use_rope=self.use_rope,
            rope_theta=self.rope_theta,
            rotary_dim=self.rotary_dim,
            max_seq_len=self.max_seq_len,

            mlp_hidden_dim=self.mlp_hidden_dim,
            mlp_expansion_factor=self.mlp_expansion_factor,
            mlp_multiple_of=self.mlp_multiple_of,
            mlp_dropout=self.mlp_dropout,
            use_mlp_bias=self.use_mlp_bias,

            init_std=self.init_std,
        )


# ============================================================
# MINI CAUSAL LM
# ============================================================

class MiniCausalLM(nn.Module):
    """
    Minimal dense causal language model baseline.

    Components:
        TokenEmbedding
        TransformerBlock x n_layers
        Final RMSNorm
        LM Head

    Input:
        input_ids: [B, T]
        labels: optional [B, T]
        attention_mask: optional [B, T]
        position_ids: optional [T] or [B, T]

    Output dict:
        {
            "logits": logits,  # [B, T, vocab_size]
            "loss": loss,      # scalar or None
            "aux": aux,        # dict
        }

    This module intentionally does NOT implement:
        - generate()
        - KV cache
        - MTP
        - MoE
        - HCA
        - CSA
        - mHC
        - optimizer/training loop
    """

    def __init__(self, config: MiniCausalLMConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.pad_token_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len

        self.embedding = TokenEmbedding(config.to_embedding_config())

        block_config = config.to_block_config()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(block_config)
                for _ in range(config.n_layers)
            ])

        self.final_norm = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False)

        self.reset_parameters()

        if config.tie_word_embeddings:
            self.tie_weights()

    def reset_parameters(self) -> None:
        """
        Initialize LM head if it is not tied.

        Note:
            TokenEmbedding, TransformerBlock, attention and MLP modules
            already initialize their own parameters.
        """
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=self.config.init_std)

    def tie_weights(self) -> None:
        """
        Tie LM head weight to token embedding weight.

        This must share the same Parameter object, not copy values.
        """
        self.lm_head.weight = self.embedding.weight


    def _validate_input_ids(self, input_ids: torch.Tensor) -> tuple[int, int]:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}"
            )

        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        return B, T


    def _build_or_validate_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

        if attention_mask is None:
            if self.pad_token_id is None:
                return None

            return (input_ids != self.pad_token_id).long()

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids. "
                f"Expected {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}")

        return attention_mask

    def _validate_labels(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor) -> None:

        if labels.shape != input_ids.shape:
            raise ValueError(
                "labels must have the same shape as input_ids. "
                f"Expected {tuple(input_ids.shape)}, got {tuple(labels.shape)}")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        return_aux: bool = False,
        need_weights: bool = False,) -> Dict[str, Any]:

        """
        Args:
            input_ids:
                Token ids [B, T].

            labels:
                Optional next-token labels [B, T].
                The dataset is expected to already shift labels.
                This model does NOT shift internally.

            attention_mask:
                Optional [B, T].
                1 = valid token.
                0 = pad token.
                If None and pad_token_id exists, it is built automatically.

            position_ids:
                Optional RoPE positions: None, [T], or [B, T].

            start_pos:
                RoPE offset if position_ids is None.

            return_aux:
                Whether to return aux dictionary.
                Current aux is empty unless need_weights=True.

            need_weights:
                Whether to collect attention weights from all layers.

        Returns:
            {
                "logits": [B, T, vocab_size],
                "loss": scalar or None,
                "aux": dict,
            }
        """
         # ----------------------------------------------------
        # Accept full batch dict as first argument
        # ----------------------------------------------------
        if isinstance(input_ids, dict):
            batch = normalize_lm_batch(input_ids)

            input_ids = batch["input_ids"]
            labels = batch.get("labels", labels)
            attention_mask = batch.get("attention_mask", attention_mask)
            position_ids = batch.get("position_ids", position_ids)

        elif isinstance(input_ids, (tuple, list)) or torch.is_tensor(input_ids):
            # Optional: only normalize tuples/tensors if this path is useful to you.
            pass

        B, T = self._validate_input_ids(input_ids)

        if labels is not None:
            self._validate_labels(labels, input_ids)

        attention_mask = self._build_or_validate_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,)

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------
        x = self.embedding(input_ids)
        # [B, T, d_model]

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------
        attn_weights_all = []

        for block in self.blocks:
            if need_weights:
                x, block_aux = block(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    start_pos=start_pos,
                    need_weights=True)

                attn_weights_all.append(block_aux["attn_weights"])
            else:
                x = block(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    start_pos=start_pos,
                    need_weights=False)

        # ----------------------------------------------------
        # Final norm + LM head
        # ----------------------------------------------------
        x = self.final_norm(x)
        logits = self.lm_head(x)
        # [B, T, vocab_size]

        # ----------------------------------------------------
        # Optional loss
        # ----------------------------------------------------
        loss = None

        if labels is not None:
            ignore_index = (
                self.pad_token_id
                if self.pad_token_id is not None
                else -100)

            loss = F.cross_entropy(
                logits.reshape(B * T, self.vocab_size),
                labels.reshape(B * T),
                ignore_index=ignore_index,)

        # ----------------------------------------------------
        # Aux
        # ----------------------------------------------------
        aux = {}

        if need_weights:
            aux["attn_weights"] = attn_weights_all

        # If return_aux=False and need_weights=False, aux remains empty.
        # We still return it for output consistency.
        return {
            "logits": logits,
            "loss": loss,
            "aux": aux if (return_aux or need_weights) else {},}