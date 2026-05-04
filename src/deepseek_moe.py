# ============================================================
# Mini DeepSeek-V4 DeepSeekMoE-style FFN
# More canonical PyTorch implementation
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_modules.SwiGLU import * 
# ============================================================
# CONFIG
# ============================================================

@dataclass
class DeepSeekMoEConfig:
    d_model: int

    # Routed experts
    num_experts: int = 8
    top_k: int = 2

    expert_hidden_dim: Optional[int] = None
    expert_expansion_factor: float = 4.0
    expert_multiple_of: int = 1

    # Shared experts: now represented as a real ModuleList, not only as a width multiplier.
    shared_experts: int = 1
    shared_hidden_dim: Optional[int] = None
    shared_expansion_factor: float = 4.0

    # Routing
    # "learned": learned router W_r x -> scores -> top-k.
    # "hash": deterministic hash routing from input_ids; useful for early-layer hash MoE experiments.
    router_type: str = "learned"
    router_score_fn: str = "sqrt_softplus"
    normalize_topk_weights: bool = True
    topk_weight_scale: float = 1.0
    router_jitter_noise: float = 0.0
    hash_routing_stride: int = 1

    # Branch scaling
    routed_scale: float = 1.0
    shared_scale: float = 1.0

    # Regularization / init
    dropout: float = 0.0
    use_bias: bool = False
    init_std: float = 0.02

    # Optional losses / stats
    # This is not DeepSeek's full auxiliary-loss-free routing system; it is a small,
    # transparent mini-project balance objective and diagnostic.
    balance_loss_weight: float = 0.0
    sequence_balance_loss_weight: float = 0.0

    eps: float = 1e-9

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}")

        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")

        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k must be <= num_experts, got top_k={self.top_k}, "
                f"num_experts={self.num_experts}"
            )

        if self.expert_hidden_dim is not None and self.expert_hidden_dim <= 0:
            raise ValueError(
                f"expert_hidden_dim must be > 0 when provided, got {self.expert_hidden_dim}"
            )

        if self.expert_expansion_factor <= 0:
            raise ValueError(
                f"expert_expansion_factor must be > 0, got {self.expert_expansion_factor}"
            )

        if self.expert_multiple_of <= 0:
            raise ValueError(
                f"expert_multiple_of must be > 0, got {self.expert_multiple_of}"
            )

        if self.shared_experts < 0:
            raise ValueError(f"shared_experts must be >= 0, got {self.shared_experts}")

        if self.shared_hidden_dim is not None and self.shared_hidden_dim <= 0:
            raise ValueError(
                f"shared_hidden_dim must be > 0 when provided, got {self.shared_hidden_dim}"
            )

        if self.shared_expansion_factor <= 0:
            raise ValueError(
                f"shared_expansion_factor must be > 0, got {self.shared_expansion_factor}"
            )

        allowed_router_types = {"learned", "hash"}
        if self.router_type not in allowed_router_types:
            raise ValueError(
                f"router_type must be one of {allowed_router_types}, got {self.router_type}"
            )

        allowed_score_fns = {"softmax", "sigmoid", "sqrt_softplus"}
        if self.router_score_fn not in allowed_score_fns:
            raise ValueError(
                f"router_score_fn must be one of {allowed_score_fns}, got {self.router_score_fn}"
            )

        if self.topk_weight_scale <= 0:
            raise ValueError(
                f"topk_weight_scale must be > 0, got {self.topk_weight_scale}"
            )

        if self.router_jitter_noise < 0:
            raise ValueError(
                f"router_jitter_noise must be >= 0, got {self.router_jitter_noise}"
            )

        if self.hash_routing_stride <= 0:
            raise ValueError(
                f"hash_routing_stride must be > 0, got {self.hash_routing_stride}"
            )

        if self.routed_scale < 0:
            raise ValueError(f"routed_scale must be >= 0, got {self.routed_scale}")

        if self.shared_scale < 0:
            raise ValueError(f"shared_scale must be >= 0, got {self.shared_scale}")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {self.dropout}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.balance_loss_weight < 0:
            raise ValueError(
                f"balance_loss_weight must be >= 0, got {self.balance_loss_weight}"
            )

        if self.sequence_balance_loss_weight < 0:
            raise ValueError(
                "sequence_balance_loss_weight must be >= 0, "
                f"got {self.sequence_balance_loss_weight}"
            )

        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def to_expert_config(self) -> SwiGLUMLPConfig:
        return SwiGLUMLPConfig(
            d_model=self.d_model,
            hidden_dim=self.expert_hidden_dim,
            expansion_factor=self.expert_expansion_factor,
            multiple_of=self.expert_multiple_of,
            dropout=self.dropout,
            use_bias=self.use_bias,
            init_std=self.init_std,
        )

    def to_shared_expert_config(self) -> SwiGLUMLPConfig:
        return SwiGLUMLPConfig(
            d_model=self.d_model,
            hidden_dim=self.shared_hidden_dim,
            expansion_factor=self.shared_expansion_factor,
            multiple_of=self.expert_multiple_of,
            dropout=self.dropout,
            use_bias=self.use_bias,
            init_std=self.init_std,
        )
    
# ============================================================
# DEEPSEEK MoE
# ============================================================

class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE-style FFN mini implementation.

    Core canonical pieces:
      - learned top-k router with sqrt(softplus(.)) affinity by default
      - optional deterministic hash routing
      - routed SwiGLU experts
      - real shared SwiGLU experts as a ModuleList
      - routed/shared branch scaling
      - top-k weight normalization and scaling
      - global and sequence-wise balance diagnostics/losses

    This intentionally does not implement industrial MoE infrastructure:
      - expert parallelism
      - fused dispatch kernels
      - all-to-all communication
      - FP4/FP8 kernels
      - auxiliary-loss-free routing internals
    """

    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router_type = config.router_type
        self.router_score_fn = config.router_score_fn
        self.normalize_topk_weights = config.normalize_topk_weights
        self.topk_weight_scale = config.topk_weight_scale
        self.router_jitter_noise = config.router_jitter_noise
        self.hash_routing_stride = config.hash_routing_stride
        self.routed_scale = config.routed_scale
        self.shared_scale = config.shared_scale
        self.balance_loss_weight = config.balance_loss_weight
        self.sequence_balance_loss_weight = config.sequence_balance_loss_weight
        self.eps = config.eps

        # The learned router exists even if router_type="hash" so the module keeps
        # a stable interface and can be switched later without reconstructing the class.
        self.router = nn.Linear(
            config.d_model,
            config.num_experts,
            bias=config.use_bias,
        )

        expert_config = config.to_expert_config()
        self.experts = nn.ModuleList(
            [SwiGLUMLP(expert_config) for _ in range(config.num_experts)]
        )

        shared_config = config.to_shared_expert_config()
        self.shared_experts = nn.ModuleList(
            [SwiGLUMLP(shared_config) for _ in range(config.shared_experts)]
        )

        self.reset_router_parameters()

    @property
    def has_shared_experts(self) -> bool:
        return len(self.shared_experts) > 0

    def reset_router_parameters(self) -> None:
        nn.init.normal_(self.router.weight, mean=0.0, std=self.config.init_std)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def _validate_x(self, x: torch.Tensor) -> Tuple[int, int]:
        if x.dim() != 3:
            raise ValueError(
                f"DeepSeekMoE expects x with shape [B,T,d_model], got {tuple(x.shape)}"
            )

        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected x.shape[-1] == d_model={self.d_model}, got {D}")

        return B, T

    def _validate_input_ids(self, input_ids: torch.Tensor, B: int, T: int) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape [B,T] for hash routing, got {tuple(input_ids.shape)}"
            )
        if input_ids.shape != (B, T):
            raise ValueError(
                f"input_ids must have shape {(B, T)} for hash routing, got {tuple(input_ids.shape)}"
            )
        return input_ids.long()

    def _compute_router_logits(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)

        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + self.router_jitter_noise * torch.randn_like(router_logits)

        return router_logits

    def _router_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.router_score_fn == "softmax":
            return F.softmax(router_logits, dim=-1)

        if self.router_score_fn == "sigmoid":
            return torch.sigmoid(router_logits)

        if self.router_score_fn == "sqrt_softplus":
            return torch.sqrt(F.softplus(router_logits) + self.eps)

        raise RuntimeError(f"Unknown router_score_fn={self.router_score_fn}")

    def _topk_routing(
        self,
        router_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_scores, topk_indices = torch.topk(
            router_scores,
            k=self.top_k,
            dim=-1,
        )

        if self.normalize_topk_weights:
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            topk_weights = topk_scores / denom
        else:
            topk_weights = topk_scores

        topk_weights = self.topk_weight_scale * topk_weights

        return topk_scores, topk_indices, topk_weights

    def _hash_routing(
        self,
        input_ids: torch.Tensor,
        B: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic hash routing for early-layer experiments.

        For every route r in 0..top_k-1:
            expert = (input_id + r * hash_routing_stride) % num_experts

        This is intentionally simple and deterministic. It is not meant to model
        learned affinity; it is a mini implementation of token-id based routing.
        """
        route_offsets = torch.arange(self.top_k, device=device, dtype=input_ids.dtype)
        route_offsets = route_offsets * self.hash_routing_stride

        topk_indices = (input_ids[..., None] + route_offsets[None, None, :]) % self.num_experts
        topk_indices = topk_indices.long()

        topk_scores = torch.ones(B, T, self.top_k, device=device, dtype=dtype)

        if self.normalize_topk_weights:
            topk_weights = topk_scores / float(self.top_k)
        else:
            topk_weights = topk_scores

        topk_weights = self.topk_weight_scale * topk_weights

        # Dense diagnostic tensors shaped like learned routing outputs.
        router_logits = torch.zeros(B, T, self.num_experts, device=device, dtype=dtype)
        router_scores = torch.zeros(B, T, self.num_experts, device=device, dtype=dtype)
        router_scores.scatter_add_(
            dim=-1,
            index=topk_indices,
            src=topk_weights.to(dtype=dtype),
        )

        return router_logits, router_scores, topk_scores, topk_indices, topk_weights

    def _compute_aux_stats(
        self,
        router_logits: torch.Tensor,
        router_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = topk_indices.device
        dtype = router_scores.dtype

        B, T, K = topk_indices.shape
        total_routes = B * T * K

        flat_indices = topk_indices.reshape(-1)

        expert_counts = torch.bincount(
            flat_indices,
            minlength=self.num_experts,
        ).to(device=device, dtype=dtype)

        expert_fraction = expert_counts / max(total_routes, 1)

        target = torch.full(
            (self.num_experts,),
            fill_value=1.0 / self.num_experts,
            device=device,
            dtype=dtype,
        )

        raw_balance_loss = ((expert_fraction - target) ** 2).mean()
        balance_loss = self.balance_loss_weight * raw_balance_loss

        # Sequence-wise load statistics: [B,E].
        seq_one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).to(dtype=dtype)
        # [B,T,K,E] -> [B,E]
        sequence_expert_counts = seq_one_hot.sum(dim=(1, 2))
        sequence_expert_fraction = sequence_expert_counts / max(T * K, 1)

        sequence_raw_balance_loss = ((sequence_expert_fraction - target[None, :]) ** 2).mean()
        sequence_balance_loss = self.sequence_balance_loss_weight * sequence_raw_balance_loss

        total_balance_loss = balance_loss + sequence_balance_loss

        # Router entropy is mainly meaningful for learned routing. For hash routing,
        # router_scores is sparse and deterministic; the entropy still serves as a diagnostic.
        router_probs_for_entropy = router_scores.float()
        prob_denom = router_probs_for_entropy.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        router_probs_for_entropy = router_probs_for_entropy / prob_denom

        router_entropy = -(
            router_probs_for_entropy
            * torch.log(router_probs_for_entropy + self.eps)
        ).sum(dim=-1).mean().to(dtype=dtype)

        aux = {
            "router_logits": router_logits,
            "router_scores": router_scores,
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "expert_counts": expert_counts,
            "expert_fraction": expert_fraction,
            "sequence_expert_counts": sequence_expert_counts,
            "sequence_expert_fraction": sequence_expert_fraction,
            "router_entropy": router_entropy,
            "raw_balance_loss": raw_balance_loss,
            "balance_loss": balance_loss,
            "sequence_raw_balance_loss": sequence_raw_balance_loss,
            "sequence_balance_loss": sequence_balance_loss,
            "total_balance_loss": total_balance_loss,
        }

        return aux

    def _dispatch_naive(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape
        K = topk_indices.shape[-1]

        x_flat = x.reshape(B * T, D)
        topk_indices_flat = topk_indices.reshape(B * T, K)
        topk_weights_flat = topk_weights.reshape(B * T, K)

        out_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            selected = topk_indices_flat == expert_id
            token_idx, route_idx = selected.nonzero(as_tuple=True)

            if token_idx.numel() == 0:
                continue

            x_e = x_flat[token_idx]
            y_e = expert(x_e[:, None, :])[:, 0, :]

            w_e = topk_weights_flat[token_idx, route_idx][:, None].to(dtype=y_e.dtype)

            source = y_e * w_e
            source = source.to(dtype=out_flat.dtype)

            out_flat.index_add_(
                dim=0,
                index=token_idx,
                source=source,
            )
        return out_flat.view(B, T, D)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_shared_experts:
            return torch.zeros_like(x)

        shared_out = torch.zeros_like(x)
        for shared_expert in self.shared_experts:
            shared_out = shared_out + shared_expert(x)

        return shared_out

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        B, T = self._validate_x(x)

        if self.router_type == "learned":
            router_logits = self._compute_router_logits(x)
            router_scores = self._router_scores(router_logits)
            topk_scores, topk_indices, topk_weights = self._topk_routing(router_scores)

        elif self.router_type == "hash":
            if input_ids is None:
                raise ValueError("input_ids must be provided when router_type='hash'")
            input_ids = self._validate_input_ids(input_ids, B=B, T=T).to(device=x.device)
            router_logits, router_scores, topk_scores, topk_indices, topk_weights = self._hash_routing(
                input_ids=input_ids,
                B=B,
                T=T,
                device=x.device,
                dtype=x.dtype,
            )

        else:
            raise RuntimeError(f"Unknown router_type={self.router_type}")

        routed_out = self._dispatch_naive(
            x=x,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )

        shared_out = self._shared_forward(x)

        out = self.routed_scale * routed_out + self.shared_scale * shared_out

        if return_aux:
            aux = self._compute_aux_stats(
                router_logits=router_logits,
                router_scores=router_scores,
                topk_indices=topk_indices,
                topk_weights=topk_weights,
                topk_scores=topk_scores,
            )
            aux["routed_out"] = routed_out
            aux["shared_out"] = shared_out
            aux["routed_scale"] = torch.tensor(self.routed_scale, device=x.device, dtype=x.dtype)
            aux["shared_scale"] = torch.tensor(self.shared_scale, device=x.device, dtype=x.dtype)
            aux["router_type"] = self.router_type
            return out, aux

        return out
    

