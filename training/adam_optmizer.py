
from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn


def _is_norm_parameter(name: str, module: Optional[nn.Module] = None) -> bool:
    """
    Conservative norm detector.

    Catches:
        - RMSNorm
        - LayerNorm
        - BatchNorm
        - parameters whose names include norm
    """
    lower_name = name.lower()

    if "norm" in lower_name:
        return True

    if module is not None:
        norm_classes = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LocalResponseNorm,
        )
        if isinstance(module, norm_classes):
            return True

    return False


def _is_embedding_parameter(name: str, module: Optional[nn.Module] = None) -> bool:
    lower_name = name.lower()

    if "embedding" in lower_name or "embed" in lower_name or "wte" in lower_name:
        return True

    if module is not None and isinstance(module, nn.Embedding):
        return True

    return False


def _get_module_by_parameter_name(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Build mapping from full parameter name to the module that owns it.

    Example:
        blocks.0.attn.q_proj.weight -> q_proj module
    """
    param_to_module: Dict[str, nn.Module] = {}

    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[full_name] = module

    return param_to_module


def build_adamw_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_weight_decay: float = 0.0,
    verbose: bool = False,
) -> Tuple[list, Dict[str, Any]]:
    """
    Build AdamW parameter groups.

    Policy:
        decay:
            Parameters with ndim >= 2 that are not embeddings and not lm_head.

        no_decay:
            biases
            norm weights
            embeddings
            lm_head
            scalar/vector params
            mHC small/static params
            router/scalar gates if they are not matrices

    This is intentionally conservative for DeepSeek-V4 mini because mHC,
    routing scalars, normalization and embeddings should not receive decay.

    Args:
        model:
            DeepSeekV4LM or any nn.Module.
        weight_decay:
            Weight decay for decay group.
        no_decay_weight_decay:
            Usually 0.0.
        verbose:
            If True, returns names and prints group sizes.

    Returns:
        optimizer_groups, info
    """
    decay_params = []
    no_decay_params = []

    decay_names = []
    no_decay_names = []

    param_to_module = _get_module_by_parameter_name(model)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = param_to_module.get(name, None)
        lower_name = name.lower()

        is_bias = name.endswith(".bias")
        is_norm = _is_norm_parameter(name, module)
        is_embedding = _is_embedding_parameter(name, module)
        is_lm_head = "lm_head" in lower_name or "prediction_head" in lower_name
        is_mtp_vocab_head = "mtp" in lower_name and "head" in lower_name
        is_scalar_or_vector = param.ndim < 2

        # mHC has small dynamic/static/gating parameters where decay is risky.
        is_mhc_small_param = (
            "mhc" in lower_name
            or "hyper" in lower_name
            or "sinkhorn" in lower_name
            or "alpha" in lower_name
            or "static_" in lower_name
        )

        # Compression/indexer biases or small controls should be no_decay.
        is_special_small_control = (
            "bias_a" in lower_name
            or "bias_b" in lower_name
            or "temperature" in lower_name
            or "scale" in lower_name
            or "gate" in lower_name and param.ndim < 2
        )

        use_no_decay = (
            is_bias
            or is_norm
            or is_embedding
            or is_lm_head
            or is_mtp_vocab_head
            or is_scalar_or_vector
            or is_mhc_small_param
            or is_special_small_control
        )

        if use_no_decay:
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    optimizer_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "group_name": "decay",
        },
        {
            "params": no_decay_params,
            "weight_decay": no_decay_weight_decay,
            "group_name": "no_decay",
        },
    ]

    info = {
        "num_decay_params": sum(p.numel() for p in decay_params),
        "num_no_decay_params": sum(p.numel() for p in no_decay_params),
        "num_decay_tensors": len(decay_params),
        "num_no_decay_tensors": len(no_decay_params),
        "decay_names": decay_names,
        "no_decay_names": no_decay_names,
    }

    if verbose:
        total = info["num_decay_params"] + info["num_no_decay_params"]
        print("=" * 80)
        print("AdamW parameter groups")
        print("=" * 80)
        print(f"Decay tensors     : {info['num_decay_tensors']}")
        print(f"No-decay tensors  : {info['num_no_decay_tensors']}")
        print(f"Decay params      : {info['num_decay_params']:,}")
        print(f"No-decay params   : {info['num_no_decay_params']:,}")
        print(f"Total params      : {total:,}")
        print("=" * 80)

    return optimizer_groups, info


def build_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    verbose: bool = False,
) -> Tuple[torch.optim.AdamW, Dict[str, Any]]:
    """
    Build AdamW optimizer with DeepSeek-V4-mini-safe parameter grouping.

    Args:
        model:
            Model to optimize.
        learning_rate:
            AdamW learning rate.
        weight_decay:
            Weight decay for matrix-like trainable weights.
        betas:
            AdamW betas.
        eps:
            AdamW epsilon.
        verbose:
            Whether to print parameter group summary.

    Returns:
        optimizer, group_info
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

    if len(betas) != 2:
        raise ValueError(f"betas must have length 2, got {betas}")

    parameter_groups, group_info = build_adamw_parameter_groups(
        model=model,
        weight_decay=weight_decay,
        no_decay_weight_decay=0.0,
        verbose=verbose,
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer, group_info