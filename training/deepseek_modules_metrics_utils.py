# ============================================================
# DEEPSEEK-V4 MODULE DIAGNOSTIC METRICS
# Loss components / MoE / MTP / mHC
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Generic helpers
# ============================================================

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def is_scalar_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.numel() == 1


def to_float(x: Any) -> Optional[float]:
    """
    Convert scalar-like objects to Python float.
    Returns None if conversion is not safe.
    """
    if x is None:
        return None

    if torch.is_tensor(x):
        if x.numel() != 1:
            return None
        value = float(x.detach().float().cpu().item())
        return value if math.isfinite(value) else None

    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def tensor_float(x: Any) -> Optional[torch.Tensor]:
    """
    Return detached fp32 tensor on CPU for diagnostics.
    """
    if not torch.is_tensor(x):
        return None

    if x.numel() == 0:
        return None

    return x.detach().float().cpu()


def get_from_output(outputs: Any, key: str, default: Any = None) -> Any:
    if isinstance(outputs, dict):
        return outputs.get(key, default)
    return getattr(outputs, key, default)


def iter_nested_dicts(obj: Any):
    """
    Recursively yield all dictionaries inside outputs/aux structures.
    """
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_nested_dicts(v)

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from iter_nested_dicts(item)


def collect_values_by_key(obj: Any, key: str) -> List[Any]:
    values = []

    for d in iter_nested_dicts(obj):
        if key in d:
            values.append(d[key])

    return values


def collect_values_by_any_key(obj: Any, keys: Iterable[str]) -> List[Any]:
    values = []
    key_set = set(keys)

    for d in iter_nested_dicts(obj):
        for k in key_set:
            if k in d:
                values.append(d[k])

    return values


def mean_of_scalar_values(values: List[Any]) -> Optional[float]:
    floats = []

    for v in values:
        fv = to_float(v)
        if fv is not None:
            floats.append(fv)

    if not floats:
        return None

    return float(sum(floats) / len(floats))


def cat_flat_tensors(values: List[Any]) -> Optional[torch.Tensor]:
    tensors = []

    for v in values:
        t = tensor_float(v)
        if t is not None:
            tensors.append(t.reshape(-1))

    if not tensors:
        return None

    return torch.cat(tensors, dim=0)


def safe_stat_tensor(t: torch.Tensor, prefix: str) -> Dict[str, float]:
    """
    Basic stats for a tensor.
    """
    t = t.detach().float().cpu()

    if t.numel() == 0:
        return {}

    return {
        f"{prefix}_mean": float(t.mean().item()),
        f"{prefix}_std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        f"{prefix}_min": float(t.min().item()),
        f"{prefix}_max": float(t.max().item()),
    }


def maybe_add_metric(metrics: Dict[str, float], key: str, value: Any) -> None:
    fv = to_float(value)
    if fv is not None:
        metrics[key] = fv

