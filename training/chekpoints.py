# ============================================================
# CHECKPOINT UTILITIES
# DeepSeek-V4 Mini Training Stack
# ============================================================

from __future__ import annotations

import os
import re
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Small helpers
# ============================================================

def _safe_to_serializable(obj: Any) -> Any:
    """
    Best-effort conversion for configs/metadata into JSON-safe objects.

    Useful for saving config snapshots next to the .pt checkpoint.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_safe_to_serializable(x) for x in obj]

    if isinstance(obj, dict):
        return {
            str(k): _safe_to_serializable(v)
            for k, v in obj.items()
        }

    if hasattr(obj, "__dict__"):
        return {
            str(k): _safe_to_serializable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return str(obj)


def _get_rng_state() -> Dict[str, Any]:
    """
    Capture random states so training can be resumed more faithfully.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def _set_rng_state(state):
    """
    Restore Python, NumPy, PyTorch CPU and CUDA RNG states safely.

    Important:
        torch.set_rng_state expects a CPU ByteTensor.
        torch.cuda.set_rng_state_all expects CUDA-compatible RNG states,
        but CPU ByteTensors are also accepted in most PyTorch versions.
    """
    if not state:
        return

    # Python RNG
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])

    # NumPy RNG
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])

    # Torch CPU RNG
    if "torch" in state and state["torch"] is not None:
        torch_state = state["torch"]

        if not torch.is_tensor(torch_state):
            torch_state = torch.tensor(torch_state, dtype=torch.uint8)

        torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
        torch.set_rng_state(torch_state)

    # CUDA RNG
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = state["cuda"]

        fixed_cuda_states = []
        for s in cuda_states:
            if not torch.is_tensor(s):
                s = torch.tensor(s, dtype=torch.uint8)

            # Keep as CPU ByteTensor; PyTorch usually accepts this.
            # If your version requires CUDA tensors, change `.cpu()` to `.cuda(i)`.
            s = s.detach().cpu().to(dtype=torch.uint8)
            fixed_cuda_states.append(s)

        torch.cuda.set_rng_state_all(fixed_cuda_states)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """
    Handles DataParallel/DDP-style wrappers.
    """
    return model.module if hasattr(model, "module") else model


def _atomic_torch_save(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Atomic save:
        1. save to temporary file
        2. rename to final path

    This avoids corrupting the last checkpoint if the process crashes mid-save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _extract_step_from_name(path: Union[str, Path]) -> int:
    """
    Extract step number from filenames like:
        checkpoint_step_000001.pt
        ckpt_step_100.pt
    """
    name = Path(path).name
    matches = re.findall(r"step[_-](\d+)", name)

    if not matches:
        return -1

    return int(matches[-1])


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
) -> None:
    """
    Keep only the last N step checkpoints.

    Does not delete:
        latest.pt
        checkpoint_best_*.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if keep_last_n <= 0:
        return

    candidates = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    if len(candidates) <= keep_last_n:
        return

    candidates = sorted(
        candidates,
        key=lambda p: (_extract_step_from_name(p), p.stat().st_mtime),
    )

    to_delete = candidates[:-keep_last_n]

    for path in to_delete:
        try:
            path.unlink()
        except Exception:
            pass

        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                sidecar.unlink()
            except Exception:
                pass


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    checkpoint_dir: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: Optional[float] = None,
    config: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    save_rng_state: bool = True,
    keep_last_n: Optional[int] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Save a full training checkpoint.

    Saves:
        - model
        - optimizer, if provided
        - scheduler, if provided
        - scaler, if provided
        - EMA, if provided
        - epoch / step / best_metric
        - config / extra_state
        - RNG state, if save_rng_state=True
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if tag is not None:
            filename = f"checkpoint_{tag}_step_{step:08d}.pt"
        else:
            filename = f"checkpoint_step_{step:08d}.pt"

    ckpt_path = checkpoint_dir / filename

    raw_model = _unwrap_model(model)

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": best_metric,
        "config": _safe_to_serializable(config),
        "extra_state": extra_state or {},
        "rng_state": _get_rng_state() if save_rng_state else None,
        "has_ema": ema is not None,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        else:
            checkpoint["scheduler_state_dict"] = None

    if scaler is not None:
        if hasattr(scaler, "state_dict"):
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        else:
            checkpoint["scaler_state_dict"] = None

    if ema is not None:
        if hasattr(ema, "state_dict"):
            checkpoint["ema_state_dict"] = ema.state_dict()
        else:
            raise TypeError(
                "ema was provided but does not implement state_dict()."
            )

    _atomic_torch_save(checkpoint, ckpt_path)

    # Save a lightweight JSON sidecar for quick inspection.
    metadata_path = ckpt_path.with_suffix(".json")
    metadata = {
        "checkpoint": ckpt_path.name,
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": best_metric,
        "tag": tag,
        "has_optimizer": optimizer is not None,
        "has_scheduler": scheduler is not None,
        "has_scaler": scaler is not None,
        "has_ema": ema is not None,
        "config": _safe_to_serializable(config),
        "extra_state": _safe_to_serializable(extra_state or {}),
    }

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # Update latest pointer.
    latest_path = checkpoint_dir / "latest.pt"
    try:
        shutil.copyfile(ckpt_path, latest_path)
    except Exception:
        pass

    # Optional retention policy.
    if keep_last_n is not None and keep_last_n > 0:
        cleanup_old_checkpoints(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=keep_last_n,
        )

    return ckpt_path


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_scaler: bool = True,
    load_ema: bool = True,
    load_rng_state: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint into model/optimizer/scheduler/scaler/EMA if provided.

    Returns:
        state dict with epoch, step, best_metric, config, extra_state, etc.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if model is not None:
        raw_model = _unwrap_model(model)

        missing, unexpected = raw_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

        checkpoint["missing_keys"] = missing
        checkpoint["unexpected_keys"] = unexpected

    if (
        optimizer is not None
        and load_optimizer
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and load_scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if (
        scaler is not None
        and load_scaler
        and "scaler_state_dict" in checkpoint
        and checkpoint["scaler_state_dict"] is not None
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if (
        ema is not None
        and load_ema
        and "ema_state_dict" in checkpoint
        and checkpoint["ema_state_dict"] is not None
    ):
        # Tu EMA acepta strict=False si usas la versión adaptada.
        # Si usas la EMA original, esta llamada también funciona sin strict.
        try:
            ema.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        except TypeError:
            ema.load_state_dict(checkpoint["ema_state_dict"])

    if load_rng_state:
        _set_rng_state(checkpoint.get("rng_state", None))

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_metric": checkpoint.get("best_metric", None),
        "config": checkpoint.get("config", None),
        "extra_state": checkpoint.get("extra_state", {}),
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": checkpoint.get("missing_keys", []),
        "unexpected_keys": checkpoint.get("unexpected_keys", []),
        "has_ema": bool(checkpoint.get("has_ema", "ema_state_dict" in checkpoint)),
        "loaded_ema": bool(
            ema is not None
            and load_ema
            and "ema_state_dict" in checkpoint
            and checkpoint["ema_state_dict"] is not None
        ),
    }