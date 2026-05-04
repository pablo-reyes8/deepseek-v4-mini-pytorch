# ============================================================
# Forward helpers
# ============================================================


from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from training.full_deepseek_metrics import * 
from training.autocast import *  
from data.data_utils import * 
from training.training_metrics import * 


def filter_forward_kwargs(model: nn.Module, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs accepted by model.forward.

    This makes the trainer robust to:
        - DeepSeekV4LM
        - simpler LM wrappers
    """
    raw_model = model.module if hasattr(model, "module") else model
    sig = inspect.signature(raw_model.forward)
    allowed = set(sig.parameters.keys())

    return {
        k: v
        for k, v in kwargs.items()
        if k in allowed and v is not None
    }


def call_model_for_training(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    return_aux: bool = False,
    need_weights: bool = False,
):
    """
    Safe model call for training.

    For DeepSeekV4LM, supports:
        input_ids, labels, mtp_labels, attention_mask,
        position_ids, start_pos, return_aux, need_weights.

    For simpler models, unsupported kwargs are dropped.
    """
    kwargs = {
        "input_ids": batch["input_ids"],
        "labels": batch.get("labels", None),
        "mtp_labels": batch.get("mtp_labels", None),
        "attention_mask": batch.get("attention_mask", None),
        "position_ids": batch.get("position_ids", None),
        "start_pos": batch.get("start_pos", None),
        "return_aux": return_aux,
        "need_weights": need_weights,
    }

    kwargs = filter_forward_kwargs(model, kwargs)
    return model(**kwargs)


def get_loss_from_outputs(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "loss" not in outputs:
            raise KeyError("Model outputs dict does not contain key 'loss'.")
        return outputs["loss"]

    if hasattr(outputs, "loss"):
        return outputs.loss

    raise KeyError("Could not extract loss from model outputs.")


# ============================================================
# Optimizer / scaler helpers
# ============================================================

def is_hybrid_muon_adamw_optimizer(optimizer: Any) -> bool:
    return hasattr(optimizer, "muon") and hasattr(optimizer, "adamw")


def unscale_optimizer_grads(scaler: Any, optimizer: Any) -> None:
    """
    GradScaler.unscale_ for AdamW or HybridMuonAdamW.
    """
    if scaler is None:
        return

    if is_hybrid_muon_adamw_optimizer(optimizer):
        scaler.unscale_(optimizer.muon)
        scaler.unscale_(optimizer.adamw)
    else:
        scaler.unscale_(optimizer)


def scaler_step_optimizer(scaler: Any, optimizer: Any) -> None:
    """
    scaler.step for AdamW or HybridMuonAdamW.
    """
    if is_hybrid_muon_adamw_optimizer(optimizer):
        scaler.step(optimizer.muon)
        scaler.step(optimizer.adamw)
    else:
        scaler.step(optimizer)


def get_current_lrs(optimizer: Any, scheduler: Optional[Any] = None) -> Dict[str, float]:
    """
    Logging-friendly LR extraction.
    """
    if scheduler is not None and hasattr(scheduler, "get_lr_dict"):
        lr_dict = scheduler.get_lr_dict()

        out = {}

        for k, v in lr_dict.items():
            if isinstance(v, (int, float)) and v is not None:
                out[f"lr/{k}"] = float(v)

        return out

    if is_hybrid_muon_adamw_optimizer(optimizer):
        out = {}

        if optimizer.muon.param_groups:
            out["lr/muon_lr"] = float(optimizer.muon.param_groups[0]["lr"])

        if optimizer.adamw.param_groups:
            out["lr/adamw_lr"] = float(optimizer.adamw.param_groups[0]["lr"])

        return out

    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return {"lr/lr": float(optimizer.param_groups[0]["lr"])}

    return {}

# ============================================================
# COLAB / DRIVE UTILS
# ============================================================


def rule(w: int = 110, ch: str = "─") -> str:
    return ch * w


def is_colab() -> bool:
    return "google.colab" in sys.modules


def ensure_drive_mounted() -> None:
    if is_colab():
        drive_root = "/content/drive"
        if not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount(drive_root, force_remount=False)
            except Exception as e:
                print(f"[DRIVE] No se pudo montar automáticamente: {e}")


def copy_ckpt_to_drive_fixed(
    src_path: str | Path,
    drive_dir: str | Path,
    fixed_name: str = "latest_deepseekv4.pt",
) -> None:
    try:
        if not drive_dir:
            return

        src_path = Path(src_path)
        drive_dir = Path(drive_dir)

        if str(drive_dir).startswith("/content/drive"):
            ensure_drive_mounted()

        drive_dir.mkdir(parents=True, exist_ok=True)

        dst_path = drive_dir / fixed_name

        if dst_path.exists():
            dst_path.unlink()

        shutil.copy2(src_path, dst_path)
        print(f"└─ [DRIVE] copiado → {dst_path}")

    except Exception as e:
        print(f"└─ [DRIVE] ERROR al copiar a Drive: {e}")


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_record = {}
    for k, v in record.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                safe_record[k] = float(v.detach().cpu().item())
            else:
                safe_record[k] = str(tuple(v.shape))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            safe_record[k] = v
        else:
            try:
                json.dumps(v)
                safe_record[k] = v
            except Exception:
                safe_record[k] = str(v)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def copy_metrics_to_drive(
    src_path: str | Path,
    drive_dir: str | Path,
    fixed_name: str = "metrics_deepseekv4.jsonl",
) -> None:
    try:
        if not drive_dir:
            return

        src_path = Path(src_path)
        drive_dir = Path(drive_dir)

        if not src_path.exists():
            return

        if str(drive_dir).startswith("/content/drive"):
            ensure_drive_mounted()

        drive_dir.mkdir(parents=True, exist_ok=True)

        dst_path = drive_dir / fixed_name
        shutil.copy2(src_path, dst_path)

    except Exception as e:
        print(f"└─ [DRIVE] ERROR al copiar métricas a Drive: {e}")


def gpu_mem_mb(device="cuda") -> Tuple[float, float]:
    device_type = torch.device(device).type

    if torch.cuda.is_available() and device_type == "cuda":
        device_obj = torch.device(device)
        alloc = torch.cuda.memory_allocated(device=device_obj) / (1024 ** 2)
        reserv = torch.cuda.memory_reserved(device=device_obj) / (1024 ** 2)
        return float(alloc), float(reserv)

    return 0.0, 0.0


# ============================================================
# MONITORING HELPERS
# ============================================================

VALID_MONITOR_NAMES = {
    "loss",
    "lm_loss",
    "mtp_loss",
    "moe_aux_loss",
    "perplexity",
    "token_accuracy",
    "top_5_accuracy",
    "sequence_accuracy",
    "mean_confidence",
    "mean_true_token_prob",
    "mean_entropy",
    "train_loss",
    "train_lm_loss",
    "train_mtp_loss",
    "train_moe_aux_loss",
    "eval_loss",
    "eval_perplexity",
    "eval_token_accuracy",
    "eval_top_5_accuracy",
    "eval_sequence_accuracy",
}


def is_better_metric(
    current: float,
    best: Optional[float],
    mode: str = "min",
) -> bool:
    if best is None:
        return True

    if mode == "min":
        return current < best

    if mode == "max":
        return current > best

    raise ValueError(f"monitor_mode must be 'min' or 'max', got {mode}")


def prefixed_stats(prefix: str, stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if stats is None:
        return {}

    return {f"{prefix}_{k}": v for k, v in stats.items()}


def resolve_monitor_value(
    monitor_name: str,
    train_stats: Dict[str, Any],
    eval_stats: Optional[Dict[str, Any]],
) -> float:
    """
    Supports:
        loss
        train_loss
        eval_loss
        eval_token_accuracy
        etc.
    """
    if monitor_name.startswith("train_"):
        key = monitor_name[len("train_"):]
        source = train_stats
    elif monitor_name.startswith("eval_"):
        if eval_stats is None:
            raise ValueError(
                f"monitor_name='{monitor_name}' requiere eval_stats, pero eval_loader=None."
            )
        key = monitor_name[len("eval_"):]
        source = eval_stats
    else:
        if eval_stats is not None and monitor_name in eval_stats:
            key = monitor_name
            source = eval_stats
        else:
            key = monitor_name
            source = train_stats

    if key not in source:
        raise KeyError(
            f"Monitor key '{key}' not found in selected stats. "
            f"Available keys: {sorted(source.keys())[:50]}"
        )

    return float(source[key])




def get_lr_message(optimizer: Any, scheduler: Optional[Any] = None) -> str:
    if scheduler is not None and hasattr(scheduler, "get_lr_dict"):
        d = scheduler.get_lr_dict()

        if "muon_lr" in d and "adamw_lr" in d:
            return f"muon_lr={d['muon_lr']:.2e} adamw_lr={d['adamw_lr']:.2e}"

        if "lr" in d:
            return f"lr={d['lr']:.2e}"

    if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
        muon_lr = optimizer.muon.param_groups[0]["lr"]
        adamw_lr = optimizer.adamw.param_groups[0]["lr"]
        return f"muon_lr={muon_lr:.2e} adamw_lr={adamw_lr:.2e}"

    return f"lr={get_main_lr(optimizer):.2e}"


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().cpu().item())
        return float(x)
    except Exception:
        return default

# ============================================================
# PRETTY PRINT HELPERS FOR DEEPSEEK-V4 TRAINING
# ============================================================


def ds_rule(width: int = 110, ch: str = "─") -> str:
    return ch * width


def ds_title(title: str, width: int = 110, ch: str = "═") -> None:
    print("\n" + ch * width)
    print(title)
    print(ch * width)


def ds_section(title: str, width: int = 110) -> None:
    print("\n" + title)
    print(ds_rule(width=width, ch="─"))


def fmt_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def fmt_num(x, digits: int = 4, sci_low: float = 1e-3, sci_high: float = 1e4) -> str:
    try:
        x = float(x)
    except Exception:
        return "—"

    if not math.isfinite(x):
        return "—"

    ax = abs(x)

    if ax == 0:
        return "0"

    if ax < sci_low or ax >= sci_high:
        return f"{x:.2e}"

    return f"{x:.{digits}f}"


def fmt_lr(x) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return "—"


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().cpu().item())
        return float(x)
    except Exception:
        return default


def get_main_lr(optimizer: Any) -> float:
    if hasattr(optimizer, "adamw"):
        if optimizer.adamw.param_groups:
            return float(optimizer.adamw.param_groups[0]["lr"])

    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return float(optimizer.param_groups[0]["lr"])

    return float("nan")


def get_optimizer_lrs(optimizer: Any) -> Dict[str, float]:
    if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
        return {
            "muon_lr": float(optimizer.muon.param_groups[0]["lr"]) if optimizer.muon.param_groups else float("nan"),
            "adamw_lr": float(optimizer.adamw.param_groups[0]["lr"]) if optimizer.adamw.param_groups else float("nan"),
        }

    return {
        "lr": get_main_lr(optimizer),
    }


def print_deepseek_run_header(
    *,
    run_name: str,
    model: nn.Module,
    device,
    precision: Dict[str, Any],
    optimizer_type: str,
    ema,
    epochs: int,
    start_epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    monitor_name: str,
    monitor_mode: str,
    best_metric,
    grad_clip,
    grad_accum_steps: int,
    val_loader,
    eval_every: int,
    eval_max_batches,
    drive_ckpt_dir,
    fixed_drive_name: str,
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    cfg = getattr(raw_model, "config", None)

    ema_str = (
        f"{ema.decay:.6f}"
        if ema is not None and hasattr(ema, "decay")
        else "off"
    )

    ds_title(f"DeepSeek-V4 run: {run_name}")

    print(
        f"Device    : {device} | AMP: {precision['amp_enabled']} "
        f"({precision['amp_dtype_requested']} -> {precision['amp_dtype_effective']})"
    )
    print(
        f"Optimizer : {optimizer_type} | EMA: {ema_str} | "
        f"grad_clip: {grad_clip} | grad_accum_steps: {grad_accum_steps}"
    )
    print(
        f"Schedule  : epochs={epochs} | start_epoch={start_epoch} | "
        f"global_step={global_step} | total_steps={total_steps} | warmup_steps={warmup_steps}"
    )
    print(
        f"Monitor   : {monitor_name} ({monitor_mode}) | best_metric={best_metric}"
    )

    if cfg is not None:
        print(
            f"Model     : layers={getattr(cfg, 'n_layers', '?')} | "
            f"d_model={getattr(cfg, 'd_model', '?')} | "
            f"attention={getattr(cfg, 'attention_type', '?')} | "
            f"ffn={getattr(cfg, 'ffn_type', '?')} | "
            f"mHC={getattr(cfg, 'use_mhc', '?')} | "
            f"MTP={getattr(cfg, 'use_mtp', '?')}"
        )

    if val_loader is not None:
        print(f"Eval      : enabled | eval_every={eval_every} | eval_max_batches={eval_max_batches}")
    else:
        print("Eval      : disabled")

    if drive_ckpt_dir:
        print(f"Drive     : {drive_ckpt_dir} | fixed checkpoint={fixed_drive_name}")

    print(ds_rule())


def print_in_epoch_header() -> None:
    print("\n┆ In-epoch training")
    print(
        "┆   {:>7} │ {:>7} │ {:>8} │ {:>8} │ {:>8} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │ {:>9}".format(
            "step",
            "batch",
            "loss",
            "lm",
            "mtp",
            "moe_aux",
            "grad",
            "muon_lr",
            "adamw_lr",
            "time",
        )
    )
    print("┆   " + "─" * 106)


def print_in_epoch_row(
    *,
    global_step: int,
    batch_idx: int,
    loss_val: float,
    lm_loss: float,
    mtp_loss: float,
    moe_aux_loss: float,
    grad_norm_value,
    optimizer,
    dt_ms: float,
) -> None:
    lrs = get_optimizer_lrs(optimizer)

    muon_lr = lrs.get("muon_lr", float("nan"))
    adamw_lr = lrs.get("adamw_lr", lrs.get("lr", float("nan")))

    print(
        "┆   {:7d} │ {:7d} │ {:>8} │ {:>8} │ {:>8} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │ {:7.1f}ms".format(
            int(global_step),
            int(batch_idx + 1),
            fmt_num(loss_val),
            fmt_num(lm_loss),
            fmt_num(mtp_loss),
            fmt_num(moe_aux_loss, digits=2),
            fmt_num(grad_norm_value, digits=2),
            fmt_lr(muon_lr),
            fmt_lr(adamw_lr),
            float(dt_ms),
        )
    )



def print_epoch_summary(
    *,
    epoch: int,
    global_step: int,
    sec: float,
    optimizer,
    train_stats: Dict[str, Any],
    eval_stats: Optional[Dict[str, Any]],
    monitor_name: str,
    current_metric: float,
    best_metric: float,
    improved: bool,
) -> None:
    lrs = get_optimizer_lrs(optimizer)

    ds_section(f"Epoch {epoch:03d} summary")

    print(
        f"step={global_step} | time={fmt_hms(sec)} | "
        + " | ".join([f"{k}={fmt_lr(v)}" for k, v in lrs.items()])
    )

    print(
        "train -> "
        f"loss={train_stats['loss']:.5f} | "
        f"lm={train_stats['lm_loss']:.5f} | "
        f"mtp={train_stats['mtp_loss']:.5f} | "
        f"moe_aux={train_stats['moe_aux_loss']:.3e} | "
        f"grad_norm={train_stats['grad_norm']:.3e} | "
        f"optim_steps={int(train_stats['n_optimizer_steps'])}"
    )

    if eval_stats is not None:
        print("eval  -> " + format_metrics(eval_stats, prefix="eval"))

    print(
        f"monitor -> {monitor_name}={current_metric:.6f} | "
        f"best={best_metric:.6f} | improved={improved}"
    )

    print(ds_rule())

# ============================================================
# END-OF-EPOCH MODULE DIAGNOSTICS HELPERS
# ============================================================

@torch.no_grad()
def compute_one_batch_deepseek_diagnostics(
    *,
    model: nn.Module,
    dataloader,
    device: torch.device,
    precision: Dict[str, Any],
    prefix: str = "train",
) -> Dict[str, float]:
    """
    Compute DeepSeek module diagnostics on one batch.

    Intended for:
        module_metrics_every=0
        -> print diagnostics once after train_one_epoch and before eval.
    """
    was_training = model.training
    model.eval()

    batch = next(iter(dataloader))
    batch = normalize_lm_batch(batch)
    batch = move_batch_to_device(batch, device)

    with autocast_ctx(
        device=device,
        enabled=precision["amp_enabled"],
        amp_dtype=precision["amp_dtype_requested"],
        cache_enabled=precision["cache_enabled"],
        fallback_bf16_to_fp16=precision["fallback_bf16_to_fp16"],
    ):
        outputs = call_model_for_training(
            model=model,
            batch=batch,
            return_aux=True,
            need_weights=False,
        )

    metrics = compute_deepseek_module_metrics(
        outputs=outputs,
        model=model,
        prefix=prefix,
    )

    if was_training:
        model.train()

    return metrics


def print_deepseek_top10_critical_metrics(
    metrics: Dict[str, float],
    *,
    prefix: str = "train",
    title: Optional[str] = None,
    num_experts: Optional[int] = None,
    n_layers: Optional[int] = None,
) -> None:
    """
    Minimal top-10 critical diagnostics across:
        loss, MoE, MTP, mHC.
    """
    if title is not None:
        print("\n" + "═" * 96)
        print(title)
        print("═" * 96)

    rows = [
        (
            f"{prefix}/loss",
            "total objective; lower is better",
        ),
        (
            f"{prefix}/lm_loss",
            "main next-token CE; lower is better",
        ),
        (
            f"{prefix}/perplexity_from_lm_loss",
            "exp(lm_loss); lower is better",
        ),
        (
            f"{prefix}/mtp/raw_mtp_loss",
            "MTP raw CE; should fall with training",
        ),
        (
            f"{prefix}/moe/router_entropy_mean",
            (
                f"max≈log(E)={math.log(num_experts):.3f}; low => router collapse"
                if num_experts is not None and num_experts > 1
                else "low => router collapse"
            ),
        ),
        (
            f"{prefix}/moe/expert_fraction_min",
            "near 0 => dead/underused expert",
        ),
        (
            f"{prefix}/moe/expert_fraction_max",
            "too high => expert overload/collapse",
        ),
        (
            f"{prefix}/moe/dead_experts_across_layers",
            "should be 0",
        ),
        (
            f"{prefix}/mhc/B_row_sum_error_mean",
            "should be near 0",
        ),
        (
            f"{prefix}/mhc/B_column_sum_error_mean",
            "should be near 0",
        ),
    ]

    available = [(k, note) for k, note in rows if k in metrics]

    if not available:
        print("No critical DeepSeek diagnostics available.")
        return

    max_name = max(len(k.split("/")[-1]) for k, _ in available)

    print("Top-10 critical DeepSeek metrics")
    print("─" * 96)

    for key, note in available:
        short = key.split("/")[-1]
        value = fmt_num(metrics[key])
        print(f"  {short:<{max_name}} : {value:<12} # {note}")

    if title is not None:
        print("═" * 96)
