# ============================================================
# EVALUATION METRICS
# DeepSeek-V4 Mini Training Stack
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

from data.data_utils import * 
from training.autocast import * 


# ============================================================
# Small helpers
# ============================================================

def get_output_value(outputs: Any, key: str, default: Any = None) -> Any:
    """
    Robustly get value from model outputs.

    Supports:
        - dict outputs
        - objects with attributes
    """
    if isinstance(outputs, dict):
        return outputs.get(key, default)

    return getattr(outputs, key, default)


def get_logits_from_outputs(outputs: Any) -> torch.Tensor:
    """
    Extract logits from model outputs.
    """
    logits = get_output_value(outputs, "logits", None)

    if logits is None:
        logits = get_output_value(outputs, "lm_logits", None)

    if logits is None:
        logits = get_output_value(outputs, "prediction_logits", None)

    if logits is None:
        raise KeyError(
            "Could not find logits in model outputs. Expected one of: "
            "'logits', 'lm_logits', 'prediction_logits'."
        )

    return logits


def get_loss_from_outputs(outputs: Any) -> Optional[torch.Tensor]:
    """
    Extract loss from model outputs if available.
    """
    loss = get_output_value(outputs, "loss", None)
    return loss


def get_model_config(model: nn.Module) -> Any:
    """
    Get model.config safely, handling DDP/DataParallel wrappers.
    """
    raw_model = model.module if hasattr(model, "module") else model
    return getattr(raw_model, "config", None)


def prepare_logits_and_labels_for_metrics(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align logits and labels according to model.config.labels_are_shifted.

    Case 1:
        labels_are_shifted=True

        input_ids: [x0, x1, ..., xT-1]
        labels:    [x1, x2, ..., xT]

        Metrics use:
            logits[:, :, :] vs labels[:, :]

    Case 2:
        labels_are_shifted=False

        input_ids: [x0, x1, ..., xT-1]
        labels:    [x0, x1, ..., xT-1]

        Metrics use:
            logits[:, :-1, :] vs labels[:, 1:]
    """
    cfg = get_model_config(model)
    labels_are_shifted = True

    if cfg is not None and hasattr(cfg, "labels_are_shifted"):
        labels_are_shifted = bool(cfg.labels_are_shifted)

    if labels_are_shifted:
        T = min(logits.size(1), labels.size(1))
        return logits[:, :T, :], labels[:, :T]

    T = min(logits.size(1) - 1, labels.size(1) - 1)

    if T <= 0:
        raise ValueError(
            "Cannot compute shifted metrics because sequence length is too short."
        )

    return logits[:, :T, :], labels[:, 1:T + 1]


def build_valid_token_mask(
    model: nn.Module,
    labels: torch.Tensor,
    ignore_index: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Build mask for valid labels.

    Excludes:
        - ignore_index
        - pad_token_id, if model.config.ignore_pad_token_in_loss=True
    """
    cfg = get_model_config(model)

    if ignore_index is None:
        ignore_index = getattr(cfg, "ignore_index", -100) if cfg is not None else -100

    if pad_token_id is None:
        pad_token_id = getattr(cfg, "pad_token_id", None) if cfg is not None else None

    ignore_pad = True
    if cfg is not None and hasattr(cfg, "ignore_pad_token_in_loss"):
        ignore_pad = bool(cfg.ignore_pad_token_in_loss)

    mask = labels != int(ignore_index)

    if pad_token_id is not None and ignore_pad:
        mask = mask & (labels != int(pad_token_id))

    return mask


# ============================================================
# Core LM metrics from logits
# ============================================================

@torch.no_grad()
def compute_lm_metrics_from_logits(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: Sequence[int] = (1, 5),
    ignore_index: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute autoregressive LM metrics from logits and labels.

    Returns:
        {
            "ce_sum",
            "valid_tokens",
            "loss",
            "perplexity",
            "token_accuracy",
            "top_5_accuracy",
            "sequence_accuracy",
            "mean_confidence",
            "mean_true_token_prob",
            "mean_entropy",
        }
    """
    logits, labels = prepare_logits_and_labels_for_metrics(
        model=model,
        logits=logits,
        labels=labels,
    )

    valid_mask = build_valid_token_mask(
        model=model,
        labels=labels,
        ignore_index=ignore_index,
        pad_token_id=pad_token_id,
    )

    valid_tokens = int(valid_mask.sum().item())

    if valid_tokens == 0:
        return {
            "ce_sum": 0.0,
            "valid_tokens": 0,
            "loss": float("nan"),
            "perplexity": float("nan"),
            "token_accuracy": float("nan"),
            "sequence_accuracy": float("nan"),
            "mean_confidence": float("nan"),
            "mean_true_token_prob": float("nan"),
            "mean_entropy": float("nan"),
        }

    V = logits.size(-1)

    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    mask_flat = valid_mask.reshape(-1)

    valid_logits = logits_flat[mask_flat]
    valid_labels = labels_flat[mask_flat]

    # CE sum in fp32 for stable aggregation.
    ce_sum = F.cross_entropy(
        valid_logits.float(),
        valid_labels,
        reduction="sum",
    )

    loss = ce_sum / max(valid_tokens, 1)

    # Avoid overflow in exp.
    ppl = math.exp(min(float(loss.detach().cpu().item()), 50.0))

    probs = torch.softmax(valid_logits.float(), dim=-1)
    pred = torch.argmax(probs, dim=-1)

    token_acc = (pred == valid_labels).float().mean()

    metrics = {
        "ce_sum": float(ce_sum.detach().cpu().item()),
        "valid_tokens": float(valid_tokens),
        "loss": float(loss.detach().cpu().item()),
        "perplexity": float(ppl),
        "token_accuracy": float(token_acc.detach().cpu().item()),
    }

    # Top-k accuracies.
    max_k = max(topk) if len(topk) > 0 else 1
    max_k = min(max_k, V)

    topk_indices = torch.topk(valid_logits.float(), k=max_k, dim=-1).indices

    for k in topk:
        k_eff = min(int(k), V)
        correct_k = (topk_indices[:, :k_eff] == valid_labels[:, None]).any(dim=-1)
        metrics[f"top_{k_eff}_accuracy"] = float(correct_k.float().mean().detach().cpu().item())

    # Sequence accuracy: sequence counts as correct if all valid positions are correct.
    # This is harsh for language modeling, but useful for tiny smoke tests.
    with torch.no_grad():
        full_pred = torch.argmax(logits.float(), dim=-1)
        token_correct_full = (full_pred == labels) | (~valid_mask)
        seq_has_any_valid = valid_mask.any(dim=1)
        seq_correct = token_correct_full.all(dim=1) & seq_has_any_valid

        if seq_has_any_valid.any():
            sequence_accuracy = seq_correct.float().sum() / seq_has_any_valid.float().sum()
            metrics["sequence_accuracy"] = float(sequence_accuracy.detach().cpu().item())
        else:
            metrics["sequence_accuracy"] = float("nan")

    # Confidence and entropy.
    confidence = probs.max(dim=-1).values.mean()
    true_token_prob = probs.gather(1, valid_labels[:, None]).squeeze(1).mean()
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()

    metrics["mean_confidence"] = float(confidence.detach().cpu().item())
    metrics["mean_true_token_prob"] = float(true_token_prob.detach().cpu().item())
    metrics["mean_entropy"] = float(entropy.detach().cpu().item())

    return metrics


# ============================================================
# Aggregator
# ============================================================

class MetricAverager:
    """
    Weighted metric accumulator.

    Loss is aggregated by token count using ce_sum / valid_tokens.
    Other metrics are weighted by valid_tokens when possible.
    """

    def __init__(self):
        self.total_ce_sum = 0.0
        self.total_valid_tokens = 0.0
        self.weighted = {}
        self.weights = {}

    def update(self, metrics: Dict[str, float]) -> None:
        valid_tokens = float(metrics.get("valid_tokens", 0.0))

        if "ce_sum" in metrics:
            self.total_ce_sum += float(metrics["ce_sum"])

        self.total_valid_tokens += valid_tokens

        for key, value in metrics.items():
            if key in {"ce_sum", "valid_tokens", "loss", "perplexity"}:
                continue

            if value is None:
                continue

            try:
                value = float(value)
            except Exception:
                continue

            if not math.isfinite(value):
                continue

            w = max(valid_tokens, 1.0)

            self.weighted[key] = self.weighted.get(key, 0.0) + value * w
            self.weights[key] = self.weights.get(key, 0.0) + w

    def compute(self) -> Dict[str, float]:
        output = {}

        if self.total_valid_tokens > 0:
            loss = self.total_ce_sum / self.total_valid_tokens
            output["loss"] = float(loss)
            output["perplexity"] = float(math.exp(min(loss, 50.0)))
            output["valid_tokens"] = float(self.total_valid_tokens)
        else:
            output["loss"] = float("nan")
            output["perplexity"] = float("nan")
            output["valid_tokens"] = 0.0

        for key, total in self.weighted.items():
            denom = self.weights.get(key, 0.0)
            output[key] = float(total / denom) if denom > 0 else float("nan")

        return output


# ============================================================
# Evaluate one dataloader
# ============================================================

@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    dataloader,
    device: torch.device,
    precision: Optional[Dict[str, Any]] = None,
    max_batches: Optional[int] = None,
    topk: Sequence[int] = (1, 5),
    ema: Optional[Any] = None,
    use_ema: bool = False,
    return_aux_loss_items: bool = True,
) -> Dict[str, float]:
    """
    Evaluate LM model over a dataloader.

    Intended use:
        - call between epochs
        - optionally evaluate with EMA weights

    Args:
        model:
            DeepSeekV4LM.
        dataloader:
            Validation dataloader.
        device:
            Training/eval device.
        precision:
            Dict returned by setup_device_and_precision.
        max_batches:
            Optional cap for faster validation.
        topk:
            Top-k accuracies to compute.
        ema:
            Optional EMA object.
        use_ema:
            If True, temporarily swaps EMA weights into model.
        return_aux_loss_items:
            If True, averages scalar loss items from model outputs when present.

    Returns:
        Dict of metrics.
    """
    was_training = model.training
    model.eval()

    if precision is None:
        precision = {
            "amp_enabled": False,
            "amp_dtype_requested": "fp32",
            "cache_enabled": True,
            "fallback_bf16_to_fp16": True,
        }

    averager = MetricAverager()
    aux_sums = {}
    aux_counts = {}

    def _eval_loop():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = normalize_lm_batch(batch)
            batch = move_batch_to_device(batch, device)

            with autocast_ctx(
                device=device,
                enabled=precision["amp_enabled"],
                amp_dtype=precision["amp_dtype_requested"],
                cache_enabled=precision["cache_enabled"],
                fallback_bf16_to_fp16=precision["fallback_bf16_to_fp16"],
            ):
                outputs = model(**batch)

            logits = get_logits_from_outputs(outputs)
            labels = batch["labels"]

            metrics = compute_lm_metrics_from_logits(
                model=model,
                logits=logits,
                labels=labels,
                topk=topk,
            )

            averager.update(metrics)

            if return_aux_loss_items:
                # Collect scalar diagnostics if present.
                # Common output keys: lm_loss, mtp_loss, moe_aux_loss, loss.
                for key in [
                    "loss",
                    "lm_loss",
                    "mtp_loss",
                    "moe_aux_loss",
                    "balance_loss",
                    "sequence_balance_loss",
                ]:
                    value = _get_output_value(outputs, key, None)

                    if value is None:
                        continue

                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            continue
                        value = float(value.detach().float().cpu().item())
                    else:
                        try:
                            value = float(value)
                        except Exception:
                            continue

                    if math.isfinite(value):
                        aux_sums[key] = aux_sums.get(key, 0.0) + value
                        aux_counts[key] = aux_counts.get(key, 0) + 1

    if use_ema:
        if ema is None:
            raise ValueError("use_ema=True but ema=None.")

        with ema.average_parameters(model):
            _eval_loop()
    else:
        _eval_loop()

    metrics = averager.compute()

    for key, total in aux_sums.items():
        count = aux_counts.get(key, 0)
        if count > 0:
            metrics[f"model_{key}"] = total / count

    metrics["num_batches"] = float(
        max_batches if max_batches is not None else len(dataloader)
    )

    metrics["used_ema"] = bool(use_ema)

    if was_training:
        model.train()

    return metrics


# ============================================================
# Pretty print
# ============================================================

def format_metrics(metrics: Dict[str, float], prefix: str = "val") -> str:
    """
    Compact formatting for logs.
    """
    keys = [
        "loss",
        "perplexity",
        "token_accuracy",
        "top_5_accuracy",
        "sequence_accuracy",
        "mean_confidence",
        "mean_true_token_prob",
        "mean_entropy",
        "valid_tokens",
    ]

    parts = []

    for key in keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if isinstance(value, bool):
            parts.append(f"{prefix}/{key}={value}")
        elif isinstance(value, (int, float)):
            if key == "valid_tokens":
                parts.append(f"{prefix}/{key}={int(value)}")
            else:
                parts.append(f"{prefix}/{key}={value:.4f}")
        else:
            parts.append(f"{prefix}/{key}={value}")

    return " | ".join(parts)