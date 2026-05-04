# ============================================================
# EVAL ONE EPOCH
# DeepSeek-V4 Mini
# ============================================================

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from training.train_one_epoch import * 
from training.train_one_epoch_utils import * 
from training.training_metrics import * 


def decode_ids(ids, tokenizer=None, id2tok_fn=None, max_tokens: Optional[int] = None) -> str:
    """
    Decode token ids using either:
      - id2tok_fn(list[int]) -> str
      - tokenizer.decode(list[int]) -> str
      - tokenizer.idx_to_token mapping
      - fallback: space-separated ids
    """
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()

    ids = [int(x) for x in ids]

    if max_tokens is not None:
        ids = ids[:max_tokens]

    if id2tok_fn is not None:
        try:
            return id2tok_fn(ids)
        except Exception:
            pass

    if tokenizer is not None:
        if hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(ids)
            except Exception:
                pass

        if hasattr(tokenizer, "idx_to_token"):
            try:
                return " ".join(tokenizer.idx_to_token.get(int(i), f"<{int(i)}>") for i in ids)
            except Exception:
                pass

    return " ".join(str(int(i)) for i in ids)


@torch.no_grad()
def autoregressive_preview(
    *,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    precision: Dict[str, Any],
    tokenizer=None,
    id2tok_fn=None,
    max_context_tokens: int = 48,
    max_new_tokens: int = 24,
    sample_idx: int = 0,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Qualitative eval preview.

    Produces:
      1. teacher-forced argmax:
         CTX / REF / HYP over the existing validation sequence.

      2. short autoregressive rollout:
         takes a prefix from CTX and generates max_new_tokens greedily
         or with sampling if temperature > 0.

    Notes:
      - This is for qualitative inspection only.
      - It is intentionally small and runs only during eval.
    """
    raw_model = model.module if hasattr(model, "module") else model
    cfg = getattr(raw_model, "config", None)

    pad_token_id = getattr(cfg, "pad_token_id", None) if cfg is not None else None
    max_seq_len = getattr(cfg, "max_seq_len", None) if cfg is not None else None

    input_ids = batch["input_ids"]
    labels = batch.get("labels", None)

    sample_idx = min(int(sample_idx), input_ids.shape[0] - 1)

    x = input_ids[sample_idx:sample_idx + 1].to(device)

    y = None
    if labels is not None:
        y = labels[sample_idx:sample_idx + 1].to(device)

    # --------------------------------------------------------
    # Teacher-forced argmax preview
    # --------------------------------------------------------
    with autocast_ctx(
        device=device,
        enabled=precision["amp_enabled"],
        amp_dtype=precision["amp_dtype_requested"],
        cache_enabled=precision["cache_enabled"],
        fallback_bf16_to_fp16=precision["fallback_bf16_to_fp16"],
    ):
        outputs = call_model_for_training(
            model=model,
            batch={
                "input_ids": x,
                "labels": y,
                "attention_mask": None,
                "mtp_labels": None,
            },
            return_aux=False,
            need_weights=False,
        )

    logits = get_logits_from_outputs(outputs)
    pred_ids = logits.argmax(dim=-1)

    show_len = min(max_context_tokens, x.shape[1])

    ctx_ids = x[0, :show_len]
    pred_show = pred_ids[0, :show_len]

    if y is not None:
        ref_show = y[0, :show_len]
    else:
        ref_show = None

    ctx_text = decode_ids(ctx_ids, tokenizer=tokenizer, id2tok_fn=id2tok_fn)
    hyp_text = decode_ids(pred_show, tokenizer=tokenizer, id2tok_fn=id2tok_fn)
    ref_text = decode_ids(ref_show, tokenizer=tokenizer, id2tok_fn=id2tok_fn) if ref_show is not None else None

    # --------------------------------------------------------
    # Autoregressive rollout preview
    # --------------------------------------------------------
    prefix_len = min(max_context_tokens, x.shape[1])
    generated = x[:, :prefix_len].clone()

    for _ in range(max_new_tokens):
        model_input = generated

        if max_seq_len is not None and model_input.shape[1] > max_seq_len:
            model_input = model_input[:, -max_seq_len:]

        with autocast_ctx(
            device=device,
            enabled=precision["amp_enabled"],
            amp_dtype=precision["amp_dtype_requested"],
            cache_enabled=precision["cache_enabled"],
            fallback_bf16_to_fp16=precision["fallback_bf16_to_fp16"],
        ):
            out_gen = call_model_for_training(
                model=model,
                batch={"input_ids": model_input},
                return_aux=False,
                need_weights=False,
            )

        gen_logits = get_logits_from_outputs(out_gen)
        next_logits = gen_logits[:, -1, :].float()

        if temperature is not None and temperature > 0:
            probs = torch.softmax(next_logits / float(temperature), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_id], dim=1)

        if pad_token_id is not None and int(next_id.item()) == int(pad_token_id):
            # Do not necessarily stop on pad, but for tiny synthetic datasets this avoids ugly tails.
            pass

    generated_text = decode_ids(
        generated[0],
        tokenizer=tokenizer,
        id2tok_fn=id2tok_fn,
        max_tokens=prefix_len + max_new_tokens,
    )

    return {
        "ctx": ctx_text,
        "ref": ref_text,
        "hyp": hyp_text,
        "generated": generated_text,
        "prefix_len": int(prefix_len),
        "max_new_tokens": int(max_new_tokens),
    }


def print_eval_preview(preview: Dict[str, Any], title: str = "Eval qualitative preview") -> None:
    print("\n" + "─" * 110)
    print(title)
    print("─" * 110)
    print("Teacher-forced argmax:")
    print(f"  CTX: {repr(preview.get('ctx', ''))}")

    if preview.get("ref", None) is not None:
        print(f"  REF: {repr(preview.get('ref', ''))}")

    print(f"  HYP: {repr(preview.get('hyp', ''))}")
    print()
    print(f"Autoregressive rollout ({preview.get('max_new_tokens', '?')} new tokens):")
    print(f"  GEN: {repr(preview.get('generated', ''))}")
    print("─" * 110)


@torch.no_grad()
def eval_one_epoch(
    *,
    model: nn.Module,
    dataloader,
    device: str | torch.device,
    precision: Dict[str, Any],
    epoch: int = 0,
    max_batches: Optional[int] = None,
    topk: Sequence[int] = (1, 5),
    ema: Optional[Any] = None,
    use_ema: bool = False,
    tokenizer=None,
    id2tok_fn=None,
    preview: bool = True,
    preview_batch_idx: int = 0,
    preview_sample_idx: int = 0,
    preview_max_context_tokens: int = 48,
    preview_max_new_tokens: int = 24,
    preview_temperature: float = 0.0,
    log_every: Optional[int] = None,
    is_main_process: bool = True,
) -> Dict[str, float]:
    """
    Evaluate one epoch for DeepSeekV4LM.

    Computes:
      - LM loss/perplexity/token accuracy/top-k/entropy
      - optional EMA eval
      - qualitative preview:
          CTX / REF / HYP teacher-forced argmax
          GEN autoregressive rollout

    Returns:
      eval_stats dict.
    """
    device = torch.device(device)

    was_training = model.training
    model.to(device)
    model.eval()

    averager = MetricAverager()
    aux_sums = {}
    aux_counts = {}

    t0 = time.time()
    n_batches = 0
    n_samples = 0
    preview_done = False

    def _eval_loop():
        nonlocal n_batches, n_samples, preview_done

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = normalize_lm_batch(batch)
            batch = move_batch_to_device(batch, device)

            n_batches += 1
            n_samples += int(batch["input_ids"].shape[0])

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
                    return_aux=False,
                    need_weights=False,
                )

            logits = get_logits_from_outputs(outputs)
            labels = batch["labels"]

            metrics = compute_lm_metrics_from_logits(
                model=model,
                logits=logits,
                labels=labels,
                topk=topk,
            )

            averager.update(metrics)

            # Collect scalar model-side loss diagnostics if present.
            for key in [
                "loss",
                "lm_loss",
                "mtp_loss",
                "moe_aux_loss",
                "raw_mtp_loss",
                "weighted_mtp_loss",
            ]:
                value = get_output_value(outputs, key, None)

                if value is None:
                    continue

                value = safe_float(value, default=float("nan"))

                if math.isfinite(value):
                    aux_sums[key] = aux_sums.get(key, 0.0) + value
                    aux_counts[key] = aux_counts.get(key, 0) + 1

            # Qualitative preview only once per eval.
            if (
                preview
                and (not preview_done)
                and batch_idx == int(preview_batch_idx)
                and is_main_process
            ):
                preview_dict = autoregressive_preview(
                    model=model,
                    batch=batch,
                    device=device,
                    precision=precision,
                    tokenizer=tokenizer,
                    id2tok_fn=id2tok_fn,
                    max_context_tokens=preview_max_context_tokens,
                    max_new_tokens=preview_max_new_tokens,
                    sample_idx=preview_sample_idx,
                    temperature=preview_temperature,
                )

                print_eval_preview(
                    preview_dict,
                    title=f"Eval qualitative preview | epoch={epoch} | batch={batch_idx}",
                )

                preview_done = True

            if log_every is not None and log_every > 0 and is_main_process:
                if n_batches % log_every == 0:
                    partial = averager.compute()
                    print(
                        f"eval batch {n_batches:04d} | "
                        f"loss={partial['loss']:.4f} | "
                        f"ppl={partial['perplexity']:.2f} | "
                        f"tok_acc={partial.get('token_accuracy', float('nan')):.4f}"
                    )

    if use_ema:
        if ema is None:
            raise ValueError("use_ema=True but ema=None.")

        with ema.average_parameters(model):
            _eval_loop()
    else:
        _eval_loop()

    stats = averager.compute()

    for key, total in aux_sums.items():
        count = aux_counts.get(key, 0)
        if count > 0:
            stats[f"model_{key}"] = total / count

    stats["n_eval_batches"] = float(n_batches)
    stats["n_eval_samples"] = float(n_samples)
    stats["eval_time_sec"] = float(time.time() - t0)
    stats["used_ema"] = float(bool(use_ema))

    if was_training:
        model.train()

    return stats