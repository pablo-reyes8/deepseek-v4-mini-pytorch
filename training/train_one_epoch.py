# ============================================================
# TRAIN ONE EPOCH
# ============================================================

from training.train_one_epoch_utils import * 

def train_one_epoch(
    *,
    model: nn.Module,
    dataloader,
    optimizer: Any,
    device: str | torch.device,
    precision: Dict[str, Any],
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    max_batches: Optional[int] = None,
    log_every: int = 10,
    module_metrics_every: Optional[int] = 50,
    print_module_diagnostics: bool = True,
    log_grad_norm: bool = True,
    log_mem: bool = False,
    on_oom: str = "skip",
    is_main_process: bool = True,
) -> Tuple[Dict[str, float], int]:
    """
    Train one full epoch for DeepSeekV4LM.

    Uses:
        - model internal loss: outputs["loss"]
        - AMP/autocast
        - AdamW or HybridMuonAdamW
        - scheduler
        - EMA
        - grad clipping
        - DeepSeek module diagnostics
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    running = {
        "loss": 0.0,
        "lm_loss": 0.0,
        "mtp_loss": 0.0,
        "moe_aux_loss": 0.0,
        "grad_norm": 0.0,
    }

    n_seen_batches = 0
    n_seen_samples = 0
    n_optimizer_steps = 0
    n_grad_logs = 0
    n_module_logs = 0

    t_epoch = time.time()

    if is_main_process and log_every:
        print_in_epoch_header()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        try:
            t0 = time.perf_counter()

            batch = normalize_lm_batch(batch)
            batch = move_batch_to_device(batch, device)

            B = int(batch["input_ids"].shape[0])
            n_seen_samples += B

            step_now = ((batch_idx + 1) % grad_accum_steps) == 0

            should_log_modules = (
                module_metrics_every is not None
                and module_metrics_every > 0
                and step_now
                and ((global_step + 1) % module_metrics_every == 0)
            )

            scaler = precision.get("scaler", None)

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
                    return_aux=bool(should_log_modules),
                    need_weights=False,
                )

                loss = get_loss_from_outputs(outputs)
                loss_for_backward = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            optimizer_step_happened = False
            grad_norm_value = None

            if step_now:
                if scaler is not None:
                    unscale_optimizer_grads(scaler, optimizer)

                if log_grad_norm or (grad_clip is not None and grad_clip > 0):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(grad_clip) if grad_clip is not None else 1e9,
                    )
                    grad_norm_value = float(grad_norm.detach().float().cpu().item())

                if scaler is not None:
                    old_scale = scaler.get_scale()

                    scaler_step_optimizer(scaler, optimizer)
                    scaler.update()

                    new_scale = scaler.get_scale()
                    optimizer_step_happened = new_scale >= old_scale
                else:
                    optimizer.step()
                    optimizer_step_happened = True

                if optimizer_step_happened:
                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1
                    n_optimizer_steps += 1

                    if ema is not None:
                        try:
                            ema.update(model, step=global_step)
                        except TypeError:
                            ema.update(model)

                optimizer.zero_grad(set_to_none=True)

            loss_val = safe_float(loss)
            running["loss"] += loss_val

            if isinstance(outputs, dict):
                lm_loss = outputs.get("lm_loss", None)
                mtp_loss = outputs.get("mtp_loss", None)
                moe_aux_loss = outputs.get("moe_aux_loss", None)
            else:
                lm_loss = getattr(outputs, "lm_loss", None)
                mtp_loss = getattr(outputs, "mtp_loss", None)
                moe_aux_loss = getattr(outputs, "moe_aux_loss", None)

            lm_loss_val = safe_float(lm_loss, 0.0)
            mtp_loss_val = safe_float(mtp_loss, 0.0)
            moe_aux_loss_val = safe_float(moe_aux_loss, 0.0)

            running["lm_loss"] += lm_loss_val
            running["mtp_loss"] += mtp_loss_val
            running["moe_aux_loss"] += moe_aux_loss_val

            if grad_norm_value is not None:
                running["grad_norm"] += grad_norm_value
                n_grad_logs += 1

            n_seen_batches += 1

            should_log = (
                log_every
                and step_now
                and optimizer_step_happened
                and (global_step % log_every == 0)
            )

            if should_log and is_main_process:
                dt_ms = (time.perf_counter() - t0) * 1000.0

                print_in_epoch_row(
                    global_step=global_step,
                    batch_idx=batch_idx,
                    loss_val=loss_val,
                    lm_loss=lm_loss_val,
                    mtp_loss=mtp_loss_val,
                    moe_aux_loss=moe_aux_loss_val,
                    grad_norm_value=grad_norm_value,
                    optimizer=optimizer,
                    dt_ms=dt_ms,
                )

                if log_mem:
                    alloc, reserv = gpu_mem_mb(device)
                    print(f"┆        memory: allocated={alloc:.0f}MB | reserved={reserv:.0f}MB")

            if should_log_modules:
                n_module_logs += 1

                module_metrics = compute_deepseek_module_metrics(
                    outputs=outputs,
                    model=model,
                    prefix="train",
                )

                if is_main_process and print_module_diagnostics:
                    raw_model = model.module if hasattr(model, "module") else model
                    cfg = getattr(raw_model, "config", None)

                    print_deepseek_module_metrics(
                        module_metrics,
                        prefix="train",
                        precision=4,
                        title=f"DeepSeek-V4 module diagnostics | epoch={epoch} step={global_step}",
                        num_experts=getattr(cfg, "num_experts", None),
                        top_k_experts=getattr(cfg, "top_k_experts", None),
                        n_layers=getattr(cfg, "n_layers", None),
                    )

                    if log_every:
                        print_in_epoch_header()

        except RuntimeError as e:
            if ("out of memory" in str(e).lower()) and (on_oom == "skip"):
                import gc

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                optimizer.zero_grad(set_to_none=True)

                if is_main_process:
                    print(f"[WARN][OOM] Batch {batch_idx} omitido. Limpié cache y sigo.")

                continue

            raise

    denom = max(1, n_seen_batches)

    epoch_stats = {
        "loss": running["loss"] / denom,
        "lm_loss": running["lm_loss"] / denom,
        "mtp_loss": running["mtp_loss"] / denom,
        "moe_aux_loss": running["moe_aux_loss"] / denom,
        "grad_norm": running["grad_norm"] / max(1, n_grad_logs),
        "n_seen_batches": float(n_seen_batches),
        "n_seen_samples": float(n_seen_samples),
        "n_optimizer_steps": float(n_optimizer_steps),
        "n_module_logs": float(n_module_logs),
        "epoch_time_sec": float(time.time() - t_epoch),
    }

    return epoch_stats, global_step