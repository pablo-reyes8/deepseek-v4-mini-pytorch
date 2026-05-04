# ============================================================
# TRAIN DEEPSEEK-V4
# High-level orchestration
# ============================================================


from training.train_one_epoch import * 
from training.eval_one_epoch import * 
from training.seed import * 
from training .scheduler import * 
from training.adam_optmizer import * 
from training.muon_optimizer import *  
from training.chekpoints import * 
from training.ema import * 


def train_deepseekv4(
    *,
    model: nn.Module,
    train_loader,
    val_loader=None,

    # Device / precision
    seed: int = 42,
    deterministic: bool = False,
    device: str = "auto",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True,

    # Optimizer
    optimizer_type: str = "adamw",  # "adamw", "muon_adamw"
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,

    # Muon
    muon_lr: Optional[float] = None,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_eps: float = 1e-7,
    muon_weight_decay: float = 0.0,

    # Scheduler
    total_steps: Optional[int] = None,
    warmup_steps: int = 500,
    min_muon_lr: Optional[float] = None,

    # EMA
    use_ema: bool = False,
    ema_decay: float = 0.999,
    ema_device: Optional[str] = "cpu",
    ema_update_after_step: int = 10,
    ema_update_every: int = 1,

    # Training
    epochs: int = 1,
    start_epoch: int = 0,
    global_step: int = 0,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    max_batches_per_epoch: Optional[int] = None,
    log_every: int = 10,

    # Module diagnostics
    # module_metrics_every > 0:
    #     print diagnostics inside train_one_epoch every N optimizer steps.
    # module_metrics_every == 0:
    #     print diagnostics once after train_one_epoch and before eval.
    # verbose=1:
    #     full diagnostics.
    # verbose=0:
    #     top-10 critical diagnostics only.
    module_metrics_every: Optional[int] = 150,
    print_module_diagnostics: bool = True,
    verbose: int = 1,

    log_grad_norm: bool = True,
    log_mem: bool = False,
    on_oom: str = "skip",

    # Eval
    eval_every: int = 1,
    eval_max_batches: Optional[int] = 50,
    eval_use_ema: bool = False,
    eval_log_every: Optional[int] = None,

    # Eval qualitative preview
    eval_preview: bool = True,
    eval_preview_batch_idx: int = 0,
    eval_preview_sample_idx: int = 0,
    eval_preview_max_context_tokens: int = 48,
    eval_preview_max_new_tokens: int = 24,
    eval_preview_temperature: float = 0.0,
    tokenizer=None,
    id2tok_fn=None,

    # Checkpoints
    ckpt_dir: str = "checkpoints/deepseekv4_mini",
    run_name: str = "deepseekv4_mini",
    save_every: int = 1,
    save_last: bool = True,
    keep_last_n_checkpoints: int = 3,
    monitor_name: str = "eval_loss",
    monitor_mode: str = "min",
    best_metric: Optional[float] = None,
    resume_path: Optional[str] = None,
    strict_resume: bool = True,
    restore_rng_state: bool = False,

    # Drive / metrics
    metrics_jsonl_name: str = "metrics.jsonl",
    drive_ckpt_dir: Optional[str] = None,
    copy_fixed_to_drive: bool = True,
    fixed_drive_name: str = "latest_deepseekv4.pt",
    fixed_drive_metrics_name: str = "metrics_deepseekv4.jsonl",

    # Metadata
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full DeepSeekV4 training orchestration.

    High-level loop calls:
        - train_one_epoch(...)
        - eval_one_epoch(...)

    Responsibilities:
        - seed/device/precision
        - optimizer creation: AdamW or Muon+AdamW
        - warmup cosine scheduler
        - optional EMA
        - resume
        - epoch-level training
        - epoch-level eval
        - qualitative autoregressive preview during eval
        - DeepSeek module diagnostics
        - checkpointing
        - Drive mirroring
        - JSONL metric logging
    """

    # ========================================================
    # Validate
    # ========================================================

    if optimizer_type not in {"adamw", "muon_adamw"}:
        raise ValueError(
            f"optimizer_type must be 'adamw' or 'muon_adamw', got {optimizer_type}."
        )

    if monitor_name not in VALID_MONITOR_NAMES:
        raise ValueError(
            f"monitor_name='{monitor_name}' no es válido. "
            f"Usa uno de: {sorted(VALID_MONITOR_NAMES)}"
        )

    if monitor_mode not in {"min", "max"}:
        raise ValueError(
            f"monitor_mode must be 'min' or 'max', got {monitor_mode}."
        )

    if verbose not in {0, 1}:
        raise ValueError(
            f"verbose must be 0 or 1. Got {verbose}."
        )

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir = str(ckpt_dir)
    metrics_path = Path(ckpt_dir) / metrics_jsonl_name

    # ========================================================
    # Seed / device / precision
    # ========================================================

    set_seed(seed, deterministic=deterministic)

    precision = setup_device_and_precision(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    device_obj = precision["device"]
    model = model.to(device_obj)

    # ========================================================
    # Optimizer
    # ========================================================

    if optimizer_type == "adamw":
        optimizer, opt_info = build_adamw_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            verbose=True,
        )

    elif optimizer_type == "muon_adamw":
        optimizer, opt_info = build_muon_adamw_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            muon_lr=muon_lr,
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            muon_ns_steps=muon_ns_steps,
            muon_eps=muon_eps,
            muon_weight_decay=muon_weight_decay,
            verbose=True,
        )

    # ========================================================
    # Scheduler
    # ========================================================

    if total_steps is None:
        steps_per_epoch = len(train_loader)

        if max_batches_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, int(max_batches_per_epoch))

        optim_steps_per_epoch = math.ceil(
            steps_per_epoch / max(1, int(grad_accum_steps))
        )
        total_steps = max(1, optim_steps_per_epoch * int(epochs))

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_learning_rate,
        min_muon_lr=min_muon_lr,
    )

    # ========================================================
    # EMA
    # ========================================================

    ema = None

    if use_ema:
        ema = EMA(
            model=model,
            decay=ema_decay,
            device=ema_device,
            use_num_updates=True,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
        )

    # ========================================================
    # Resume
    # ========================================================

    if resume_path is not None and os.path.exists(resume_path):
        ds_section("Resume checkpoint")

        state = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=precision.get("scaler", None),
            ema=ema,
            map_location="cpu",
            strict=strict_resume,
            load_optimizer=True,
            load_scheduler=True,
            load_scaler=True,
            load_ema=True,
            load_rng_state=restore_rng_state,
        )

        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["step"])
        best_metric = state["best_metric"]

        model = model.to(device_obj)

        print(f"[RESUME] path={resume_path}")
        print(
            f"[RESUME] start_epoch={start_epoch} | "
            f"global_step={global_step} | best_metric={best_metric}"
        )
        print(ds_rule())

    # ========================================================
    # Header
    # ========================================================

    print_deepseek_run_header(
        run_name=run_name,
        model=model,
        device=device_obj,
        precision=precision,
        optimizer_type=optimizer_type,
        ema=ema,
        epochs=epochs,
        start_epoch=start_epoch,
        global_step=global_step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        monitor_name=monitor_name,
        monitor_mode=monitor_mode,
        best_metric=best_metric,
        grad_clip=grad_clip,
        grad_accum_steps=grad_accum_steps,
        val_loader=val_loader,
        eval_every=max(1, int(eval_every)),
        eval_max_batches=eval_max_batches,
        drive_ckpt_dir=drive_ckpt_dir,
        fixed_drive_name=fixed_drive_name,
    )

    # ========================================================
    # Resolve module diagnostics policy
    # ========================================================

    # module_metrics_every > 0:
    #   train_one_epoch handles module diagnostics during the epoch.
    # module_metrics_every == 0:
    #   train_one_epoch does not print diagnostics;
    #   this wrapper prints once after train_one_epoch and before eval.
    if module_metrics_every is not None and module_metrics_every > 0:
        module_metrics_inside_epoch = int(module_metrics_every)
    else:
        module_metrics_inside_epoch = None

    print_full_module_diagnostics_inside_epoch = (
        bool(print_module_diagnostics)
        and int(verbose) >= 1
        and module_metrics_inside_epoch is not None
    )

    print_end_epoch_module_diagnostics = (
        bool(print_module_diagnostics)
        and module_metrics_every == 0
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    total_time = 0.0
    train_stats = None
    eval_stats = None
    combined_metrics = {}

    eval_every = max(1, int(eval_every))

    for epoch in range(start_epoch, epochs):
        ds_title(f"Epoch {epoch:03d}/{epochs - 1:03d}", ch="─")

        t0 = time.time()

        # ----------------------------------------------------
        # Train one epoch
        # ----------------------------------------------------

        train_stats, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device_obj,
            precision=precision,
            scheduler=scheduler,
            ema=ema,
            epoch=epoch,
            global_step=global_step,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            max_batches=max_batches_per_epoch,
            log_every=log_every,
            module_metrics_every=module_metrics_inside_epoch,
            print_module_diagnostics=print_full_module_diagnostics_inside_epoch,
            log_grad_norm=log_grad_norm,
            log_mem=log_mem,
            on_oom=on_oom,
            is_main_process=True,
        )

        # ----------------------------------------------------
        # Optional end-of-train-epoch module diagnostics
        # Happens before eval.
        # ----------------------------------------------------

        if print_end_epoch_module_diagnostics:
            raw_model = model.module if hasattr(model, "module") else model
            cfg = getattr(raw_model, "config", None)

            module_metrics = compute_one_batch_deepseek_diagnostics(
                model=model,
                dataloader=train_loader,
                device=device_obj,
                precision=precision,
                prefix="train",
            )

            if int(verbose) >= 1:
                print_deepseek_module_metrics(
                    module_metrics,
                    prefix="train",
                    precision=4,
                    title=(
                        f"DeepSeek-V4 module diagnostics | "
                        f"end of train epoch={epoch} step={global_step}"
                    ),
                    num_experts=getattr(cfg, "num_experts", None),
                    top_k_experts=getattr(cfg, "top_k_experts", None),
                    n_layers=getattr(cfg, "n_layers", None),
                )
            else:
                print_deepseek_top10_critical_metrics(
                    module_metrics,
                    prefix="train",
                    title=(
                        f"DeepSeek-V4 critical diagnostics | "
                        f"end of train epoch={epoch} step={global_step}"
                    ),
                    num_experts=getattr(cfg, "num_experts", None),
                    n_layers=getattr(cfg, "n_layers", None),
                )

        # ----------------------------------------------------
        # Eval one epoch
        # ----------------------------------------------------

        eval_stats = None

        if val_loader is not None and ((epoch - start_epoch) % eval_every) == 0:
            ds_section("Evaluation")

            eval_stats = eval_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device_obj,
                precision=precision,
                epoch=epoch,
                max_batches=eval_max_batches,
                topk=(1, 5),
                ema=ema,
                use_ema=bool(eval_use_ema and ema is not None),
                tokenizer=tokenizer,
                id2tok_fn=id2tok_fn,
                preview=eval_preview,
                preview_batch_idx=eval_preview_batch_idx,
                preview_sample_idx=eval_preview_sample_idx,
                preview_max_context_tokens=eval_preview_max_context_tokens,
                preview_max_new_tokens=eval_preview_max_new_tokens,
                preview_temperature=eval_preview_temperature,
                log_every=eval_log_every,
                is_main_process=True,
            )

            print(format_metrics(eval_stats, prefix="eval"))

        # ----------------------------------------------------
        # Metrics / monitor
        # ----------------------------------------------------

        sec = time.time() - t0
        total_time += sec

        combined_metrics = {}
        combined_metrics.update(prefixed_stats("train", train_stats))
        combined_metrics.update(prefixed_stats("eval", eval_stats))

        if eval_stats is not None:
            combined_metrics.update(eval_stats)
        else:
            combined_metrics.update(train_stats)

        current_metric = resolve_monitor_value(
            monitor_name=monitor_name,
            train_stats=train_stats,
            eval_stats=eval_stats,
        )

        improved = is_better_metric(
            current=current_metric,
            best=best_metric,
            mode=monitor_mode,
        )

        if improved:
            best_metric = current_metric

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        print_epoch_summary(
            epoch=epoch,
            global_step=global_step,
            sec=sec,
            optimizer=optimizer,
            train_stats=train_stats,
            eval_stats=eval_stats,
            monitor_name=monitor_name,
            current_metric=current_metric,
            best_metric=best_metric,
            improved=improved,
        )

        # ----------------------------------------------------
        # Metrics JSONL
        # ----------------------------------------------------

        metrics_record = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "time_sec": float(sec),
            "monitor_name": monitor_name,
            "monitor_value": float(current_metric),
            "best_metric": float(best_metric) if best_metric is not None else None,
            "improved": bool(improved),
            "optimizer_type": optimizer_type,
            **combined_metrics,
        }

        append_jsonl(metrics_path, metrics_record)

        if drive_ckpt_dir:
            copy_metrics_to_drive(
                src_path=metrics_path,
                drive_dir=drive_ckpt_dir,
                fixed_name=fixed_drive_metrics_name,
            )

        # ----------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------

        ds_section("Checkpointing")

        if improved:
            best_path = save_checkpoint(
                checkpoint_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=precision.get("scaler", None),
                ema=ema,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                config=config if config is not None else getattr(model, "config", None),
                extra_state={
                    "monitor_name": monitor_name,
                    "monitor_mode": monitor_mode,
                    "monitor_value": current_metric,
                    "train_stats": train_stats,
                    "eval_stats": eval_stats,
                    "optimizer_type": optimizer_type,
                    "opt_info": opt_info,
                },
                filename=f"{run_name}_best.pt",
                save_rng_state=True,
                keep_last_n=None,
                tag="best",
            )

            print(f"└─ [BEST] improved {monitor_name} -> {best_metric:.6f}")
            print(f"└─ [BEST] saved → {best_path}")

            if copy_fixed_to_drive and drive_ckpt_dir:
                copy_ckpt_to_drive_fixed(
                    src_path=best_path,
                    drive_dir=drive_ckpt_dir,
                    fixed_name=f"best_{fixed_drive_name}",
                )
        else:
            print("└─ [BEST] no improvement")

        should_save_epoch = (
            save_every is not None
            and save_every > 0
            and ((epoch % save_every == 0) or (epoch == epochs - 1))
        )

        if should_save_epoch:
            ckpt_path = save_checkpoint(
                checkpoint_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=precision.get("scaler", None),
                ema=ema,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                config=config if config is not None else getattr(model, "config", None),
                extra_state={
                    "monitor_name": monitor_name,
                    "monitor_mode": monitor_mode,
                    "monitor_value": current_metric,
                    "train_stats": train_stats,
                    "eval_stats": eval_stats,
                    "optimizer_type": optimizer_type,
                    "opt_info": opt_info,
                },
                filename=f"{run_name}_e{epoch:03d}.pt",
                save_rng_state=True,
                keep_last_n=keep_last_n_checkpoints,
            )

            print(f"└─ [CKPT] saved → {ckpt_path}")

            if copy_fixed_to_drive and drive_ckpt_dir:
                copy_ckpt_to_drive_fixed(
                    src_path=ckpt_path,
                    drive_dir=drive_ckpt_dir,
                    fixed_name=fixed_drive_name,
                )
        else:
            print("└─ [CKPT] skipped by save_every")

    # ========================================================
    # Final checkpoint
    # ========================================================

    if save_last and train_stats is not None:
        ds_section("Final checkpoint")

        last_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=precision.get("scaler", None),
            ema=ema,
            epoch=epochs - 1,
            step=global_step,
            best_metric=best_metric,
            config=config if config is not None else getattr(model, "config", None),
            extra_state={
                "monitor_name": monitor_name,
                "monitor_mode": monitor_mode,
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "optimizer_type": optimizer_type,
                "opt_info": opt_info,
            },
            filename=f"{run_name}_last_manual.pt",
            save_rng_state=True,
            keep_last_n=None,
            tag="last",
        )

        print(f"└─ [CKPT] final saved → {last_path}")

        if copy_fixed_to_drive and drive_ckpt_dir:
            copy_ckpt_to_drive_fixed(
                src_path=last_path,
                drive_dir=drive_ckpt_dir,
                fixed_name=fixed_drive_name,
            )

    # ========================================================
    # Done
    # ========================================================

    ds_title("Training complete")
    print(f"Total time : {fmt_hms(total_time)}")
    print(f"Final step : {global_step}")
    print(f"Best metric: {best_metric}")
    print(f"Metrics    : {metrics_path}")
    print(f"Checkpoints: {ckpt_dir}")
    print(ds_rule())

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "ema": ema,
        "precision": precision,
        "opt_info": opt_info,
        "global_step": global_step,
        "best_metric": best_metric,
        "last_train_stats": train_stats,
        "last_eval_stats": eval_stats,
        "metrics_path": str(metrics_path),
        "checkpoint_dir": ckpt_dir,
    }