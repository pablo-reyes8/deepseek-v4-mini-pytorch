# Training Pipeline

The training stack is centered on `train_deepseekv4`.

## High-Level Flow

```text
set seed
resolve device and precision
build optimizer: AdamW or Muon+AdamW
build warmup cosine scheduler
optionally create EMA
optionally resume checkpoint

for epoch:
    train_one_epoch
    optionally compute module diagnostics
    optionally eval_one_epoch
    save checkpoints
    append metrics JSONL
```

## `train_one_epoch`

What it does:

- Normalizes dataloader batches into model kwargs.
- Moves tensors to target device.
- Applies AMP/autocast when enabled.
- Computes model loss.
- Handles gradient accumulation.
- Clips gradients.
- Steps optimizer, scheduler, and EMA.
- Logs loss, grad norm, LR, and optional module diagnostics.

Important controls:

- `grad_accum_steps`: number of microbatches per optimizer step.
- `grad_clip`: max gradient norm.
- `max_batches_per_epoch`: caps work for smoke runs.
- `module_metrics_every`: controls how often module diagnostics are computed.
- `on_oom`: currently supports skipping CUDA OOM batches.

## `eval_one_epoch`

What it does:

- Computes LM metrics: loss, perplexity, accuracy, top-k accuracy, entropy.
- Supports EMA evaluation.
- Can print qualitative teacher-forced and autoregressive previews.

Important controls:

- `eval_every`: epoch interval for evaluation.
- `eval_max_batches`: caps validation work.
- `eval_use_ema`: evaluates EMA weights if available.
- `eval_preview`: enables qualitative preview.
- `eval_preview_max_context_tokens`: context shown in preview.
- `eval_preview_max_new_tokens`: generated continuation length.
- `eval_preview_temperature`: greedy when 0, sampling when > 0.

## Checkpoints and Metrics

Checkpoints can include:

- model state
- optimizer state
- scheduler state
- scaler state
- EMA state
- RNG state
- config snapshot
- extra run metadata

Metrics are appended as JSONL for easy inspection and plotting.
