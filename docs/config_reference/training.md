# Training Config Reference

Primary entrypoint: `train_deepseekv4`.

## Device and Precision

| Parameter | Meaning |
| :--- | :--- |
| `seed` | Global random seed. |
| `deterministic` | Enables deterministic PyTorch behavior where possible. |
| `device` | `auto`, `cpu`, `cuda`, or another torch device string. |
| `amp_enabled` | Enables automatic mixed precision when supported. |
| `amp_dtype` | Requested AMP dtype: `bf16`, `fp16`, or `fp32`. |
| `fallback_bf16_to_fp16` | Uses fp16 on CUDA if bf16 is unsupported. |

## Epoch and Step Controls

| Parameter | Meaning |
| :--- | :--- |
| `epochs` | Number of epochs. |
| `start_epoch` | Starting epoch when resuming. |
| `global_step` | Starting optimizer step. |
| `grad_clip` | Max gradient norm. Use `None` to disable. |
| `grad_accum_steps` | Microbatches per optimizer step. |
| `max_batches_per_epoch` | Limit batches per epoch for smoke tests. |
| `log_every` | Print interval in optimizer steps. |
| `on_oom` | OOM behavior. `skip` skips CUDA OOM batches. |

## Module Diagnostics

| Parameter | Meaning |
| :--- | :--- |
| `module_metrics_every` | Computes module diagnostics every N optimizer steps. `0` means once after train epoch. |
| `print_module_diagnostics` | Prints diagnostics tables. |
| `verbose` | `1` for full diagnostics, `0` for top critical metrics. |
| `log_grad_norm` | Tracks gradient norm. |
| `log_mem` | Prints CUDA memory stats. |

## Evaluation

| Parameter | Meaning |
| :--- | :--- |
| `eval_every` | Evaluate every N epochs. |
| `eval_max_batches` | Limit validation batches. |
| `eval_use_ema` | Evaluate EMA weights if available. |
| `eval_log_every` | Optional eval logging interval. |
| `eval_preview` | Print qualitative preview. |
| `eval_preview_batch_idx` | Validation batch used for preview. |
| `eval_preview_sample_idx` | Sample inside preview batch. |
| `eval_preview_max_context_tokens` | Context tokens shown. |
| `eval_preview_max_new_tokens` | New tokens generated. |
| `eval_preview_temperature` | 0 for greedy generation, >0 for sampling. |
| `tokenizer` / `id2tok_fn` | Used to decode previews. |
