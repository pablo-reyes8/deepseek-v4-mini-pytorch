# Logging, Evaluation, and Checkpointing

## Logging

| Parameter | Meaning |
| :--- | :--- |
| `log_every` | Print training row every N optimizer steps. |
| `log_grad_norm` | Include gradient norm in training stats. |
| `log_mem` | Print CUDA memory stats. |
| `module_metrics_every` | Collect DeepSeek module diagnostics every N steps. |
| `print_module_diagnostics` | Print module diagnostics to console. |
| `verbose` | Controls diagnostics detail level. |
| `metrics_jsonl_name` | File name for JSONL metrics inside checkpoint dir. |

## Evaluation Preview

| Parameter | Meaning |
| :--- | :--- |
| `eval_preview` | Enables qualitative preview. |
| `eval_preview_batch_idx` | Batch index used for preview. |
| `eval_preview_sample_idx` | Sample index inside the batch. |
| `eval_preview_max_context_tokens` | Number of input/context tokens shown. |
| `eval_preview_max_new_tokens` | Number of autoregressive tokens generated. |
| `eval_preview_temperature` | 0 means greedy; >0 enables sampling. |
| `tokenizer` | Optional tokenizer with `.decode`. |
| `id2tok_fn` | Optional custom id-to-text function. |

## Checkpointing

| Parameter | Meaning |
| :--- | :--- |
| `ckpt_dir` | Directory where checkpoints and metrics are written. |
| `run_name` | Human-readable run name. |
| `save_every` | Save every N epochs. |
| `save_last` | Save/update latest checkpoint. |
| `keep_last_n_checkpoints` | Delete older step checkpoints beyond N. |
| `monitor_name` | Metric used for best checkpoint. |
| `monitor_mode` | `min` or `max`. |
| `best_metric` | Starting best metric when resuming. |
| `resume_path` | Checkpoint path to resume from. |
| `strict_resume` | Strict model state loading. |
| `restore_rng_state` | Restores Python/NumPy/PyTorch RNG states. |

## Drive Mirroring

These are useful for notebook/Colab workflows:

| Parameter | Meaning |
| :--- | :--- |
| `drive_ckpt_dir` | Optional external checkpoint mirror directory. |
| `copy_fixed_to_drive` | Copies latest checkpoint to fixed filename. |
| `fixed_drive_name` | Fixed checkpoint file name. |
| `fixed_drive_metrics_name` | Fixed metrics file name. |
