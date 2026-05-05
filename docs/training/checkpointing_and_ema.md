# Checkpointing and EMA

Checkpointing and EMA are part of the training infrastructure rather than the architecture, but they matter for reproducible experiments.

## Checkpointing

Implementation:

```text
training/chekpoints.py
```

Yes, the file is currently named `chekpoints.py`.

## What a Checkpoint Can Store

- model state,
- optimizer state,
- scheduler state,
- grad scaler state,
- EMA state,
- epoch,
- step,
- best metric,
- config snapshot,
- extra metadata,
- RNG state.

## Key Functions

| Function | Role |
| :--- | :--- |
| `save_checkpoint` | Writes a checkpoint and sidecar metadata. |
| `load_checkpoint` | Restores model/training state. |
| `cleanup_old_checkpoints` | Keeps only the last N step checkpoints. |

## Important Hyperparameters

| Parameter | Meaning |
| :--- | :--- |
| `ckpt_dir` | Checkpoint directory. |
| `run_name` | Name stored in metadata. |
| `save_every` | Save every N epochs. |
| `save_last` | Maintain latest checkpoint. |
| `keep_last_n_checkpoints` | Delete older step checkpoints. |
| `monitor_name` | Metric used to choose best checkpoint. |
| `monitor_mode` | `min` or `max`. |
| `resume_path` | Checkpoint path to resume. |
| `strict_resume` | Strict model state loading. |
| `restore_rng_state` | Restore RNG states for better reproducibility. |

## Atomic Save

Checkpoint writes use a temporary path and then rename into place. This reduces the chance of corrupting the final checkpoint if the process stops mid-save.

## EMA

Implementation:

```text
training/ema.py
```

EMA means exponential moving average of model weights.

Role:

- keeps smoothed model weights,
- can improve evaluation stability,
- can be evaluated separately with `eval_use_ema`.

## EMA Hyperparameters

| Parameter | Meaning |
| :--- | :--- |
| `use_ema` | Enables EMA tracking. |
| `ema_decay` | Decay coefficient. Higher means slower updates. |
| `ema_device` | Device for EMA shadow weights, often `cpu`. |
| `ema_update_after_step` | Delay before EMA begins. |
| `ema_update_every` | Update interval in optimizer steps. |
| `eval_use_ema` | Use EMA weights during evaluation. |

## Practical Advice

- For tiny smoke tests, keep EMA off.
- For longer runs, enable EMA after the training loop is stable.
- Store EMA on CPU if GPU memory is tight.
