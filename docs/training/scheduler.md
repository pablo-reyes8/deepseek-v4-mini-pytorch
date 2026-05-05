# Scheduler

Primary class: `WarmupCosineLR`.

## Role

The scheduler controls learning rate at optimizer-step granularity.

It supports:

- normal PyTorch optimizers,
- hybrid `Muon + AdamW` optimizer,
- separate AdamW and Muon base LRs,
- optional separate Muon minimum LR,
- checkpoint/resume through `state_dict`.

## Schedule Shape

```text
warmup:
    linearly increase from 0 to base_lr

after warmup:
    cosine decay from base_lr to min_lr
```

Formula after warmup:

```text
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

## Main Hyperparameters

| Parameter | Meaning |
| :--- | :--- |
| `total_steps` | Total optimizer steps expected for the run. |
| `warmup_steps` | Number of linear warmup steps. |
| `min_lr` | Final/minimum LR for normal optimizer or AdamW branch. |
| `min_muon_lr` | Optional final/minimum LR for Muon branch. |

## Runtime Methods

| Method | Meaning |
| :--- | :--- |
| `step()` | Advances one scheduler step and updates optimizer LRs. |
| `set_step(step)` | Sets scheduler to an explicit step, useful for resume. |
| `get_last_lr()` | Returns current LR list. |
| `get_lr_dict()` | Logging-friendly LR dictionary. |
| `state_dict()` | Serializable state. |
| `load_state_dict(state)` | Restores state and LR values. |

## Recommended Settings

Tiny CPU smoke:

```yaml
total_steps: 2
warmup_steps: 1
min_learning_rate: 0.00003
```

Mini training:

```yaml
warmup_steps: 500
learning_rate: 0.0003
min_learning_rate: 0.00003
```

## Notes

- Call order should be `optimizer.step()` then `scheduler.step()`.
- `total_steps` should count optimizer steps, not raw batches.
- With gradient accumulation, optimizer steps are fewer than dataloader batches.
