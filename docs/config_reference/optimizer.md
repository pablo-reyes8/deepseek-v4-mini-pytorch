# Optimizer and Scheduler Config Reference

The training stack supports:

- `adamw`
- `muon_adamw`

## AdamW Parameters

| Parameter | Meaning |
| :--- | :--- |
| `learning_rate` | Base AdamW learning rate. |
| `min_learning_rate` | Final/minimum LR for cosine schedule. |
| `weight_decay` | Decoupled AdamW weight decay for decay group. |
| `betas` | AdamW beta coefficients. |
| `eps` | AdamW epsilon. |

Parameter grouping:

- Decay group: matrix-like trainable weights.
- No-decay group: biases, norms, embeddings, LM head, scalar/vector params, mHC gates/static params, and small control parameters.

## Muon + AdamW Parameters

| Parameter | Meaning |
| :--- | :--- |
| `optimizer_type="muon_adamw"` | Enables hybrid optimizer. |
| `muon_lr` | Optional separate LR for Muon parameters. Defaults to `learning_rate`. |
| `muon_momentum` | Muon momentum coefficient. |
| `muon_nesterov` | Uses Nesterov-style momentum update. |
| `muon_ns_steps` | Newton-Schulz orthogonalization iterations. |
| `muon_eps` | Muon numerical epsilon. |
| `muon_weight_decay` | Decoupled weight decay for Muon group. |
| `min_muon_lr` | Optional separate min LR for Muon schedule. |

Muon is intended for 2D matrix parameters, not embeddings, LM heads, norms, biases, scalar/vector params, or mHC small parameters.

## Warmup Cosine Scheduler

Primary class: `WarmupCosineLR`.

| Parameter | Meaning |
| :--- | :--- |
| `total_steps` | Total optimizer steps for schedule. |
| `warmup_steps` | Linear warmup steps. |
| `min_lr` / `min_learning_rate` | Floor LR after cosine decay. |
| `min_muon_lr` | Optional Muon-specific LR floor. |

Behavior:

```text
step <= warmup_steps:
    lr increases linearly to base LR

step > warmup_steps:
    lr follows cosine decay to min LR
```

Recommended tiny start:

```yaml
optimizer_type: adamw
learning_rate: 0.0003
min_learning_rate: 0.00003
weight_decay: 0.1
warmup_steps: 1
```
