# Muon Optimizer

Muon is one of the most important training components in this project.

Implementation file:

```text
training/muon_optimizer.py
```

## What It Is

Muon is an optimizer for 2D matrix parameters. It applies momentum and then orthogonalizes the update direction using Newton-Schulz iterations.

In this repo, Muon is used through a hybrid optimizer:

```text
HybridMuonAdamW = Muon branch + AdamW branch
```

## Role in DeepSeek-V4 Mini

The DeepSeek-V4 paper uses Muon for most matrix-like model parameters while keeping AdamW for parameters where Muon is not appropriate.

This mini repo mirrors that idea:

- Muon handles suitable 2D hidden matrices.
- AdamW handles embeddings, LM head, norms, biases, scalar/vector parameters, and sensitive small controls.

## Newton-Schulz Update

Function:

```python
zeropower_via_newtonschulz5(G, steps=5, eps=1e-7)
```

Role:

- receives a 2D gradient/update matrix,
- normalizes it,
- applies a quintic Newton-Schulz iteration,
- returns an approximately orthogonalized update.

Key hyperparameters:

- `steps`: number of Newton-Schulz iterations.
- `eps`: stability epsilon.

## Muon Hyperparameters

| Parameter | Meaning |
| :--- | :--- |
| `lr` / `muon_lr` | Muon learning rate. |
| `momentum` / `muon_momentum` | Momentum coefficient. |
| `weight_decay` / `muon_weight_decay` | Decoupled Muon branch weight decay. |
| `nesterov` / `muon_nesterov` | Uses Nesterov-style momentum. |
| `ns_steps` / `muon_ns_steps` | Newton-Schulz iterations. |
| `eps` / `muon_eps` | Numerical epsilon. |

## Hybrid Optimizer Builder

Function:

```python
build_muon_adamw_optimizer(...)
```

Important arguments:

| Argument | Meaning |
| :--- | :--- |
| `learning_rate` | AdamW LR and default Muon LR. |
| `muon_lr` | Optional separate Muon LR. |
| `weight_decay` | AdamW decay. |
| `muon_weight_decay` | Muon decay. |
| `betas` | AdamW beta values. |
| `eps` | AdamW epsilon. |
| `muon_momentum` | Muon momentum. |
| `muon_nesterov` | Muon Nesterov flag. |
| `muon_ns_steps` | Newton-Schulz iteration count. |
| `muon_eps` | Muon epsilon. |

## Parameter Grouping

Muon should receive:

- hidden Linear weights,
- attention projection matrices,
- FFN/MoE projection matrices,
- other suitable 2D internal transforms.

AdamW should receive:

- token embeddings,
- LM head,
- norm weights,
- biases,
- scalar/vector parameters,
- mHC static/gating parameters,
- small routing/control parameters.

## Practical Configs

Tiny debugging usually uses AdamW:

```yaml
optimizer_type: adamw
learning_rate: 0.0003
```

Muon research run:

```yaml
optimizer_type: muon_adamw
learning_rate: 0.0003
muon_lr: null
muon_momentum: 0.95
muon_nesterov: true
muon_ns_steps: 5
muon_eps: 0.0000001
muon_weight_decay: 0.0
```

## Failure Modes

- Muon only supports 2D tensors.
- Non-finite gradients raise errors.
- If no parameters are assigned to Muon, the builder raises because that usually means grouping logic or architecture assumptions are wrong.

## Why It Matters

Muon is not just another optimizer option here. It is part of the paper-faithful training story: a different update geometry for large matrix parameters, paired with AdamW for parameters that should not receive Muon updates.
