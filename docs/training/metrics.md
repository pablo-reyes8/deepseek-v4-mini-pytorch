# Metrics and Diagnostics

The training stack has two levels of metrics:

1. standard LM training/evaluation metrics,
2. DeepSeek-module diagnostics for MoE, mHC, MTP, CSA/HCA, and loss health.

## LM Metrics

Implemented mainly in:

```text
training/training_metrics.py
training/loss_metrics.py
```

Core metrics:

- loss,
- perplexity,
- token accuracy,
- top-k accuracy,
- entropy,
- valid token count.

Role:

- tracks normal language modeling progress,
- works on train/eval outputs,
- supports ignored labels and pad masking.

## Module Diagnostics

Implemented across:

```text
training/full_deepseek_metrics.py
training/moe_metrics.py
training/mhc_metrics.py
training/mtp_metrics.py
training/deepseek_modules_metrics_utils.py
```

Role:

- exposes internal behavior that is invisible in scalar LM loss,
- helps diagnose routing collapse, unstable mHC matrices, MTP behavior, and auxiliary losses,
- supports compact top-critical views or more verbose diagnostics.

## MoE Diagnostics

Typical signals:

- router entropy,
- expert load distribution,
- selected expert fractions,
- active experts,
- balance losses,
- sequence-wise imbalance.

Why they matter:

- MoE can silently collapse into a few experts.
- Loss may improve while routing quality worsens.
- Expert balance matters for sparse capacity.

## mHC Diagnostics

Typical signals:

- A/B/C matrix statistics,
- Sinkhorn row/column sums,
- alpha gate values,
- stream mixing behavior.

Why they matter:

- mHC is numerically sensitive.
- B should behave like a doubly stochastic residual mixing matrix.
- Alpha gates should not explode early in training.

## MTP Diagnostics

Typical signals:

- raw MTP loss,
- weighted MTP loss,
- per-depth losses,
- depth weights.

Why they matter:

- MTP can help representation quality but can also dominate training if overweighted.

## Logging Controls

| Parameter | Meaning |
| :--- | :--- |
| `module_metrics_every` | Collect diagnostics every N optimizer steps. |
| `print_module_diagnostics` | Print diagnostics to console. |
| `verbose` | `1` full diagnostics, `0` compact critical diagnostics. |
| `log_grad_norm` | Include gradient norm in train stats. |
| `log_mem` | Print CUDA memory stats. |
| `metrics_jsonl_name` | JSONL metrics file name. |

## Practical Advice

- Keep diagnostics off or sparse for fastest CPU smoke tests.
- Turn diagnostics on when testing MoE/mHC/CSA changes.
- Use `verbose=0` when you only want high-risk signals.
- Use JSONL metrics for plotting experiments later.
