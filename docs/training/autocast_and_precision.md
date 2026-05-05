# Autocast and Precision

Precision helpers live in `training/autocast.py`.

## Role

This module centralizes device and AMP behavior so the training loop can stay simple.

It handles:

- selecting `cpu`, `cuda`, or `mps`,
- resolving requested AMP dtype,
- falling back from bf16 to fp16 on CUDA when needed,
- deciding whether a grad scaler is required,
- moving nested batches to device,
- exposing a safe `autocast_ctx`.

## Device Resolution

Function: `resolve_device`.

Accepted values:

- `auto`: prefer CUDA, then MPS, then CPU.
- `cpu`
- `cuda`
- `mps`
- explicit `torch.device`.

If CUDA or MPS is requested but unavailable, it raises instead of silently falling back.

## AMP Dtypes

Supported string values:

```text
bf16, bfloat16
fp16, float16
fp32, float32
none
```

Key functions:

- `resolve_amp_dtype`: maps string to torch dtype.
- `get_effective_amp_dtype`: checks whether requested dtype is actually usable.
- `cuda_supports_bf16`: detects CUDA bf16 support.
- `should_use_grad_scaler`: grad scaler only for CUDA fp16.
- `make_grad_scaler`: creates scaler when required.

## `setup_device_and_precision`

Returns a dictionary used by `train_one_epoch` and `eval_one_epoch`:

```python
{
    "device": resolved_device,
    "device_type": "cpu" | "cuda" | "mps",
    "amp_enabled": bool,
    "amp_dtype_requested": str,
    "amp_dtype_effective": torch.dtype | None,
    "use_grad_scaler": bool,
    "scaler": scaler_or_none,
    "cache_enabled": bool,
    "fallback_bf16_to_fp16": bool,
}
```

## Practical Defaults

CPU tests:

```python
setup_device_and_precision(device="cpu", amp_enabled=False)
```

CUDA bf16 training:

```python
setup_device_and_precision(device="cuda", amp_enabled=True, amp_dtype="bf16")
```

CUDA fp16 training:

```python
setup_device_and_precision(device="cuda", amp_enabled=True, amp_dtype="fp16")
```

## Why This Matters

CSA, HCA, mHC, and MoE have more numerically sensitive paths than a minimal Transformer. Centralizing precision behavior makes it easier to:

- disable AMP for CPU smoke tests,
- use bf16 when available,
- avoid unnecessary grad scaling,
- keep batch movement consistent,
- prevent context-manager bugs when model forward raises.
