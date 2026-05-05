# mHC Residual Streams

mHC means Manifold-Constrained Hyper-Connections.

## What It Is

mHC expands the residual stream from:

```text
[B, T, D]
```

to:

```text
[B, T, n_hc, D]
```

Then each sublayer update is controlled by three constrained mappings:

```text
X_next = B X + C F(A X)
```

Where:

- `A` mixes the expanded stream into the sublayer input.
- `B` mixes residual streams and is projected toward a doubly stochastic matrix.
- `C` injects the sublayer output back into the expanded stream.

## Role in the Model

- Replaces simple residual addition with a richer residual routing mechanism.
- Gives the model another axis of capacity without changing the inner sublayer hidden size.
- Improves the project's faithfulness to the DeepSeek-V4 paper.

## Main Hyperparameters

- `use_mhc`: enables mHC inside `DeepSeekV4Block`.
- `n_hc`: number of residual streams. Must be at least 2 when enabled.
- `mhc_sinkhorn_iters`: number of Sinkhorn normalization iterations for `B`.
- `mhc_eps`: numerical epsilon used in normalization.
- `mhc_dynamic`: enables input-dependent dynamic generation of A/B/C.
- `mhc_expand_mode`: how `[B,T,D]` is expanded into `[B,T,n_hc,D]`.
- `mhc_collapse_mode`: how expanded streams are collapsed back. Supported values include `mean`, `first`, `sum`, and `readout`.
- `mhc_use_log_sinkhorn`: use log-space Sinkhorn variant.
- `mhc_sinkhorn_fp32`: force Sinkhorn computation to fp32 for stability.
- `mhc_init_alpha`: initial strength of dynamic A/B/C components.
- `mhc_alpha_max`: cap for bounded dynamic gates.
- `mhc_bounded_alpha`: constrains dynamic alpha gates with tanh.

Lower-level initialization controls:

- `static_a_stream0`: initial preference for stream 0 in A.
- `static_a_other`: initial score for other streams in A.
- `static_b_diag`: initial diagonal score for B.
- `static_b_offdiag`: initial off-diagonal score for B.
- `static_c_stream0`: initial injection into stream 0.
- `static_c_other`: initial injection score for other streams.
- `init_std`: initialization scale for dynamic generators.

Practical notes:

- `n_hc=2` is good for CPU tests.
- `n_hc=4` matches the mini paper-inspired default.
- More Sinkhorn iterations make `B` closer to doubly stochastic but cost more compute.
- mHC is more sensitive than classic residuals; keep tiny configs small when debugging.
