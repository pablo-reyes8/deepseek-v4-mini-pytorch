# mHC Config Reference

Primary config: `ManifoldHyperConnectionConfig`.

Model-level fields are prefixed with `mhc_` inside `DeepSeekV4LMConfig`.

| Model Field | mHC Field | Meaning |
| :--- | :--- | :--- |
| `use_mhc` | n/a | Enables mHC around attention and FFN. |
| `n_hc` | `n_hc` | Number of expanded residual streams. |
| `mhc_sinkhorn_iters` | `sinkhorn_iters` | Sinkhorn iterations used to constrain residual matrix B. |
| `mhc_eps` | `eps` | Numerical epsilon. |
| `mhc_dynamic` | `dynamic` | Enables input-dependent A/B/C generation. |
| `mhc_expand_mode` | n/a | Expansion mode from `[B,T,D]` to `[B,T,n_hc,D]`. |
| `mhc_collapse_mode` | n/a | Collapse mode back to `[B,T,D]`: `mean`, `first`, `sum`, `readout`. |
| `mhc_use_log_sinkhorn` | `use_log_sinkhorn` | Uses log-space Sinkhorn. |
| `mhc_sinkhorn_fp32` | `sinkhorn_fp32` | Computes Sinkhorn in fp32. |
| `mhc_init_alpha` | `init_alpha` | Initial dynamic contribution. |
| `mhc_alpha_max` | `alpha_max` | Maximum bounded dynamic alpha. |
| `mhc_bounded_alpha` | `bounded_alpha` | Bounds alpha with tanh. |

Lower-level static initialization fields:

| Field | Meaning |
| :--- | :--- |
| `static_a_stream0` | Initial A score for stream 0. |
| `static_a_other` | Initial A score for other streams. |
| `static_b_diag` | Initial B diagonal score. |
| `static_b_offdiag` | Initial B off-diagonal score. |
| `static_c_stream0` | Initial C score for stream 0. |
| `static_c_other` | Initial C score for other streams. |
| `init_std` | Dynamic generator initialization scale. |

Recommended tiny CPU start:

```yaml
use_mhc: true
n_hc: 2
mhc_sinkhorn_iters: 5
mhc_collapse_mode: readout
```

Paper-inspired mini start:

```yaml
use_mhc: true
n_hc: 4
mhc_sinkhorn_iters: 20
mhc_collapse_mode: readout
```
