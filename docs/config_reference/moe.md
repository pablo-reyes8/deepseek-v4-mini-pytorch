# MoE and FFN Config Reference

## Dense FFN Parameters

| Parameter | Meaning |
| :--- | :--- |
| `ffn_type="dense"` | Selects dense SwiGLU FFN. |
| `mlp_hidden_dim` | Explicit hidden width. |
| `mlp_expansion_factor` | Hidden width multiplier from `d_model` when `mlp_hidden_dim` is absent. |
| `mlp_multiple_of` | Rounds hidden width to a multiple. |
| `mlp_dropout` | Dropout inside the MLP. |
| `use_mlp_bias` | Enables linear biases. |

## MoE Parameters

| Parameter | Meaning |
| :--- | :--- |
| `ffn_type="moe"` | Selects DeepSeekMoE FFN. |
| `num_experts` | Number of routed experts. |
| `top_k_experts` / `top_k` | Number of routed experts activated per token. |
| `expert_hidden_dim` | Explicit hidden width for routed experts. |
| `expert_expansion_factor` | Routed expert hidden multiplier from `d_model`. |
| `expert_multiple_of` | Rounds expert hidden width. |
| `shared_experts` | Number of always-on shared experts. |
| `shared_hidden_dim` | Explicit hidden width for shared experts. |
| `shared_expansion_factor` | Shared expert hidden multiplier from `d_model`. |
| `router_type` | `learned` or `hash`. |
| `router_score_fn` | `softmax`, `sigmoid`, or `sqrt_softplus`. |
| `normalize_topk_weights` | Normalizes selected expert weights. |
| `topk_weight_scale` | Multiplies selected expert weights. |
| `router_jitter_noise` | Adds training-time noise to routing logits. |
| `hash_routing_stride` | Stride used by deterministic hash routing. |
| `routed_scale` | Output scale for routed experts. |
| `shared_scale` | Output scale for shared experts. |
| `balance_loss_weight` | Global balance auxiliary loss weight. |
| `sequence_balance_loss_weight` | Sequence-wise balance auxiliary loss weight. |
| `dropout` / `mlp_dropout` | Expert dropout. |
| `use_bias` / `use_mlp_bias` | Enables projection biases. |
| `init_std` | Initialization scale. |
| `eps` | Numerical epsilon for routing normalization. |

Recommended tiny CPU start:

```yaml
ffn_type: moe
num_experts: 4
top_k_experts: 2
expert_hidden_dim: 64
shared_experts: 1
shared_hidden_dim: 64
```
