# MoE and Dense FFN

The feed-forward branch of each `DeepSeekV4Block` is selected with `ffn_type`.

Supported modes:

- `dense`: standard SwiGLU MLP.
- `moe`: DeepSeek-style routed/shared experts.

## Dense SwiGLU FFN

What it is:

- A standard gated MLP using SwiGLU-style activation.

Role:

- Baseline FFN for tiny runs and ablations.
- Useful when MoE routing is not the focus.

Key hyperparameters:

- `mlp_hidden_dim`: explicit hidden width. If omitted, width is derived from `mlp_expansion_factor`.
- `mlp_expansion_factor`: multiplier from `d_model` to hidden dimension.
- `mlp_multiple_of`: rounds hidden dimension to a multiple for cleaner shapes.
- `mlp_dropout`: dropout inside the MLP.
- `use_mlp_bias`: enables linear biases.

## DeepSeekMoE

What it is:

- A mini DeepSeek-style mixture-of-experts FFN with routed experts, shared experts, learned or hash routing, and balance diagnostics.

Role:

- Adds sparse conditional capacity.
- Only activates `top_k_experts` routed experts per token.
- Shared experts provide always-on capacity.

Main hyperparameters:

- `num_experts`: total number of routed experts.
- `top_k_experts`: number of routed experts selected per token.
- `expert_hidden_dim`: hidden width inside each routed expert.
- `expert_expansion_factor`: derives routed expert width from `d_model` when `expert_hidden_dim` is not set.
- `expert_multiple_of`: rounds expert hidden width.
- `shared_experts`: number of shared experts always evaluated.
- `shared_hidden_dim`: hidden width for shared experts.
- `shared_expansion_factor`: derives shared expert width when explicit width is absent.
- `router_type`: `learned` or `hash`.
- `router_score_fn`: scoring nonlinearity; supported values are `softmax`, `sigmoid`, `sqrt_softplus`.
- `normalize_topk_weights`: normalizes selected expert weights to sum to one.
- `topk_weight_scale`: multiplier applied to selected expert weights.
- `router_jitter_noise`: random noise added to router logits during training.
- `hash_routing_stride`: stride used by deterministic hash routing.
- `routed_scale`: scales routed expert output.
- `shared_scale`: scales shared expert output.
- `balance_loss_weight`: weight for global expert load balancing loss.
- `sequence_balance_loss_weight`: weight for sequence-wise balance loss.
- `dropout`, `use_bias`, `init_std`, `eps`: regularization, initialization, and numerical controls.

Practical notes:

- Use `router_type="learned"` for normal experiments.
- Use `router_type="hash"` for deterministic early-layer routing experiments.
- Keep `num_experts` small on CPU, such as 4 or 8.
- `top_k_experts=2` is a good default for mini models.
- Balance losses are diagnostics/training helpers, not the full industrial auxiliary-loss-free routing system from the paper.
