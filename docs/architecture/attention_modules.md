# Attention Modules

DeepSeek-V4 Mini supports four attention modes through `attention_type`:

- `mha`: standard causal multi-head attention.
- `hca`: Heavily Compressed Attention.
- `csa`: Compressed Sparse Attention.
- `hybrid`: layer-wise schedule over `attention_pattern`, usually `["csa", "hca"]`.

This file is the high-level attention index. The two core DeepSeek-V4-inspired modules have dedicated pages:

- [HCA: Heavily Compressed Attention](hca.md)
- [CSA: Compressed Sparse Attention](csa.md)

## Standard MHA

What it is:

- A baseline causal multi-head attention module with optional RoPE.

Role:

- Provides a dense-attention reference path.
- Useful for sanity checks, tiny CPU training, and ablations.
- Gives CSA/HCA a known-good baseline for shape, causality, gradients, and loss behavior.

Key hyperparameters:

- `n_heads`: number of query/key/value attention heads.
- `head_dim`: dimensionality per head.
- `use_rope`: enables rotary positional embeddings.
- `rotary_dim`: number of head dimensions receiving RoPE.
- `attention_dropout`: dropout on attention weights.
- `residual_dropout`: dropout after output projection.
- `max_seq_len`: maximum sequence length for positional buffers and validation.

When to use:

- First smoke tests.
- Small CPU training.
- Comparing dense attention against compressed/sparse variants.

## HCA Summary

HCA compresses token-level KV entries aggressively, then performs dense MQA-style attention over the compressed memory plus exact local window tokens.

Use HCA when:

- You want strong compression.
- Long context matters more than precise global token-level memory.
- You want a cheaper global attention branch.

See [HCA](hca.md) for internals and hyperparameters.

## CSA Summary

CSA compresses KV blocks, scores compressed blocks with a lightweight indexer, selects top-k blocks per query, and combines those sparse global blocks with exact local window tokens.

Use CSA when:

- Retrieval quality matters.
- The model needs selective long-range access.
- You are testing synthetic key-value retrieval or long-context behavior.

See [CSA](csa.md) for internals and hyperparameters.

## Hybrid Attention

What it is:

- A cyclic schedule over attention modules.

Role:

- Mimics the paper's interleaving of CSA and HCA.
- Lets some layers focus on sparse retrieval and others on heavily compressed global context.

Key hyperparameters:

- `attention_type="hybrid"`.
- `attention_pattern`: tuple/list such as `("csa", "hca")`.

Example:

```yaml
attention_type: hybrid
attention_pattern: [csa, hca]
```
