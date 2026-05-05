# Attention Config Reference

This file documents attention hyperparameters used by MHA, HCA, CSA, and hybrid attention.

## Shared Attention Parameters

| Parameter | Used by | Meaning |
| :--- | :--- | :--- |
| `d_model` | all | Hidden size entering attention. |
| `n_heads` | all | Number of query heads. More heads can model more attention patterns. |
| `head_dim` | all | Size per attention head. Current CSA/HCA require `n_heads * head_dim == d_model`. |
| `attention_dropout` | all | Dropout on attention probabilities. |
| `residual_dropout` | all | Dropout after attention output projection. |
| `use_bias` / `use_attention_bias` | all | Enables projection biases. |
| `use_rope` | all | Enables rotary positional embeddings. |
| `rope_theta` | all | RoPE base frequency. |
| `rotary_dim` | all | Number of head dimensions rotated by RoPE. |
| `max_seq_len` | all | Maximum sequence length for buffers/validation. |
| `init_std` | all | Weight initialization standard deviation. |

## HCA-Specific Parameters

| Parameter | Meaning |
| :--- | :--- |
| `compression_factor` / `hca_compression_factor` | Number of tokens compressed into one KV entry. Larger values reduce compute but discard detail. |
| `window_size` | Exact local context added beside compressed global memory. |
| `use_attention_sink` | Adds learnable sink entries that can absorb attention mass. |
| `use_grouped_output_projection` | Splits head outputs into groups before projection. |
| `output_projection_groups` | Number of groups; must divide `n_heads`. |

## CSA-Specific Parameters

| Parameter | Meaning |
| :--- | :--- |
| `compression_factor` | Number of tokens per compressed block. |
| `top_k` / `top_k_blocks` | Number of compressed blocks selected per query. |
| `window_size` | Exact local window appended to selected sparse global blocks. |
| `indexer_dim` | Size of indexer key/query vectors. |
| `n_indexer_heads` | Number of scoring heads in the sparse selector. |
| `query_compression_dim` | Low-rank query bottleneck before producing indexer/core queries. |
| `use_indexer_score_bias` | Lets indexer scores influence core attention logits. |
| `use_separate_local_kv` | Keeps local branch projection independent from compressed global branch. |
| `use_attention_sink` | Adds learnable global sink entries. |
| `use_grouped_output_projection` | Enables grouped projection after attention. |
| `output_projection_groups` | Number of output projection groups. |

## Hybrid Attention Parameters

| Parameter | Meaning |
| :--- | :--- |
| `attention_type="hybrid"` | Enables layer-wise attention schedule. |
| `attention_pattern` | Tuple/list cycled over layers, e.g. `[csa, hca]`. |

Example:

```yaml
attention_type: hybrid
attention_pattern: [csa, hca]
```
