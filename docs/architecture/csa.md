# CSA: Compressed Sparse Attention

CSA is the most important long-context attention module in this repository.

## What It Is

Compressed Sparse Attention combines:

1. KV compression,
2. lightweight indexer scoring,
3. top-k sparse block selection,
4. MQA-style attention over selected compressed blocks,
5. exact local sliding-window attention.

It is designed to keep targeted long-range retrieval while reducing attention cost.

## Role in DeepSeek-V4 Mini

CSA is the selective retrieval branch.

It is useful when:

- long-range facts matter,
- the query should attend to a few relevant global regions,
- dense attention is too expensive,
- HCA compression is too coarse by itself.

CSA is especially important for synthetic retrieval experiments because the model must recover specific key-value facts from distractor context.

## Forward Path

Input:

```text
x: [B, T, d_model]
```

High-level flow:

```text
x
  -> q_down_proj
  -> q_up_proj                    -> Q for core attention
  -> index_q_up_proj              -> indexer queries
  -> index_weight_proj            -> indexer head weights

x
  -> a_kv_proj, b_kv_proj         -> token-level KV values
  -> a_z_proj, b_z_proj           -> compression logits
  -> CSAOverlappedCompressor      -> compressed KV entries

x
  -> a_index_kv_proj, b_index_kv_proj
  -> a_index_z_proj, b_index_z_proj
  -> CSAOverlappedCompressor      -> compressed indexer keys

indexer queries + compressed indexer keys
  -> CSALightningIndexer
  -> top-k causal block indices

Q attends to:
  -> selected compressed KV entries
  -> exact local KV tokens
  -> optional sink KV

output
  -> grouped or dense output projection
  -> residual dropout
```

## Overlapped A/B Compression

CSA uses two KV branches:

- `a_*` branch for the current compression block,
- `b_*` branch for an overlapped previous block contribution.

This approximates the paper's overlapped compression idea while staying readable in PyTorch.

Role:

- keeps compressed entries from being too block-boundary brittle,
- gives each compressed block a richer local neighborhood,
- still reduces sequence length by roughly `compression_factor`.

## Lightning Indexer

The indexer decides which compressed blocks each query should see.

Inputs:

- compressed indexer keys,
- low-rank indexer queries,
- per-indexer-head weights.

Output:

- top-k compressed block indices per query.

Important property:

- selection is causal; future blocks are masked out.

## Causal Rule

CSA must avoid selecting compressed blocks that include future information.

For query token `t`, a compressed block is valid only if it is strictly before the query's current compression block:

```text
block_idx < floor(t / compression_factor)
```

Current/recent tokens are handled by the exact local window branch.

## Local Window Branch

The local branch gives exact access to recent tokens:

```text
allowed[t, s] = s <= t and t - s < window_size
```

This is critical because compression can lose token-level detail.

## Main Hyperparameters

| Parameter | Role |
| :--- | :--- |
| `d_model` | Hidden width entering and leaving CSA. |
| `n_heads` | Number of query heads for core attention. |
| `head_dim` | Width per query head and compressed KV entry. Requires `n_heads * head_dim == d_model`. |
| `compression_factor` | Number of tokens per compressed block. |
| `top_k` / `top_k_blocks` | Number of compressed blocks selected per query. |
| `window_size` | Exact local context length. |
| `indexer_dim` | Width of compressed indexer keys and queries. |
| `n_indexer_heads` | Number of indexer scoring heads. |
| `query_compression_dim` | Low-rank query bottleneck shared by core and indexer queries. |
| `attention_dropout` | Dropout on attention weights. |
| `residual_dropout` | Dropout after output projection. |
| `use_bias` / `use_attention_bias` | Enables projection biases. |
| `use_rope` | Applies rotary positional embeddings. |
| `rope_theta` | RoPE frequency base. |
| `rotary_dim` | Number of per-head dimensions using RoPE. |
| `max_seq_len` | Maximum sequence length accepted by validation/buffers. |
| `init_std` | Initialization scale. |
| `use_attention_sink` | Adds learnable sink KV entries. |
| `use_grouped_output_projection` | Enables grouped output projection. |
| `output_projection_groups` | Number of projection groups. Must divide `n_heads`. |
| `use_indexer_score_bias` | Optionally injects indexer scores into core attention logits. |
| `use_separate_local_kv` | Uses a separate projection for exact local KV tokens. |

## How Hyperparameters Change Behavior

### `compression_factor`

Higher values:

- shorten compressed memory,
- reduce attention/indexer cost,
- make each compressed entry represent more tokens,
- can weaken precise retrieval.

Tiny debug:

```yaml
compression_factor: 4
```

### `top_k_blocks`

Higher values:

- let each query inspect more global blocks,
- improve recall in long-context retrieval,
- increase attention cost.

Tiny debug:

```yaml
top_k_blocks: 2
```

Mini research:

```yaml
top_k_blocks: 8
```

### `indexer_dim`

Higher values:

- make sparse selection more expressive,
- increase indexer projection and scoring cost.

### `n_indexer_heads`

More indexer heads:

- allow multiple scoring perspectives,
- can improve block selection,
- add cost to the indexer path.

### `query_compression_dim`

This is the low-rank bottleneck for query production.

Smaller values:

- cheaper,
- less expressive.

Larger values:

- richer query representation,
- more parameters and compute.

### `window_size`

Higher values:

- preserve more exact recent context,
- cost more local attention compute.

## Recommended Configs

CPU smoke:

```yaml
attention_type: csa
d_model: 32
n_heads: 4
head_dim: 8
compression_factor: 4
top_k_blocks: 2
window_size: 4
indexer_dim: 8
n_indexer_heads: 2
query_compression_dim: 8
rotary_dim: 8
```

Mini research:

```yaml
attention_type: csa
d_model: 256
n_heads: 4
head_dim: 64
compression_factor: 4
top_k_blocks: 8
window_size: 32
indexer_dim: 64
n_indexer_heads: 4
query_compression_dim: 64
use_attention_sink: true
use_grouped_output_projection: true
use_separate_local_kv: true
```

## Tests That Protect CSA

Relevant tests:

- `tests/test_csa.py`
- `tests/test_deepseek_model.py`

Behavior covered:

- output shape,
- top-k future masking,
- no future leakage,
- finite gradients,
- local window behavior,
- compressed KV length reduction,
- model-level integration.
