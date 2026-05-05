# HCA: Heavily Compressed Attention

HCA is one of the two core long-context attention modules in DeepSeek-V4 Mini.

## What It Is

Heavily Compressed Attention compresses groups of token-level KV entries into a much shorter global KV memory. Every query attends to:

1. fully completed compressed global KV blocks,
2. exact local sliding-window KV tokens,
3. optional attention sink entries.

The goal is to make global context cheap while preserving recent local detail.

## Role in DeepSeek-V4 Mini

HCA is the aggressive compression branch.

It is useful when:

- the sequence is long,
- global context can be represented coarsely,
- local syntax/detail is handled by the sliding window,
- you want lower memory and lower attention cost than dense MHA.

In `hybrid` mode, HCA layers complement CSA layers: HCA gives broad compressed context, CSA gives selective sparse retrieval.

## Forward Path

Input:

```text
x: [B, T, d_model]
```

Internal steps:

```text
x
  -> q_proj                 -> Q: [B, T, n_heads, head_dim]
  -> kv_proj                -> C: [B, T, head_dim]
  -> z_proj                 -> Z: [B, T, head_dim]

(C, Z)
  -> HCATokenCompressor     -> compressed_C: [B, S, head_dim]

Q attends to:
  -> compressed_C blocks that are causally complete
  -> exact local KV tokens inside window_size
  -> optional sink KV

attention output
  -> grouped or dense output projection
  -> residual dropout
  -> [B, T, d_model]
```

Where:

```text
S = ceil(T / compression_factor)
```

## Causal Rule

For compressed global attention, query token `t` can attend only to compressed block `s` if the block is fully completed before the query's current compression block:

```text
allowed[t, s] = s < floor(t / compression_factor)
```

The current block is not accessed through compressed memory because it may contain future tokens. Current/recent information comes from the local sliding-window branch.

## Local Window Rule

The local branch uses exact token-level KV entries:

```text
allowed[t, s] = s <= t and t - s < window_size
```

This is the mechanism that keeps short-range details precise.

## Main Hyperparameters

| Parameter | Role |
| :--- | :--- |
| `d_model` | Hidden width entering and leaving HCA. |
| `n_heads` | Number of query heads. More heads allow more query subspaces. |
| `head_dim` | Width of each query head and shared KV vector. Current implementation requires `n_heads * head_dim == d_model`. |
| `compression_factor` / `hca_compression_factor` | Number of tokens compressed into one global KV entry. Larger means cheaper global memory and more information loss. |
| `window_size` | Number of exact recent tokens visible to each query. Larger preserves more local detail. |
| `attention_dropout` | Dropout on attention weights. |
| `residual_dropout` | Dropout after HCA output projection. |
| `use_bias` / `use_attention_bias` | Enables projection biases. |
| `use_rope` | Applies rotary positional embeddings to queries. |
| `rope_theta` | RoPE frequency base. |
| `rotary_dim` | Number of per-head dimensions using RoPE. |
| `max_seq_len` | Maximum sequence length accepted by validation/buffers. |
| `init_std` | Initialization scale. |
| `use_attention_sink` | Adds learnable global sink key/value entries. |
| `use_grouped_output_projection` | Projects attention outputs through grouped projections. |
| `output_projection_groups` | Number of groups. Must divide `n_heads`. |

## How Hyperparameters Change Behavior

### `compression_factor`

Higher values:

- reduce compressed sequence length,
- reduce global attention memory,
- make each compressed entry summarize more tokens,
- can hurt fine retrieval if too large.

Tiny debug:

```yaml
hca_compression_factor: 4
```

Long-context-oriented:

```yaml
hca_compression_factor: 16
```

### `window_size`

Higher values:

- improve local precision,
- increase local attention cost,
- reduce artifacts from aggressive compression.

Tiny debug:

```yaml
window_size: 4
```

Mini research:

```yaml
window_size: 32
```

### `n_heads` and `head_dim`

These control attention capacity.

In this implementation:

```text
n_heads * head_dim == d_model
```

Example:

```yaml
d_model: 256
n_heads: 4
head_dim: 64
```

### `use_attention_sink`

Attention sinks are learnable KV entries. They provide a safe place for attention mass and can stabilize attention patterns when available real keys are weak or masked.

### `use_grouped_output_projection`

Grouped output projection keeps the paper-inspired grouped projection idea without custom kernels.

## Recommended Configs

CPU smoke:

```yaml
attention_type: hca
d_model: 32
n_heads: 4
head_dim: 8
hca_compression_factor: 4
window_size: 4
rotary_dim: 8
```

Mini research:

```yaml
attention_type: hca
d_model: 256
n_heads: 4
head_dim: 64
hca_compression_factor: 16
window_size: 32
rotary_dim: 64
use_attention_sink: true
use_grouped_output_projection: true
```

## Tests That Protect HCA

Relevant tests:

- `tests/test_hca.py`
- `tests/test_deepseek_model.py`

Behavior covered:

- output shape,
- compressed length,
- causal masking,
- local window masking,
- finite gradients,
- attention weights,
- grouped output projection,
- integration inside a block/model.
