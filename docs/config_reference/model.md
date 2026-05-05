# Model Config Reference

Primary class: `DeepSeekV4LMConfig`.

This config controls the full model: embeddings, block count, attention selection, FFN/MoE, mHC, MTP, loss behavior, and initialization.

## Core Shape

| Parameter | Meaning |
| :--- | :--- |
| `vocab_size` | Number of tokenizer vocabulary entries. Must be > 0. |
| `d_model` | Hidden size of the model. All block inputs/outputs use this width. |
| `n_layers` | Number of `DeepSeekV4Block` layers. |
| `max_seq_len` | Maximum sequence length accepted by the model. |
| `pad_token_id` | Optional tokenizer pad id. Used for attention masks and loss masking. |
| `ignore_index` | Label value ignored by cross entropy. Default `-100`. |

## Loss Semantics

| Parameter | Meaning |
| :--- | :--- |
| `labels_are_shifted` | If true, labels already represent next-token targets. If false, labels are shifted internally. |
| `ignore_pad_token_in_loss` | Masks pad tokens in loss when `pad_token_id` is set. |

## Embedding and Norm

| Parameter | Meaning |
| :--- | :--- |
| `embedding_dropout` | Dropout after token embedding. |
| `scale_embeddings` | Scales embeddings by hidden-size convention when enabled. |
| `tie_word_embeddings` | Shares LM head weight with token embedding weight. |
| `rms_norm_eps` | Epsilon for RMSNorm layers. |
| `init_std` | Default normal initialization standard deviation. |

## Attention Selection

| Parameter | Meaning |
| :--- | :--- |
| `attention_type` | One of `mha`, `hca`, `csa`, `hybrid`. |
| `attention_pattern` | Layer cycle used when `attention_type="hybrid"`. |
| `n_heads` | Number of attention query heads. |
| `head_dim` | Dimension per head. If absent, derived from `d_model / n_heads`. |
| `attention_dropout` | Dropout on attention weights. |
| `residual_dropout` | Dropout on attention/FFN outputs before residual integration. |
| `use_attention_bias` | Enables biases in attention projections. |
| `use_rope` | Enables rotary positional embeddings. |
| `rope_theta` | RoPE frequency base. |
| `rotary_dim` | Number of head dimensions using RoPE. Must be even and <= `head_dim`. |

## Shared HCA/CSA Controls

| Parameter | Meaning |
| :--- | :--- |
| `compression_factor` | CSA compression block size. |
| `hca_compression_factor` | HCA compression block size. |
| `window_size` | Exact local sliding-window context length. |

## CSA Controls

| Parameter | Meaning |
| :--- | :--- |
| `top_k_blocks` | Number of compressed blocks selected by CSA indexer. |
| `indexer_dim` | Size of compressed indexer key/query vectors. |
| `n_indexer_heads` | Number of heads used by the CSA indexer. |
| `query_compression_dim` | Optional low-rank query bottleneck dimension. |
| `use_attention_sink` | Adds learnable sink KV entries. |
| `use_grouped_output_projection` | Enables grouped attention output projection. |
| `output_projection_groups` | Number of projection groups. Must divide `n_heads`. |
| `use_indexer_score_bias` | Adds indexer score signal into core attention logits. |
| `use_separate_local_kv` | Uses a separate exact local KV branch. |

## FFN Selection

| Parameter | Meaning |
| :--- | :--- |
| `ffn_type` | `dense` or `moe`. |
| `mlp_hidden_dim` | Dense FFN hidden size. |
| `mlp_expansion_factor` | Dense FFN hidden multiplier when explicit hidden dim is absent. |
| `mlp_multiple_of` | Rounds dense hidden dim to this multiple. |
| `mlp_dropout` | Dense FFN dropout. |
| `use_mlp_bias` | Enables biases in dense FFN and related projections. |

## MoE, mHC, and MTP

These groups are documented separately:

- [MoE Config](moe.md)
- [mHC Config](mhc.md)
- [MTP Config](mtp.md)
