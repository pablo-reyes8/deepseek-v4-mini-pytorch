# Architecture Overview

DeepSeek-V4 Mini is a configurable causal language model that keeps the important architectural ideas of the DeepSeek-V4 paper while staying small enough to inspect and test on CPU.

The model is built around `DeepSeekV4LM` and `DeepSeekV4Block`.

## High-Level Flow

```text
input_ids
  -> TokenEmbedding
  -> DeepSeekV4Block x n_layers
       -> attention branch: MHA / HCA / CSA / hybrid schedule
       -> feed-forward branch: dense SwiGLU / DeepSeekMoE
       -> optional mHC residual stream around attention and FFN
  -> final RMSNorm
  -> LM head
  -> optional MTP heads
```

## Main Modules

### `DeepSeekV4LM`

The full language model wrapper.

Role:

- Owns embeddings, blocks, final norm, LM head, and optional MTP head.
- Computes causal LM loss.
- Supports dense, MoE, HCA, CSA, hybrid attention, mHC, and MTP through config flags.

### `DeepSeekV4Block`

The repeated Transformer block.

Role:

- Chooses attention type per layer.
- Chooses dense FFN or MoE.
- Wraps attention and FFN with classic residuals or mHC residual streams.
- Collects auxiliary outputs from MoE, mHC, CSA, HCA, and MTP when requested.

## Important Design Choices

- This project favors readable PyTorch over custom kernels.
- Long-context mechanisms are implemented pedagogically, not as production kernels.
- CPU tests use tiny configs, but the same config surface scales to larger experiments.
- Industrial systems from the paper, such as fused MoE kernels, FP4 QAT, expert parallelism, and on-disk KV cache, are intentionally out of scope for this mini repo.

## Core Hyperparameter Groups

- Model size: `vocab_size`, `d_model`, `n_layers`, `max_seq_len`.
- Attention: `attention_type`, `n_heads`, `head_dim`, `compression_factor`, `top_k_blocks`, `window_size`.
- FFN/MoE: `ffn_type`, `mlp_hidden_dim`, `num_experts`, `top_k_experts`, `router_type`.
- mHC: `use_mhc`, `n_hc`, `mhc_sinkhorn_iters`.
- MTP: `use_mtp`, `mtp_depth`, `mtp_loss_weight`.
- Training: optimizer, LR schedule, AMP, EMA, checkpointing, logging, diagnostics.
