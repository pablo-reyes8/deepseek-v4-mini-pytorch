# Multi-Token Prediction

MTP means Multi-Token Prediction.

## What It Is

The main LM head predicts the next token. MTP adds auxiliary heads that predict farther future tokens:

```text
main LM head: token t + 1
MTP head 0:  token t + 2
MTP head 1:  token t + 3
...
```

## Role in the Model

- Provides extra autoregressive supervision.
- Encourages hidden states to encode information useful for more than one next-token step.
- Mirrors the DeepSeek-V3/V4 training strategy at mini scale.

## Main Hyperparameters

- `use_mtp`: enables MTP heads and loss.
- `mtp_depth`: number of auxiliary prediction depths.
- `mtp_hidden_dim`: hidden size inside optional MTP transform.
- `use_mtp_transform`: inserts a small transform before each MTP head.
- `mtp_activation`: activation in the transform. Supported values: `silu`, `gelu`, `relu`, `identity`.
- `mtp_dropout`: dropout in the MTP transform.
- `mtp_loss_weight`: global multiplier on MTP auxiliary loss.
- `mtp_tie_with_lm_head`: shares MTP head weights with the main LM head.
- `mtp_depth_loss_weights`: optional per-depth loss weighting.
- `mtp_validate_label_range`: validates MTP labels against vocab range and ignore index.
- `ignore_index`: target value ignored by cross entropy.
- `pad_token_id`: tokenizer pad id, used when constructing labels.

Practical notes:

- Start with `mtp_depth=1` or `2`.
- Keep `mtp_loss_weight=0.3` for paper-inspired mini runs.
- If training is unstable, lower `mtp_loss_weight` before disabling MTP entirely.
