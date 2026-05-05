# MTP Config Reference

Primary config: `MTPConfig`.

Model-level fields are prefixed with `mtp_` inside `DeepSeekV4LMConfig`.

| Model Field | MTP Field | Meaning |
| :--- | :--- | :--- |
| `use_mtp` | n/a | Enables auxiliary MTP heads and loss. |
| `mtp_depth` | `mtp_depth` | Number of future-token depths to predict. |
| `mtp_hidden_dim` | `hidden_dim` | Hidden width in optional MTP transform. |
| `use_mtp_transform` | `use_mtp_transform` | Adds transform before MTP heads. |
| `mtp_activation` | `activation` | `silu`, `gelu`, `relu`, or `identity`. |
| `mtp_dropout` | `dropout` | Dropout in MTP transform. |
| `use_mlp_bias` | `use_bias` | Bias in transform layers. |
| `mtp_loss_weight` | `mtp_loss_weight` | Global MTP auxiliary loss multiplier. |
| `mtp_tie_with_lm_head` | `tie_with_lm_head` | Shares MTP head weight with LM head. |
| `ignore_index` | `ignore_index` | Label value ignored by cross entropy. |
| `pad_token_id` | `pad_token_id` | Tokenizer pad id. |
| `mtp_depth_loss_weights` | `depth_loss_weights` | Optional per-depth loss weights. |
| `mtp_validate_label_range` | `validate_label_range` | Validates label ids. |

Recommended start:

```yaml
use_mtp: true
mtp_depth: 1
mtp_loss_weight: 0.3
use_mtp_transform: true
mtp_activation: silu
```
