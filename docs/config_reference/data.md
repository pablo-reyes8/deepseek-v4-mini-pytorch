# Data Config Reference

This project supports local synthetic retrieval data and Hugging Face text datasets.

## Synthetic Retrieval Config

Primary config: `SyntheticRetrievalConfig`.

| Parameter | Meaning |
| :--- | :--- |
| `num_train_examples` | Number of generated training examples. |
| `num_val_examples` | Number of generated validation examples. |
| `block_size` | Input sequence length. Labels are shifted one token ahead. |
| `min_filler_tokens` | Minimum distractor/filler tokens between facts and question. |
| `max_filler_tokens` | Maximum distractor/filler tokens. |
| `num_keys_per_example` | Number of key-value facts per example. |
| `vocab_filler_size` | Number of filler token types. |
| `num_key_types` | Number of possible key tokens. |
| `num_value_types` | Number of possible value tokens. |
| `batch_size` | Dataloader batch size. |
| `num_workers` | Dataloader worker count. |
| `seed` | Generator seed. |

Role:

- Tests whether the model can retrieve a value associated with a key across distractor context.
- Useful for CSA/HCA long-context smoke tests without downloading data.

## Hugging Face Text Dataset Presets

Configured in `data/text_datasets.py`.

| Preset | Dataset | Role |
| :--- | :--- | :--- |
| `wikitext2` | `Salesforce/wikitext`, `wikitext-2-raw-v1` | Small language modeling benchmark. |
| `tinystories` | `roneneldan/TinyStories` | Tiny generation-friendly corpus. |
| `ag_news` | `fancyzhx/ag_news` | Compact news-domain text. |
| `imdb` | `stanfordnlp/imdb`, `plain_text` | Longer review text. |
| `minipile` | `JeanKaddour/minipile` | Small diverse pretraining mix. |
| `fineweb_edu_10bt_mincols` | `EliMC/fineweb-edu-10BT-mincols` | Educational web sample; use document limits. |

## Generic HF Loader Parameters

Function: `create_hf_text_dataloaders`.

| Parameter | Meaning |
| :--- | :--- |
| `preset_name` | Dataset preset key. |
| `block_size` | Causal LM sequence length. Defaults to preset recommendation. |
| `batch_size` | Dataloader batch size. |
| `num_workers` | Dataloader worker count. |
| `tokenizer_path` | Path to save/load byte-level BPE tokenizer. |
| `vocab_size` | Tokenizer vocabulary size. |
| `min_frequency` | Minimum BPE token frequency. |
| `max_tokenizer_documents` | Limits docs used for tokenizer training. |
| `max_train_documents` | Limits train docs used for dataset construction. |
| `max_validation_documents` | Limits validation docs. |

Output batch format:

```python
{
    "input_ids": LongTensor[B, T],
    "labels": LongTensor[B, T],
}
```
