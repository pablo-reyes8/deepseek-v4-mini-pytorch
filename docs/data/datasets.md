# Dataset Guide

## Local Synthetic Retrieval

Use this when:

- You want no downloads.
- You want deterministic long-context key-value retrieval.
- You want to smoke-test CSA/HCA quickly.

CLI:

```bash
python -m scripts.data_cli synthetic-inspect --block-size 64 --batch-size 2
```

Python:

```python
from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)

cfg = SyntheticRetrievalConfig(block_size=128, batch_size=4)
train_loader, val_loader, tokenizer = create_synthetic_retrieval_dataloaders(cfg)
```

## Hugging Face Text Presets

Use this when:

- You want realistic text.
- You want causal LM batches.
- You want document limits to keep local experiments manageable.

CLI:

```bash
python -m scripts.data_cli presets
python -m scripts.data_cli hf-info wikitext2
python -m scripts.data_cli hf-prepare wikitext2 --max-train-documents 1000
```

Python:

```python
from data.text_datasets import create_hf_text_dataloaders

train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "wikitext2",
    block_size=256,
    batch_size=8,
    max_train_documents=20000,
    max_validation_documents=2000,
)
```

## Batch Contract

All project dataloaders should produce either:

```python
(input_ids, labels)
```

or:

```python
{
    "input_ids": input_ids,
    "labels": labels,
}
```

Training normalizes these formats through `normalize_lm_batch`.
