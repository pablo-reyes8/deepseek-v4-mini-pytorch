# DeepSeek-V4 Mini

A serious PyTorch implementation of the core ideas behind the DeepSeek-V4 architecture, scaled down for readable code, CPU-safe tests, and fast research iteration.

This repository is not a toy Transformer wrapper. It implements the pieces that make the DeepSeek-V4 paper interesting as a system:

- **Hybrid long-context attention** with CSA and HCA variants.
- **DeepSeek-style MoE feed-forward layers** with routed and shared experts.
- **mHC residual streams** for expanded residual routing.
- **Multi-Token Prediction** auxiliary heads.
- **Training infrastructure** for optimizer grouping, Muon/AdamW, AMP, EMA, checkpoints, module metrics, and qualitative eval previews.
- **Reproducible configs, CI, Docker, CPU tests, and dataset loaders** suitable for a first public push.

The target is a compact research implementation: close enough to the paper to inspect the actual mechanics, small enough to run and test locally.

## Why This Repo Exists

DeepSeek-V4 pushes the Transformer in three directions that are worth studying independently:

1. Long context needs something better than naive full attention.
2. Capacity needs sparse activation, not just dense scaling.
3. Deep training stability needs residual and optimization machinery, not only a bigger model.

This project isolates those ideas into a mini implementation where each component can be tested, ablated, and trained on small corpora before scaling.

## Architecture Coverage

| Area | Implemented |
| --- | --- |
| Causal Transformer baseline | token embeddings, RMSNorm, RoPE, MHA, LM head |
| HCA | compressed KV branch, sliding window branch, causal tests |
| CSA | compressed sparse block selection, local window, indexer, causal tests |
| MoE | learned/hash routing, top-k experts, shared experts, balance metrics |
| mHC | hyper-connection stream expansion, Sinkhorn mixing, modular block API |
| MTP | auxiliary next-n-token heads and loss |
| Training | AdamW groups, Muon+AdamW, cosine schedule, AMP, EMA, checkpoints, metrics |
| Data | synthetic retrieval, TinyStories, WikiText-2, AG News, IMDB, MiniPile, FineWeb-Edu sample preset |

## Repository Layout

```text
src/                  model components
src/transformer_modules/
training/             training loop, schedulers, optimizers, metrics, checkpoints
data/                 dataset builders and causal LM dataloaders
tests/                CPU-safe component tests
tests/training/       training-stack tests
config/               YAML experiment profiles
.github/              path-aware CI and Dependabot
paper/                DeepSeek-V4 paper reference
proyect_structure/    project scope and implementation guide
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,data]"
```

Minimal install:

```bash
pip install -r requirements.txt
```

## Run Tests

Full local CPU suite:

```bash
pytest
```

Training-only tests:

```bash
pytest tests/training
```

Dataset loader tests:

```bash
pytest tests/data
```

Current validation on CPU:

```text
649 passed, 4 skipped
```

The skipped tests are CUDA-only checks that correctly skip when no GPU is available.

## Model Configs

Start from the YAML profiles in `config/model/`:

| Config | Purpose |
| --- | --- |
| `deepseekv4_tiny.yaml` | CPU smoke model for CI/debugging |
| `deepseekv4_mini.yaml` | default research model with hybrid attention + MoE + mHC + MTP |
| `deepseekv4_csa_moe_mhc_mtp.yaml` | full-feature integration variant |

Typical tiny model shape:

```yaml
model:
  vocab_size: 128
  d_model: 32
  n_layers: 1
  max_seq_len: 32
  attention_type: mha
  ffn_type: dense
```

Mini research profile:

```yaml
model:
  d_model: 256
  n_layers: 6
  attention_type: hybrid
  attention_pattern: [csa, hca]
  ffn_type: moe
  num_experts: 8
  top_k_experts: 2
  use_mhc: true
  use_mtp: true
```

## Dataset Presets

The project now supports a broader set of small-to-medium text corpora through `data/text_datasets.py`:

| Preset | HF dataset | Use |
| --- | --- | --- |
| `synthetic_long_context` | local generator | retrieval stress tests for CSA/HCA |
| `tinystories` | `roneneldan/TinyStories` | tiny LM generation and curriculum-style training |
| `wikitext2` | `Salesforce/wikitext`, `wikitext-2-raw-v1` | classic language modeling benchmark |
| `ag_news` | `fancyzhx/ag_news` | compact news-domain corpus |
| `imdb` | `stanfordnlp/imdb` | longer review text and domain shift |
| `minipile` | `JeanKaddour/minipile` | diverse small pretraining mix |
| `fineweb_edu_10bt_mincols` | `EliMC/fineweb-edu-10BT-mincols` | educational web sample; limit documents locally |

The generic loader returns batches shaped for the training pipeline:

```python
from data.text_datasets import create_hf_text_dataloaders

train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "wikitext2",
    block_size=256,
    batch_size=8,
    vocab_size=16_000,
    max_tokenizer_documents=50_000,
    max_train_documents=20_000,
    max_validation_documents=2_000,
)
```

Every batch is a dict:

```python
{
    "input_ids": LongTensor[B, T],
    "labels": LongTensor[B, T],
}
```

## Training A Tiny Model

The high-level API is `training.train_deepseek.train_deepseekv4`. A minimal CPU smoke run looks like:

```python
from data.text_datasets import create_hf_text_dataloaders
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig
from training.train_deepseek import train_deepseekv4

train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "wikitext2",
    block_size=64,
    batch_size=4,
    vocab_size=4096,
    max_tokenizer_documents=1000,
    max_train_documents=1000,
    max_validation_documents=200,
)

model = DeepSeekV4LM(
    DeepSeekV4LMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=64,
        n_layers=2,
        max_seq_len=64,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        attention_type="hca",
        n_heads=4,
        head_dim=16,
        rotary_dim=16,
        ffn_type="dense",
        mlp_hidden_dim=128,
    )
)

history = train_deepseekv4(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cpu",
    amp_enabled=False,
    optimizer_type="adamw",
    learning_rate=3e-4,
    epochs=1,
    max_batches_per_epoch=10,
    eval_max_batches=5,
    ckpt_dir="checkpoints/wikitext2_tiny",
)
```

## Training With Batches and Indexing

For quick iteration, limit the number of documents used to build blocks:

```python
train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "ag_news",
    block_size=128,
    batch_size=16,
    max_train_documents=5000,
    max_validation_documents=1000,
)

for step, batch in enumerate(train_loader):
    input_ids = batch["input_ids"]  # [B, T]
    labels = batch["labels"]        # [B, T]
    if step == 0:
        print(input_ids.shape, labels.shape)
    break
```

For component debugging, use the synthetic retrieval dataset because it exposes controlled long-range key/value dependencies:

```python
from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)

cfg = SyntheticRetrievalConfig(
    block_size=256,
    min_filler_tokens=64,
    max_filler_tokens=220,
    batch_size=8,
)

train_loader, val_loader, tokenizer = create_synthetic_retrieval_dataloaders(cfg)
```

## Docker

```bash
docker build -t deepseekv4-mini .
docker compose run --rm tests
```

## CI Strategy

CI is path-aware:

- `src/`, model tests, configs, or packaging changes run model/component tests.
- `training/` or `tests/training/` changes run training-stack tests.
- `data/`, `tests/data/`, or data configs run dataset loader tests.
- All changes run a lightweight import smoke test.

This keeps pull requests fast without losing coverage where it matters.

## Notes on Scope

This project aims to be a faithful mini representation of the architectural ideas, not a claim of parity with production DeepSeek-V4 weights, kernels, distributed training, tokenizer, data mixture, or frontier-scale infrastructure. The value is that the components are visible, tested, configurable, and trainable in small regimes.

## References

- Paper copy: `paper/DeepSeek_V4.pdf`
- Dataset cards: WikiText, TinyStories, AG News, IMDB, MiniPile, FineWeb-Edu sample on Hugging Face
