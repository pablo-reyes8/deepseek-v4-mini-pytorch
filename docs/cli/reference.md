# CLI Reference

The project exposes three command groups.

After editable install:

```bash
deepseekv4-data
deepseekv4-train
deepseekv4-inspect
```

Without installation:

```bash
python -m scripts.data_cli
python -m scripts.train_cli
python -m scripts.inspect_cli
```

## Data CLI

List presets:

```bash
python -m scripts.data_cli presets
```

Inspect synthetic data:

```bash
python -m scripts.data_cli synthetic-inspect \
  --block-size 64 \
  --batch-size 2 \
  --num-train-examples 8
```

Show one HF preset:

```bash
python -m scripts.data_cli hf-info wikitext2
```

Prepare and inspect HF data:

```bash
python -m scripts.data_cli hf-prepare wikitext2 \
  --block-size 256 \
  --batch-size 8 \
  --max-tokenizer-documents 10000 \
  --max-train-documents 2000
```

## Train CLI

Run tiny CPU smoke training:

```bash
python -m scripts.train_cli smoke \
  --attention mha \
  --ffn dense \
  --max-batches 1 \
  --quiet
```

Try HCA:

```bash
python -m scripts.train_cli smoke \
  --attention hca \
  --ffn dense \
  --block-size 64 \
  --max-batches 2
```

Try MoE:

```bash
python -m scripts.train_cli smoke \
  --attention csa \
  --ffn moe \
  --num-experts 4 \
  --top-k-experts 2
```

## Inspect CLI

Model summary:

```bash
python -m scripts.inspect_cli model-summary --attention csa --ffn moe
```

Run tests for one module group:

```bash
python -m scripts.inspect_cli module-tests csa --quiet
python -m scripts.inspect_cli module-tests training --quiet
python -m scripts.inspect_cli module-tests data --quiet
```
