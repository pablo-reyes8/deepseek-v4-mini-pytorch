# Contributing

Thanks for helping improve DeepSeek-V4 Mini.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data]"
pytest
```

## Contribution Rules

- Keep default tests CPU-safe and small.
- Use tiny configs in unit tests to avoid high RAM usage.
- Add CUDA-only tests behind `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- Prefer clear component tests over large end-to-end tests.
- Keep model code pedagogical unless an optimization is clearly isolated.
- Update `config/` when adding a new model or training variant.

## Pull Request Checklist

- Tests pass locally with `pytest`.
- New behavior has focused tests.
- Docs/configs are updated when behavior changes.
- Large generated artifacts, checkpoints, datasets, and notebooks outputs are not committed.
