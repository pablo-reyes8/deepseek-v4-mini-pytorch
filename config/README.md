# Configs

These YAML files are intentionally explicit. They document reproducible experiment profiles without forcing the training code to depend on a single config loader yet.

- `model/`: architecture variants.
- `data/`: dataset and tokenizer settings.
- `training/`: optimizer, scheduler, checkpoint, and precision profiles.

Recommended first runs:

- `model/deepseekv4_tiny.yaml` + `data/synthetic_long_context.yaml` + `training/cpu_smoke.yaml`
- `model/deepseekv4_tiny.yaml` + `data/wikitext2.yaml` + `training/cpu_smoke.yaml`
- `model/deepseekv4_mini.yaml` + `data/tinystories.yaml` + `training/gpu_single.yaml`
