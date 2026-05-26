<p align="center">
  <img src="header_image.png" width="1000"/>
</p>


# DeepSeek-V4 Mini：忠于论文、从零实现的 PyTorch 版本

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-active_research-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

这是一个非官方的、从零构建的 PyTorch 实现，复现并拆解 **DeepSeek-V4** 背后的核心架构思想。本项目将系统缩小到可读、可测试、适合 CPU 安全测试、可控消融实验和快速研究迭代的规模。

这个仓库不是一个玩具级 Transformer 封装，也不是生产级模型的克隆。它实现了使 DeepSeek-V4 技术报告在系统层面具有研究价值的关键机制：混合压缩注意力、稀疏长上下文检索、Mixture-of-Experts 路由、流形约束的超连接、多 token 预测，以及基于 Muon 的训练流程。

> [NOTE]
> 本项目与 DeepSeek-AI 没有任何隶属关系。它不复现官方 DeepSeek-V4 权重、训练数据、分布式基础设施或生产级 kernel。本项目的目标是提供架构透明性，并支持面向研究的实验。

## 目录

- [🎯 为什么存在这个仓库](#-为什么存在这个仓库)
- [架构覆盖范围](#架构覆盖范围)
- [推理状态](#推理状态)
- [🏗️ 仓库结构](#️-仓库结构)
- [📘 文档](#-文档)
- [安装](#安装)
- [运行测试](#运行测试)
- [⚙️ 模型配置](#️-模型配置)
- [📚 数据集预设](#-数据集预设)
- [🔬 训练一个 Tiny 模型](#-训练一个-tiny-模型)
- [批处理与索引训练](#批处理与索引训练)
- [消融实验套件](#消融实验套件)
- [并行化](#并行化)
- [Docker 支持](#docker-支持)
- [🛠️ 命令行工具](#️-命令行工具)
- [CI 策略](#ci-策略)
- [范围说明](#范围说明)
- [📖 参考文献与引用](#-参考文献与引用)

## 🎯 为什么存在这个仓库

DeepSeek-V4 从三个方向推动了 Transformer 的发展，而这三个方向都值得被独立研究：

1. **上下文长度限制：** 长上下文需要比朴素全注意力更高效的机制。
2. **模型容量：** 扩展模型不仅仅是增加稠密参数，还需要稀疏激活算法。
3. **训练稳定性：** 深层训练的稳定性需要复杂的残差路由与优化机制，而不只是更大的模型。

本项目将这些创新隔离到一个 mini 实现中，使每个组件都能在扩展前被测试、消融，并在小规模语料上训练。

## 架构覆盖范围

| 领域 | 实现状态 |
| :--- | :--- |
| **因果 Transformer** | Token embeddings、RMSNorm、RoPE、MHA、LM head |
| **HCA（Hybrid Context）**| 压缩 KV 分支、滑动窗口分支、因果性测试 |
| **CSA（Compressed Sparse）**| 压缩稀疏块选择、局部窗口、indexer、因果性测试 |
| **MoE（Mixture of Experts）**| 学习式/hash 路由、top-k experts、shared experts、balance metrics |
| **mHC（Hyper-Connections）**| 流扩展、Sinkhorn mixing、模块化 block API |
| **MTP（Multi-Token）** | 辅助 next-n-token heads 与预测损失 |
| **训练引擎** | AdamW 参数组、Muon+AdamW、cosine schedule、AMP、EMA、checkpoints、metrics |
| **数据流水线** | Synthetic retrieval、TinyStories、WikiText-2、AG News、IMDB、MiniPile、FineWeb-Edu |
| **消融实验套件** | 针对注意力、压缩、mHC、MoE、MTP 与全栈组合的六个高层实验套件 |

## 推理状态

该仓库目前包含一个完整的纯 PyTorch 推理路径，适用于 mini DeepSeek stack。它支持通过 tokenizer 输入文本 prompt、用于调试的 token-id prompt、greedy/sampling generation、MTP draft diagnostics，以及三个显式 cache 模式：`audit`、`mha_decode` 和 `deepseek_decode`。

主要的 DeepSeek 推理路径如下：

```python
from inference import inference_autoregresive

out = inference_autoregresive(
    model=model,
    prompt="key key_1 is value_7 question : what is key_1 ? answer :",
    tokenizer=tokenizer,
    cache_mode="deepseek_decode",
    deepseek_prefill_mode="parallel",
    max_new_tokens=32,
    do_sample=False,
    return_cache_stats=True,
)
```

`deepseek_decode` 会构建真实的 HCA/CSA 层级 cache。在 `parallel` prefill 模式下，prompt 只会被处理一次，每个 block 捕获其归一化后的注意力输入，并用压缩全局条目、pending tails 和局部窗口填充 HCA/CSA cache。后续 token 会基于该 cache 逐个解码。

控制台生成使用同一个 wrapper：

```bash
python -m scripts.inference_cli generate \
  --checkpoint outputs/deepseekv4_mini_muon_last_manual.pt \
  --config-json outputs/deepseekv4_mini_muon_last_manual.json \
  --prompt "key key_1 is value_7 question : what is key_1 ? answer :" \
  --synthetic-tokenizer \
  --cache-mode deepseek_decode \
  --deepseek-prefill-mode parallel \
  --max-new-tokens 16 \
  --no-do-sample \
  --return-cache-stats
```

更多内容请参见 [Inference Overview](docs/inference/overview.md)、[Cache Modes](docs/inference/cache_modes.md) 和 [HCA/CSA KV Cache Mechanics](docs/inference/kv_cache.md)。

## 🏗️ 仓库结构

```text
.
├── config/                         # YAML profiles for reproducible experiments
│   ├── data/                       # dataset presets: synthetic, TinyStories, WikiText-2, AG News, IMDB, MiniPile
│   ├── model/                      # model variants: tiny, mini, CSA+MoE+mHC+MTP
│   └── training/                   # CPU smoke and single-GPU training profiles
│
├── data/                           # causal LM datasets, tokenization, and dataloader inspection
│   ├── data_utils.py               # batch normalization helpers used by training/eval
│   ├── inspection.py               # tensor and dataloader summaries for CLI inspection
│   ├── syntethic_long_context_retrieval.py
│   │                                  # local synthetic retrieval task for long-context smoke tests
│   ├── text_datasets.py            # Hugging Face text presets and generic causal LM loader
│   └── tinystories_data.py         # TinyStories-specific tokenizer and dataloaders
│
├── src/                            # DeepSeek-V4 Mini architecture
│   ├── deepseek_csa_attention.py   # Compressed Sparse Attention
│   ├── deepseek_hca_attention.py   # Heavily Compressed Attention
│   ├── deepseek_moe.py             # DeepSeek-style routed/shared MoE layer
│   ├── mHC_residuals.py            # Manifold-Constrained Hyper-Connections
│   ├── deepseek_mtp.py             # Multi-Token Prediction head
│   ├── deepseek_block.py           # configurable Transformer block composition
│   ├── mini_deepseek_class.py      # DeepSeekV4LM wrapper
│   └── transformer_modules/        # baseline RMSNorm, RoPE, MHA, SwiGLU, embeddings, blocks
│
├── training/                       # train/eval stack and diagnostics
│   ├── train_deepseek.py           # high-level orchestration
│   ├── train_one_epoch.py          # single-epoch training loop
│   ├── eval_one_epoch.py           # evaluation and qualitative preview
│   ├── muon_optimizer.py           # Muon + AdamW hybrid optimizer
│   ├── scheduler.py                # warmup + cosine scheduler
│   ├── chekpoints.py               # checkpoint save/load utilities
│   └── *_metrics.py                # LM, MoE, mHC, MTP, and module diagnostics
│
├── ablations/                      # high-level experimental suites
│   ├── ablation_configs.py         # A1-A6 variant generation and memory-scaled configs
│   ├── model_factory.py            # DeepSeekV4LM and MiniCausalLM ablation factory
│   ├── data_factory.py             # synthetic/HF dataloader construction
│   ├── evaluate_ablation.py        # LM, retrieval, and inference/cache metrics
│   ├── run_ablation.py             # sequential runner with checkpointing and cache cleanup
│   ├── report.py                   # summary.csv and summary.md aggregation helpers
│   └── suites.py                   # ablation_1 ... ablation_6 public wrappers
│
├── inference/                      # generation, sampling, active decode, and cache implementations
│   ├── inference_config.py         # generation and cache-mode configuration
│   ├── prefill.py                  # audit, MHA, and DeepSeek prefill paths
│   ├── decode.py                   # one-token decode step and cache metadata
│   ├── hca_cache.py                # HCA compressed/global/local cache state
│   ├── csa_cache.py                # CSA main/index/local cache state
│   ├── deepseek_cache_builder.py   # full-prompt cache construction from layer inputs
│   ├── generate.py                 # autoregressive generation loop
│   └── audit.py                    # high-level audit wrapper and logit comparison
│
├── parallel/                       # PyTorch-native educational parallelism
│   ├── parallel_config.py          # DDP/model-parallel configuration object
│   ├── parallel_utils.py           # rank helpers, seeding, device moves, metric reduction
│   ├── data_parallel.py            # DDP setup, samplers, train/eval wrappers, save helpers
│   ├── model_parallel.py           # layerwise/blockwise DeepSeekV4LM placement
│   └── README.md                   # scope, limitations, and usage notes
│
├── scripts/                        # operational CLIs
│   ├── data_cli.py                 # list, download, tokenize, and inspect datasets
│   ├── train_cli.py                # tiny synthetic training smoke runs
│   ├── inspect_cli.py              # model summaries and module-level test runner
│   ├── ablation_cli.py             # run A1-A6 quick/full ablation suites
│   ├── inference_cli.py            # checkpoint loading and text generation with KV caches
│   └── parallel_cli.py             # DDP/model-parallel smoke tests and placement plans
│
├── docs/                           # architecture and configuration reference
│   ├── architecture/               # CSA, HCA, MoE, mHC, MTP, and model overview
│   ├── training/                   # pipeline, Muon, scheduler, autocast, metrics, EMA/checkpoints
│   ├── config_reference/           # hyperparameter reference by subsystem
│   ├── data/                       # dataset guide
│   ├── parallel/                   # DDP/model-parallel scope and limitations
│   └── cli/                        # command line reference
│
├── tests/                          # CPU-safe coverage for model behavior and causality
│   ├── data/                       # dataset preset and causal text loader tests
│   ├── training/                   # optimizer, scheduler, batch, and tiny-training tests
│   ├── parallel/                   # DDP/model-parallel CPU smoke and utility tests
│   ├── inference/                  # generation, sampling, MTP draft, and active-cache tests
│   ├── test_csa.py                 # CSA shape, causality, and gradient checks
│   ├── test_hca.py                 # HCA compression/local-window checks
│
│
├── notebooks/                      # interactive exploration
├── paper/                          # local DeepSeek-V4 paper reference
├── Dockerfile                      # containerized test/dev environment
├── docker-compose.yml              # compose entrypoint for tests/shell
├── LICENSE
└── README.md
```

## 📘 文档

完整的技术文档位于 [`docs/`](docs/README.md)。文档围绕本项目最重要的两个目标组织：理解架构，以及明确知道哪些超参数可以配置。

推荐入口：

- [Architecture Overview](docs/architecture/overview.md)
- [Attention Modules: MHA, HCA, CSA](docs/architecture/attention_modules.md)
- [HCA: Heavily Compressed Attention](docs/architecture/hca.md)
- [CSA: Compressed Sparse Attention](docs/architecture/csa.md)
- [MoE and Dense FFN](docs/architecture/moe_and_ffn.md)
- [mHC Residual Streams](docs/architecture/mhc.md)
- [MTP Auxiliary Prediction](docs/architecture/mtp.md)
- [Inference Overview](docs/inference/overview.md)
- [Inference Cache Modes](docs/inference/cache_modes.md)
- [HCA/CSA KV Cache Mechanics](docs/inference/kv_cache.md)
- [Training Pipeline](docs/training/pipeline.md)
- [Muon Optimizer](docs/training/muon.md)
- [Metrics and Diagnostics](docs/training/metrics.md)
- [Model Config Reference](docs/config_reference/model.md)
- [Training Config Reference](docs/config_reference/training.md)
- [Data Config Reference](docs/config_reference/data.md)
- [Parallelism Guide](docs/parallel/overview.md)
- [CLI Reference](docs/cli/reference.md)

## 🚀 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,data]"
```

仅用于推理的最小安装：
```bash
pip install -r requirements.txt
```

## 运行测试

该仓库包含一套完整的 CPU 安全测试。仅 CUDA 的检查会在没有 GPU 时正确跳过。

完整本地 CPU 测试：
```bash
pytest
```

仅训练相关测试：
```bash
pytest tests/training
```

数据集 loader 测试：
```bash
pytest tests/data
```

*当前 CPU 验证结果：`756 passed, 4 skipped`*

## ⚙️ 模型配置

可以从 `config/model/` 中的 YAML profiles 开始。这些 profiles 允许你在标准稠密模型和完整 DeepSeek 架构之间无缝切换。

| 配置 | 用途 |
| :--- | :--- |
| `deepseekv4_tiny.yaml` | 用于 CI 和调试的 CPU smoke model |
| `deepseekv4_mini.yaml` | 默认研究模型，包含 Hybrid Attention + MoE + mHC + MTP |
| `deepseekv4_csa_moe_mhc_mtp.yaml` | 全功能集成变体 |

**典型 tiny 模型形状（`deepseekv4_tiny.yaml`）：**
```yaml
model:
  vocab_size: 128
  d_model: 32
  n_layers: 1
  max_seq_len: 32
  attention_type: mha
  ffn_type: dense
```

**Mini 研究 profile（`deepseekv4_mini.yaml`）：**
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

## 📚 数据集预设

本项目通过 `data/text_datasets.py` 支持一组稳健的小到中等规模文本语料：

| Preset | HF Dataset | 主要用途 |
| :--- | :--- | :--- |
| `synthetic_long_context`| Local generator | 用于 CSA/HCA 的 retrieval stress tests |
| `tinystories` | `roneneldan/TinyStories` | Tiny LM generation 与 curriculum training |
| `wikitext2` | `Salesforce/wikitext` | 经典语言建模 benchmark |
| `ag_news` | `fancyzhx/ag_news` | 紧凑的新闻领域语料 |
| `imdb` | `stanfordnlp/imdb` | 更长的评论文本与 domain shift |
| `minipile` | `JeanKaddour/minipile` | 多样化的小型预训练混合语料 |
| `fineweb_edu_10bt_mincols`| `EliMC/fineweb-edu-10BT` | 教育类 web 样本（本地限制） |

通用 loader 会返回适配训练流水线的 dict batch：

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

# Batch structure:
# {
#     "input_ids": LongTensor[B, T],
#     "labels": LongTensor[B, T],
# }
```

## 🔬 训练一个 Tiny 模型

高层 API 是 `training.train_deepseek.train_deepseekv4`。一个最小的 CPU smoke run 如下：

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

## 批处理与索引训练

为了快速迭代和架构调试，可以限制用于构建 blocks 的文档数量：

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

对于组件级调试，强烈推荐使用 **synthetic retrieval dataset**，因为它显式暴露了可控的长距离 key/value 依赖关系：

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

## 消融实验套件

顶层 `ablations/` package 将该仓库从单纯实现扩展成一个实验平台。每个 suite 都有一个高层 wrapper，可以接收数据集配置、面向显存/内存的模型限制、随机种子和训练预算覆盖项。runner 会顺序执行各个变体，在清理前保存 checkpoints，并在进入下一个模型前清理 Python/Torch/CUDA cache。

| Suite | 问题 | 主要变体 |
| :--- | :--- | :--- |
| `A1` Hybrid Attention Composition | 与 MHA、HCA-only、CSA-only 相比，CSA/HCA hybrid attention 是否改善质量-内存权衡？ | `dense_mha_baseline`, `hca_only`, `csa_only`, `hybrid_csa_hca`, `hybrid_hca_csa` |
| `A2` Compression and Window Trade-off | compression factor、local window 和 sparse top-k 如何影响 retrieval、perplexity 与 cache cost？ | HCA 和 CSA 在 `compression_factor`、`window_size`、`top_k_blocks` 上的网格 |
| `A3` mHC Utility | mHC 是否能在浅层和更深的低-token regime 中改善稳定性？ | 带/不带 mHC 的 MHA/hybrid，覆盖浅层与更深层设置 |
| `A4` MoE Routing | routed experts、shared experts 和 balance losses 在 mini regime 中是否有用？ | Dense FFN、无 shared 的 MoE、shared experts、无 balance、hash routing |
| `A5` MTP Auxiliary Loss | MTP 是帮助收敛和生成，还是干扰 next-token learning？ | MTP off、depth/weight sweeps、weighted depth loss |
| `A6` System-Level Stack | 当 DeepSeek-style 组件组合起来时，哪些组件最重要？ | baseline、+compressed attention、+MoE、+mHC、+MTP、full-minus variants |

Python wrapper 示例：

```python
from ablations import ablation_1

results = ablation_1(
    data_config={
        "dataset": "synthetic_long_context",
        "block_size": 128,
        "batch_size": 4,
        "num_train_examples": 2_000,
        "num_val_examples": 300,
    },
    max_model={
        "d_model": 128,
        "n_layers": 4,
        "max_seq_len": 128,
        "n_heads": 4,
        "head_dim": 32,
    },
    training_config={
        "epochs": 1,
        "max_batches_per_epoch": 30,
        "eval_max_batches": 10,
        "optimizer_type": "adamw",
        "device": "cuda",
    },
    seeds=[1],
    quick=False,
)
```

CLI smoke run：

```bash
python -m scripts.ablation_cli --ablation A1 --quick --limit-variants 1 --device cpu
```

输出会写入 `outputs/ablations/{ablation_id}/`，其中每个 variant/seed 都有一个 `final_metrics.json`，并且包含 suite-level 的 `summary.csv` 和 `summary.md`。

## 并行化

该仓库包含一个顶层 `parallel/` package，用于 PyTorch-native 分布式实验。它尽量贴近架构本身，但不声称复现论文中的定制运行时工程。

目前已实现：

- **DDP 数据并行：** `torch.distributed`、`DistributedDataParallel`、`DistributedSampler`、rank-aware checkpoint saves，以及 scalar metric aggregation。
- **Layerwise model parallelism：** 将整个 `DeepSeekV4LM` blocks 分配到有序设备上，并在 block 边界移动 activations。
- **CPU 验证：** 配置校验、单进程 `gloo` DDP、sampler 行为、metric reductions、CPU 上的 model-parallel 等价性，以及 mHC wrapper 兼容性。

V1 只接受活跃的 model-parallel 设备：每个 `balance` 条目必须大于零，因此 `len(devices) <= n_layers`。训练时，请先包装模型，再构建 optimizer：

```python
model = DeepSeekV4LM(config)
model = wrap_model_parallel(model, devices=["cuda:0", "cuda:1"], balance=[8, 8])
optimizer = build_optimizer(model, train_config)
```

按设计未实现的部分包括：custom CUDA kernels、FP4/FP8 training kernels、NCCL topology scheduling、DualPipe，以及真正的 all-to-all expert parallelism。

```bash
python -m scripts.parallel_cli plan --n-layers 6 --devices cpu,cpu --balance 2,4
python -m scripts.parallel_cli model-parallel-smoke --devices cpu --n-layers 2
python -m scripts.parallel_cli ddp-smoke --backend gloo --n-layers 1
```

完整范围与限制请参见 [Parallelism Guide](docs/parallel/overview.md)。

## Docker 支持

```bash
docker build -t deepseekv4-mini .
docker compose run --rm tests
```

## 🛠️ 命令行工具

使用 `pip install -e ".[dev,data]"` 安装后，本项目提供四个透明的 CLI，便于立即交互：

```bash
deepseekv4-data presets
deepseekv4-data synthetic-inspect --block-size 32 --batch-size 2
deepseekv4-train smoke --attention hca --ffn dense --max-batches 2
deepseekv4-inspect model-summary --attention csa --ffn moe
deepseekv4-inspect module-tests csa --quiet
deepseekv4-ablate --ablation A1 --quick --limit-variants 1 --device cpu
deepseekv4-parallel plan --n-layers 4 --devices cpu,cpu
deepseekv4-parallel model-parallel-smoke --devices cpu --n-layers 2
deepseekv4-parallel ddp-smoke --backend gloo --n-layers 1
```

同样的命令也可以不安装 CLI，直接通过 Python modules 原生运行：

```bash
python -m scripts.data_cli synthetic-inspect --block-size 32 --batch-size 2
python -m scripts.train_cli smoke --attention mha --ffn dense --max-batches 1 --quiet
python -m scripts.inspect_cli module-tests training --quiet
python -m scripts.ablation_cli --ablation A1 --quick --limit-variants 1 --device cpu
python -m scripts.parallel_cli tests --quiet
```

**CLI 范围：**
- `data_cli`：列出 presets、检查 synthetic data，并下载 HF text presets。
- `train_cli`：运行一个 tiny synthetic training smoke test，可配置 attention/FFN/mHC/MTP。
- `inspect_cli`：总结模型参数结构，并执行定向 module tests。
- `ablation_cli`：顺序生成并运行 A1-A6 ablation configs，支持 checkpointing 和 cache cleanup。
- `parallel_cli`：检查模型放置方案，运行 CPU-safe layerwise model-parallel smoke tests，运行单进程 `gloo` DDP smoke tests，并启动 `tests/parallel`。

## CI 策略

Continuous Integration 严格采用 path-aware 策略，以在保证关键覆盖率的同时维持速度：
- `src/`、configs 或 packaging 的变更会触发 **model & component tests**。
- `training/` 或 `tests/training/` 的变更会触发 **training-stack tests**。
- `data/` 或 `tests/data/` 的变更会触发 **dataset loader tests**。
- `parallel/`、`scripts/parallel_cli.py` 或 `tests/parallel/` 的变更会触发 **parallelism tests**。
- `ablations/`、`scripts/ablation_cli.py` 或 `tests/experiments/` 的变更会触发 **ablation-suite tests**。
- *所有* 变更都会运行一个轻量级 import smoke test。

## 范围说明

本项目旨在成为架构思想的忠实 mini representation。它**并不**声称与生产级 DeepSeek-V4 权重、高度优化的 custom CUDA kernels、分布式训练框架或数据混合方案达到同等水平。它的价值在于可见性：这些组件是透明的、经过严格测试的、可配置的，并且容易在小规模研究 regime 中训练。

## 📖 参考文献与引用

- **论文副本：** `paper/DeepSeek_V4.pdf`
- **数据集卡片：** Hugging Face 上的 WikiText、TinyStories、AG News、IMDB、MiniPile、FineWeb-Edu sample


本实现基于 DeepSeek-V4 技术报告：

```bibtex
@misc{deepseekai2026deepseekv4,
  author       = {{DeepSeek-AI}},
  title        = {DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/collections/deepseek-ai/deepseek-v4}},
  note         = {Technical report / preview paper}
}
```

如果你在研究中使用本实现，或基于其模块进行改造，请考虑引用：

```bibtex
@misc{reyes2026deepseekv4mini,
  author       = {Reyes Granados, Pablo Alejandro},
  title        = {DeepSeek-V4 Mini: A Paper-Faithful PyTorch Research Implementation},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/pablo-reyes8/deepseek-v4-mini-pytorch}}
}
```
