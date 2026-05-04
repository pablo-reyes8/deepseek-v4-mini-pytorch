# ============================================================
# SYNTHETIC LONG-CONTEXT RETRIEVAL DATASET
# Needle / Key-Value Retrieval for Mini DeepSeek-V4
# ============================================================

import random
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

### Debug rápido ###
#block_size = 256
#min_filler_tokens = 32
#max_filler_tokens = 160
#batch_size = 32

### Long-context medio ###
#block_size = 1024
#min_filler_tokens = 256
#max_filler_tokens = 900
#batch_size = 8

### CSA/HCA stress test ###
#block_size = 2048
#min_filler_tokens = 768
#max_filler_tokens = 1800
#batch_size = 2


@dataclass
class SyntheticRetrievalConfig:
    # Dataset size
    num_train_examples: int = 50_000
    num_val_examples: int = 5_000

    # Sequence length
    block_size: int = 64         # input length
    min_filler_tokens: int = 64
    max_filler_tokens: int = 420

    # Task structure
    num_keys_per_example: int = 8  # number of key-value pairs in context
    vocab_filler_size: int = 550
    num_key_types: int = 128
    num_value_types: int = 256

    # DataLoader
    batch_size: int = 32
    num_workers: int = 0

    # Reproducibility
    seed: int = 42


CFG = SyntheticRetrievalConfig()

CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = CFG.num_workers if CFG.num_workers is not None else min(4, CPU_COUNT - 1)


# ============================================================
# SIMPLE TOKENIZER
# ============================================================

class SimpleWordTokenizer:
    """
    Tokenizer simple basado en espacios.

    Ventaja:
    - Totalmente controlado.
    - Perfecto para synthetic retrieval.
    - No dependemos de HuggingFace ni tokenizers externos.
    """

    def __init__(self):
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_idx = {}
        self.idx_to_token = {}

        for tok in self.special_tokens:
            self.add_token(tok)

    def add_token(self, token: str):
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def build_vocab(self, cfg: SyntheticRetrievalConfig):
        # Structural tokens
        structural_tokens = [
            "key", "is", "question", "what", "?", "answer", ":",
            "noise", "the", "and", "then", "because", "context",
            "remember", "value", "token"
        ]

        for tok in structural_tokens:
            self.add_token(tok)

        # Key tokens
        for i in range(cfg.num_key_types):
            self.add_token(f"key_{i}")

        # Value tokens
        for i in range(cfg.num_value_types):
            self.add_token(f"value_{i}")

        # Filler tokens
        for i in range(cfg.vocab_filler_size):
            self.add_token(f"filler_{i}")

    def encode(self, text: str) -> List[int]:
        tokens = text.strip().split()
        unk = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(tok, unk) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.idx_to_token.get(int(i), "<unk>") for i in ids)

    @property
    def vocab_size(self):
        return len(self.token_to_idx)

    @property
    def pad_id(self):
        return self.token_to_idx["<pad>"]

    @property
    def bos_id(self):
        return self.token_to_idx["<bos>"]

    @property
    def eos_id(self):
        return self.token_to_idx["<eos>"]


# ============================================================
# EXAMPLE GENERATOR
# ============================================================

class SyntheticRetrievalGenerator:
    """
    Genera ejemplos de recuperación de largo contexto.

    Estructura:

    <bos>
    key_3 is value_87
    key_10 is value_21
    ...
    filler_832 filler_14 ...
    question : what is key_10 ?
    answer : value_21
    <eos>

    El objetivo autoregresivo será predecir todo el texto.
    Para análisis específico, también devolvemos query_key y answer_value.
    """

    def __init__(self, cfg: SyntheticRetrievalConfig, tokenizer: SimpleWordTokenizer, seed: int = 42):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)

        self.keys = [f"key_{i}" for i in range(cfg.num_key_types)]
        self.values = [f"value_{i}" for i in range(cfg.num_value_types)]
        self.fillers = [f"filler_{i}" for i in range(cfg.vocab_filler_size)]

    def _sample_filler(self, n: int) -> List[str]:
        return [self.rng.choice(self.fillers) for _ in range(n)]

    def generate_text_example(self) -> Tuple[str, Dict]:
        cfg = self.cfg

        # Sample unique keys
        selected_keys = self.rng.sample(self.keys, cfg.num_keys_per_example)
        selected_values = [self.rng.choice(self.values) for _ in range(cfg.num_keys_per_example)]

        kv_pairs = dict(zip(selected_keys, selected_values))

        # Pick one key to query
        query_key = self.rng.choice(selected_keys)
        answer_value = kv_pairs[query_key]

        tokens = ["<bos>"]

        # Add key-value facts
        for k, v in kv_pairs.items():
            tokens.extend([k, "is", v])

            # Small local noise between facts
            local_noise_len = self.rng.randint(1, 5)
            tokens.extend(self._sample_filler(local_noise_len))

        # Long filler between facts and question
        filler_len = self.rng.randint(cfg.min_filler_tokens, cfg.max_filler_tokens)
        tokens.extend(self._sample_filler(filler_len))

        # Query and answer
        tokens.extend([
            "question", ":", "what", "is", query_key, "?",
            "answer", ":", answer_value,
            "<eos>"
        ])

        text = " ".join(tokens)

        meta = {
            "query_key": query_key,
            "answer_value": answer_value,
            "kv_pairs": kv_pairs,
            "filler_len": filler_len,
        }

        return text, meta


# ============================================================
# DATASET
# ============================================================

class SyntheticRetrievalDataset(Dataset):
    """
    Dataset autoregresivo.

    Devuelve:
      input_ids: [block_size]
      labels:    [block_size]

    labels es input_ids desplazado un token hacia adelante.
    """

    def __init__(
        self,
        cfg: SyntheticRetrievalConfig,
        tokenizer: SimpleWordTokenizer,
        split: str = "train",
    ):
        super().__init__()

        assert split in ["train", "val"]

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.split = split
        self.num_examples = cfg.num_train_examples if split == "train" else cfg.num_val_examples

        seed = cfg.seed if split == "train" else cfg.seed + 10_000
        self.generator = SyntheticRetrievalGenerator(cfg, tokenizer, seed=seed)

        self.pad_id = tokenizer.pad_id
        self.block_size = cfg.block_size

    def __len__(self):
        return self.num_examples

    def _pad_or_truncate(self, ids: List[int], target_len: int) -> List[int]:
        if len(ids) >= target_len:
            return ids[:target_len]
        return ids + [self.pad_id] * (target_len - len(ids))

    def __getitem__(self, idx):
        text, meta = self.generator.generate_text_example()

        # Need block_size + 1 because input is ids[:-1], label is ids[1:]
        ids = self.tokenizer.encode(text)
        ids = self._pad_or_truncate(ids, self.block_size + 1)

        ids = torch.tensor(ids, dtype=torch.long)

        input_ids = ids[:-1]
        labels = ids[1:]

        return {
        "input_ids": input_ids,
        "labels": labels,
    }


# ============================================================
# OPTIONAL: MTP DATASET VERSION
# ============================================================

class SyntheticRetrievalMTPDataset(Dataset):
    """
    Dataset con soporte para Multi-Token Prediction.

    Devuelve:
      input_ids: [block_size]
      labels:    [block_size]  predice x_{t+1}
      mtp_labels:[mtp_depth, block_size] predice x_{t+2}, x_{t+3}, ...
    """

    def __init__(
        self,
        cfg: SyntheticRetrievalConfig,
        tokenizer: SimpleWordTokenizer,
        split: str = "train",
        mtp_depth: int = 1,
    ):
        super().__init__()

        assert split in ["train", "val"]

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.split = split
        self.mtp_depth = mtp_depth

        self.num_examples = cfg.num_train_examples if split == "train" else cfg.num_val_examples

        seed = cfg.seed if split == "train" else cfg.seed + 10_000
        self.generator = SyntheticRetrievalGenerator(cfg, tokenizer, seed=seed)

        self.pad_id = tokenizer.pad_id
        self.block_size = cfg.block_size

    def __len__(self):
        return self.num_examples

    def _pad_or_truncate(self, ids: List[int], target_len: int) -> List[int]:
        if len(ids) >= target_len:
            return ids[:target_len]
        return ids + [self.pad_id] * (target_len - len(ids))

    def __getitem__(self, idx):
        text, meta = self.generator.generate_text_example()

        # Need block_size + 1 + mtp_depth
        total_len = self.block_size + 1 + self.mtp_depth

        ids = self.tokenizer.encode(text)
        ids = self._pad_or_truncate(ids, total_len)
        ids = torch.tensor(ids, dtype=torch.long)

        input_ids = ids[:self.block_size]
        labels = ids[1:self.block_size + 1]

        mtp_labels = []
        for k in range(2, 2 + self.mtp_depth):
            mtp_labels.append(ids[k:k + self.block_size])

        mtp_labels = torch.stack(mtp_labels, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mtp_labels": mtp_labels,
        }


# ============================================================
# DATALOADERS
# ============================================================

def create_synthetic_retrieval_dataloaders(
    cfg: SyntheticRetrievalConfig = CFG,
    use_mtp: bool = False,
    mtp_depth: int = 1,):
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(cfg)

    print("Synthetic tokenizer vocab size:", tokenizer.vocab_size)

    if use_mtp:
        train_ds = SyntheticRetrievalMTPDataset(
            cfg=cfg,
            tokenizer=tokenizer,
            split="train",
            mtp_depth=mtp_depth,
        )

        val_ds = SyntheticRetrievalMTPDataset(
            cfg=cfg,
            tokenizer=tokenizer,
            split="val",
            mtp_depth=mtp_depth,
        )

    else:
        train_ds = SyntheticRetrievalDataset(
            cfg=cfg,
            tokenizer=tokenizer,
            split="train",
        )

        val_ds = SyntheticRetrievalDataset(
            cfg=cfg,
            tokenizer=tokenizer,
            split="val",
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, tokenizer

