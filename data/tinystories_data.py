# ============================================================
# TINYSTORIES DATA PIPELINE
# Mini DeepSeek-V4 / GPT-style causal LM
# ============================================================

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "roneneldan/TinyStories"

# TinyStories tiene train / validation
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"

VOCAB_SIZE = 16000
MIN_FREQ = 2

BLOCK_SIZE = 256   
BATCH_SIZE = 64

TOKENIZER_PATH = Path("tinystories_tokenizer.json")

CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = 2 if CPU_COUNT <= 2 else min(4, CPU_COUNT - 1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# LOAD DATASET
# ============================================================

def load_tinystories():
    """
    Carga TinyStories desde Hugging Face.

    Dataset:
      roneneldan/TinyStories

    Splits esperados:
      - train
      - validation
    """
    ds = load_dataset(DATASET_NAME)

    train_ds = ds[TRAIN_SPLIT]
    val_ds = ds[VAL_SPLIT]

    print(train_ds)
    print(val_ds)

    return train_ds, val_ds


# ============================================================
# TOKENIZER
# ============================================================

def train_tokenizer(
    train_ds,
    vocab_size=VOCAB_SIZE,
    min_freq=MIN_FREQ,
    save_path=TOKENIZER_PATH,
):
    """
    Entrena un tokenizer BPE byte-level estilo GPT.

    Nota importante:
    - NO usamos lowercase.
    - Queremos preservar mayúsculas, nombres, puntuación y estructura.
    """

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),])

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    def batch_iterator():
        for ex in train_ds:
            txt = ex["text"]
            if txt is not None and len(txt.strip()) > 0:
                yield txt

    print("Entrenando tokenizer BPE sobre TinyStories...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    print("Tamaño vocabulario:", tokenizer.get_vocab_size())

    save_path = Path(save_path)
    tokenizer.save(str(save_path))

    print(f"Tokenizer guardado en: {save_path.resolve()}")

    return tokenizer


def load_or_train_tokenizer(train_ds):
    """
    Carga tokenizer si existe; si no, lo entrena.
    """
    if TOKENIZER_PATH.exists():
        print(f"Cargando tokenizer desde {TOKENIZER_PATH}...")
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    else:
        tokenizer = train_tokenizer(train_ds)

    return tokenizer


# ============================================================
# DATASET CAUSAL LM
# ============================================================

class TinyStoriesCausalDataset(Dataset):
    """
    Dataset autoregresivo:

      - Concatena historias.
      - Añade <bos> al inicio y <eos> al final de cada historia.
      - Parte en chunks de block_size + 1.
      - input_ids = ids[:-1]
      - labels    = ids[1:]

    Devuelve:
      input_ids: [block_size]
      labels:    [block_size]
    """

    def __init__(self, hf_split, tokenizer, block_size=BLOCK_SIZE):
        super().__init__()

        self.block_size = block_size

        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        pad_id = tokenizer.token_to_id("<pad>")

        if bos_id is None:
            raise ValueError("El tokenizer no tiene token <bos>.")
        if eos_id is None:
            raise ValueError("El tokenizer no tiene token <eos>.")
        if pad_id is None:
            raise ValueError("El tokenizer no tiene token <pad>.")

        all_ids = []

        print("Tokenizando y concatenando TinyStories...")

        for ex in hf_split:
            txt = ex["text"]

            if txt is None or len(txt.strip()) == 0:
                continue

            enc = tokenizer.encode(txt)

            # Añadimos frontera explícita de documento/historia
            all_ids.extend([bos_id] + enc.ids + [eos_id])

        self.data = torch.tensor(all_ids, dtype=torch.long)

        print(f"Total de tokens en split: {len(self.data):,}")

        chunk_len = block_size + 1
        n_chunks = len(self.data) // chunk_len

        if n_chunks == 0:
            raise ValueError(
                "Muy pocos tokens para formar chunks. "
                "Baja BLOCK_SIZE o usa más datos.")

        self.data = self.data[: n_chunks * chunk_len]
        self.data = self.data.view(n_chunks, chunk_len)

        self.inputs = self.data[:, :-1]
        self.targets = self.data[:, 1:]

        print(f"Número de secuencias: {len(self.inputs):,}")
        print(f"Forma inputs:  {self.inputs.shape}")
        print(f"Forma targets: {self.targets.shape}")

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# OPTIONAL MTP DATASET
# ============================================================

class TinyStoriesMTPDataset(Dataset):
    """
    Dataset para Multi-Token Prediction.

    Devuelve diccionario:

      {
        "input_ids":  [block_size],
        "labels":     [block_size],              # x_{t+1}
        "mtp_labels": [mtp_depth, block_size],   # x_{t+2}, x_{t+3}, ...
      }
    """

    def __init__(self, hf_split, tokenizer, block_size=BLOCK_SIZE, mtp_depth=1):
        super().__init__()

        self.block_size = block_size
        self.mtp_depth = mtp_depth

        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")

        if bos_id is None:
            raise ValueError("El tokenizer no tiene token <bos>.")
        if eos_id is None:
            raise ValueError("El tokenizer no tiene token <eos>.")

        all_ids = []

        print("Tokenizando TinyStories para MTP...")

        for ex in hf_split:
            txt = ex["text"]

            if txt is None or len(txt.strip()) == 0:
                continue

            enc = tokenizer.encode(txt)
            all_ids.extend([bos_id] + enc.ids + [eos_id])

        self.data = torch.tensor(all_ids, dtype=torch.long)

        print(f"Total de tokens en split: {len(self.data):,}")

        chunk_len = block_size + 1 + mtp_depth
        n_chunks = len(self.data) // chunk_len

        if n_chunks == 0:
            raise ValueError(
                "Muy pocos tokens para formar chunks MTP. "
                "Baja BLOCK_SIZE o usa más datos.")

        self.data = self.data[: n_chunks * chunk_len]
        self.data = self.data.view(n_chunks, chunk_len)

        print(f"Número de secuencias MTP: {len(self.data):,}")
        print(f"Forma data: {self.data.shape}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        ids = self.data[idx]

        input_ids = ids[:self.block_size]
        labels = ids[1:self.block_size + 1]

        mtp_labels = []

        for k in range(2, 2 + self.mtp_depth):
            mtp_labels.append(ids[k:k + self.block_size])

        mtp_labels = torch.stack(mtp_labels, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mtp_labels": mtp_labels}


# ============================================================
# DATALOADERS
# ============================================================

def create_tinystories_dataloaders(
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_mtp=False,
    mtp_depth=1):

    train_hf, val_hf = load_tinystories()

    tokenizer = load_or_train_tokenizer(train_hf)

    if use_mtp:
        train_ds = TinyStoriesMTPDataset(
            train_hf,
            tokenizer,
            block_size=block_size,
            mtp_depth=mtp_depth)

        val_ds = TinyStoriesMTPDataset(
            val_hf,
            tokenizer,
            block_size=block_size,
            mtp_depth=mtp_depth)

    else:
        train_ds = TinyStoriesCausalDataset(
            train_hf,
            tokenizer,
            block_size=block_size)

        val_ds = TinyStoriesCausalDataset(
            val_hf,
            tokenizer,
            block_size=block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, tokenizer
