
# ============================================================
# tests/test_token_embedding.py
# ============================================================


import math
import pytest
import torch
from src.transformer_modules.embedding_module import *
from data.syntethic_long_context_retrieval import *

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def base_config():
    return EmbeddingConfig(
        vocab_size=128,
        d_model=32,
        pad_token_id=0,
        max_seq_len=64,
        embedding_dropout=0.0,
        scale_embeddings=False,
        init_std=0.02,
        tie_word_embeddings=True,
    )


@pytest.fixture
def embedding(base_config):
    return TokenEmbedding(base_config)


@pytest.fixture
def input_ids(base_config):
    return torch.randint(
        low=1,
        high=base_config.vocab_size,
        size=(4, 16),
        dtype=torch.long,
    )


# ============================================================
# A. Config tests
# ============================================================

def test_valid_config_builds_embedding(base_config):
    embedding = TokenEmbedding(base_config)

    assert embedding.vocab_size == base_config.vocab_size
    assert embedding.d_model == base_config.d_model
    assert embedding.pad_token_id == base_config.pad_token_id
    assert embedding.max_seq_len == base_config.max_seq_len


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 0),
        ("vocab_size", -1),
        ("d_model", 0),
        ("d_model", -1),
        ("max_seq_len", 0),
        ("max_seq_len", -1),
        ("embedding_dropout", -0.1),
        ("embedding_dropout", 1.0),
        ("embedding_dropout", 1.5),
        ("init_std", 0.0),
        ("init_std", -0.01),
        ("pad_token_id", -1),
        ("pad_token_id", 128),
    ],
)
def test_invalid_config_raises_error(base_config, field, value):
    kwargs = vars(base_config).copy()
    kwargs[field] = value

    bad_config = EmbeddingConfig(**kwargs)

    with pytest.raises(ValueError):
        TokenEmbedding(bad_config)


# ============================================================
# B. Forward basic tests
# ============================================================

def test_output_shape(embedding, input_ids, base_config):
    hidden_states = embedding(input_ids)

    assert hidden_states.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        base_config.d_model,
    )


def test_output_dtype_is_floating_point(embedding, input_ids):
    hidden_states = embedding(input_ids)

    assert hidden_states.dtype.is_floating_point


def test_accepts_int32_input_ids(embedding, base_config):
    input_ids = torch.randint(
        low=1,
        high=base_config.vocab_size,
        size=(4, 16),
        dtype=torch.int32,
    )

    hidden_states = embedding(input_ids)

    assert hidden_states.shape == (4, 16, base_config.d_model)
    assert hidden_states.dtype.is_floating_point


def test_rejects_float_input_ids(embedding):
    input_ids = torch.ones((4, 16), dtype=torch.float32)

    with pytest.raises(TypeError):
        embedding(input_ids)


def test_rejects_wrong_shape_1d(embedding):
    input_ids = torch.ones((16,), dtype=torch.long)

    with pytest.raises(ValueError):
        embedding(input_ids)


def test_rejects_wrong_shape_3d(embedding):
    input_ids = torch.ones((4, 16, 1), dtype=torch.long)

    with pytest.raises(ValueError):
        embedding(input_ids)


def test_rejects_sequence_longer_than_max_seq_len(embedding, base_config):
    input_ids = torch.ones(
        (4, base_config.max_seq_len + 1),
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        embedding(input_ids)


def test_rejects_negative_token_ids(embedding):
    input_ids = torch.ones((4, 16), dtype=torch.long)
    input_ids[0, 0] = -1

    with pytest.raises(ValueError):
        embedding(input_ids)


def test_rejects_out_of_vocab_token_ids(embedding, base_config):
    input_ids = torch.ones((4, 16), dtype=torch.long)
    input_ids[0, 0] = base_config.vocab_size

    with pytest.raises(ValueError):
        embedding(input_ids)


# ============================================================
# C. Initialization tests
# ============================================================

def test_embedding_weight_shape(embedding, base_config):
    assert embedding.token_embedding.weight.shape == (
        base_config.vocab_size,
        base_config.d_model,
    )


def test_embedding_weights_are_finite_after_init(embedding):
    assert torch.isfinite(embedding.token_embedding.weight).all()


def test_padding_embedding_is_zero_after_init(embedding, base_config):
    pad_token_id = base_config.pad_token_id

    assert pad_token_id is not None

    pad_row = embedding.weight[pad_token_id]
    expected = torch.zeros_like(pad_row)

    assert torch.allclose(pad_row, expected)


def test_non_padding_embeddings_have_reasonable_std(base_config):
    torch.manual_seed(123)

    embedding = TokenEmbedding(base_config)

    weights = embedding.weight.detach()

    if base_config.pad_token_id is not None:
        mask = torch.ones(base_config.vocab_size, dtype=torch.bool)
        mask[base_config.pad_token_id] = False
        weights = weights[mask]

    mean = weights.mean().item()
    std = weights.std(unbiased=False).item()

    assert abs(mean) < base_config.init_std
    assert abs(std - base_config.init_std) < base_config.init_std


# ============================================================
# D. Padding tests
# ============================================================

@pytest.mark.parametrize("scale_embeddings", [False, True])
def test_padding_output_is_zero_when_dropout_zero(base_config, scale_embeddings):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.0,
            "scale_embeddings": scale_embeddings,
        }
    )

    embedding = TokenEmbedding(config)

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 0, 5, 6],
        ],
        dtype=torch.long,
    )

    hidden_states = embedding(input_ids)

    pad_mask = input_ids == config.pad_token_id
    pad_outputs = hidden_states[pad_mask]

    assert torch.allclose(pad_outputs, torch.zeros_like(pad_outputs))


def test_padding_embedding_gets_zero_gradient(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.0,
            "scale_embeddings": False,
        }
    )

    embedding = TokenEmbedding(config)

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 0, 5, 6],
        ],
        dtype=torch.long,
    )

    hidden_states = embedding(input_ids)
    loss = hidden_states.sum()
    loss.backward()

    pad_grad = embedding.weight.grad[config.pad_token_id]

    assert torch.allclose(pad_grad, torch.zeros_like(pad_grad))


def test_non_padding_tokens_get_gradient(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.0,
            "scale_embeddings": False,
        }
    )

    embedding = TokenEmbedding(config)

    token_id = 7

    input_ids = torch.tensor(
        [
            [0, token_id, 2, 3],
            [4, 0, 5, 6],
        ],
        dtype=torch.long,
    )

    hidden_states = embedding(input_ids)
    loss = hidden_states.sum()
    loss.backward()

    token_grad = embedding.weight.grad[token_id]

    assert token_grad.abs().sum() > 0


# ============================================================
# E. Scaling tests
# ============================================================

def test_forward_without_scaling_matches_raw_lookup(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.0,
            "scale_embeddings": False,
        }
    )

    embedding = TokenEmbedding(config)

    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )

    out = embedding(input_ids)
    expected = embedding.token_embedding(input_ids)

    assert torch.allclose(out, expected)


def test_forward_with_scaling_matches_sqrt_d_model(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.0,
            "scale_embeddings": True,
        }
    )

    embedding = TokenEmbedding(config)

    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )

    out = embedding(input_ids)
    expected = embedding.token_embedding(input_ids) * math.sqrt(config.d_model)

    assert torch.allclose(out, expected)


# ============================================================
# F. Dropout tests
# ============================================================

def test_dropout_zero_is_deterministic(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.0,
        }
    )

    embedding = TokenEmbedding(config)
    embedding.train()

    input_ids = torch.randint(
        low=1,
        high=config.vocab_size,
        size=(8, 32),
        dtype=torch.long,
    )

    out1 = embedding(input_ids)
    out2 = embedding(input_ids)

    assert torch.equal(out1, out2)


def test_dropout_active_changes_output_in_train_mode(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.5,
        }
    )

    embedding = TokenEmbedding(config)
    embedding.train()

    input_ids = torch.randint(
        low=1,
        high=config.vocab_size,
        size=(16, 64),
        dtype=torch.long,)

    out1 = embedding(input_ids)
    out2 = embedding(input_ids)

    assert not torch.equal(out1, out2)


def test_dropout_disabled_in_eval_mode(base_config):
    config = EmbeddingConfig(
        **{
            **vars(base_config),
            "embedding_dropout": 0.5,
        } )

    embedding = TokenEmbedding(config)
    embedding.eval()

    input_ids = torch.randint(
        low=1,
        high=config.vocab_size,
        size=(16, 64),
        dtype=torch.long,)

    out1 = embedding(input_ids)
    out2 = embedding(input_ids)

    assert torch.equal(out1, out2)


# ============================================================
# G. Device and dtype tests
# ============================================================

def test_embedding_runs_on_cpu(base_config):
    embedding = TokenEmbedding(base_config).cpu()

    input_ids = torch.randint(
        low=1,
        high=base_config.vocab_size,
        size=(4, 16),
        dtype=torch.long,
        device="cpu",)

    hidden_states = embedding(input_ids)

    assert hidden_states.device == input_ids.device
    assert hidden_states.shape == (4, 16, base_config.d_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_embedding_runs_on_cuda_if_available(base_config):
    embedding = TokenEmbedding(base_config).cuda()

    input_ids = torch.randint(
        low=1,
        high=base_config.vocab_size,
        size=(4, 16),
        dtype=torch.long,
        device="cuda",)

    hidden_states = embedding(input_ids)

    assert hidden_states.device.type == "cuda"
    assert hidden_states.shape == (4, 16, base_config.d_model)


def test_embedding_respects_module_dtype_bfloat16_if_supported(base_config):
    embedding = TokenEmbedding(base_config).to(dtype=torch.bfloat16)

    input_ids = torch.randint(
        low=1,
        high=base_config.vocab_size,
        size=(4, 16),
        dtype=torch.long,)

    hidden_states = embedding(input_ids)

    assert hidden_states.dtype == torch.bfloat16


# ============================================================
# H. Optional integration test with synthetic dataset
# ============================================================
def test_synthetic_dataset_batch_passes_embedding():
    """
    Integration test: synthetic retrieval dataset -> dict batch -> embedding.
    """

    data_cfg = SyntheticRetrievalConfig(
        num_train_examples=128,
        num_val_examples=32,
        block_size=64,
        min_filler_tokens=8,
        max_filler_tokens=32,
        num_keys_per_example=4,
        vocab_filler_size=100,
        num_key_types=32,
        num_value_types=64,
        batch_size=8,
        num_workers=0,
        seed=123,
    )

    train_loader, _, tokenizer = create_synthetic_retrieval_dataloaders(
        cfg=data_cfg,
        use_mtp=False,
    )

    batch = next(iter(train_loader))

    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "labels" in batch

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    config = EmbeddingConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=data_cfg.block_size,
        embedding_dropout=0.0,
        scale_embeddings=False,
        init_std=0.02,
        tie_word_embeddings=True,
    )

    embedding = TokenEmbedding(config)

    hidden_states = embedding(input_ids)

    assert input_ids.shape == (data_cfg.batch_size, data_cfg.block_size)
    assert labels.shape == (data_cfg.batch_size, data_cfg.block_size)
    assert input_ids.min() >= 0
    assert input_ids.max() < config.vocab_size

    assert hidden_states.shape == (
        data_cfg.batch_size,
        data_cfg.block_size,
        config.d_model,
    )

    assert torch.isfinite(hidden_states).all()


def test_tokenizer_pad_id_matches_embedding_config():
    """
    This test assumes the synthetic tokenizer exposes pad_id.

    If your tokenizer lives elsewhere, adjust the import.
    """

    data_cfg = SyntheticRetrievalConfig()
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(data_cfg)

    config = EmbeddingConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=data_cfg.block_size,
        embedding_dropout=0.0,
        scale_embeddings=False,
        init_std=0.02,
        tie_word_embeddings=True,)

    assert config.pad_token_id == tokenizer.pad_id


# ============================================================
# I. Weight tying preparation
# ============================================================

def test_embedding_weight_property_exposes_parameter(embedding):
    assert embedding.weight is embedding.token_embedding.weight