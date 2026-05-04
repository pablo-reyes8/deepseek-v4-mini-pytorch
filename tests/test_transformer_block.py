# @title
# ============================================================
# TransformerBlock baseline tests
# ============================================================

import pytest
import torch


from src.transformer_modules.transformer_block import *
from data.syntethic_long_context_retrieval import *
# ============================================================
# Helpers
# ============================================================

def make_block_config(**overrides):
    cfg = dict(
        d_model=64,
        rms_norm_eps=1e-6,

        n_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,
        max_seq_len=128,

        mlp_hidden_dim=256,
        mlp_expansion_factor=4.0,
        mlp_multiple_of=1,
        mlp_dropout=0.0,
        use_mlp_bias=False,

        init_std=0.02,
    )
    cfg.update(overrides)
    return TransformerBlockConfig(**cfg)


def make_block(**overrides):
    return TransformerBlock(make_block_config(**overrides))


def make_block_input(B=2, T=8, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


def zero_module_parameters(module):
    for param in module.parameters():
        with torch.no_grad():
            param.zero_()


# ============================================================
# A. Config tests
# ============================================================

def test_valid_block_config_builds():
    config = make_block_config(
        d_model=256,
        n_heads=4,
        head_dim=64,
        mlp_hidden_dim=1024,
        max_seq_len=128,
        rotary_dim=64,
    )
    block = TransformerBlock(config)

    assert block.d_model == 256
    assert block.attention.n_heads == 4
    assert block.attention.head_dim == 64
    assert block.mlp.hidden_dim == 1024


@pytest.mark.parametrize("d_model", [0, -1, -256])
def test_invalid_block_d_model_raises(d_model):
    with pytest.raises(ValueError):
        TransformerBlock(make_block_config(d_model=d_model))


@pytest.mark.parametrize("rms_norm_eps", [0.0, -1e-6])
def test_invalid_rms_norm_eps_raises(rms_norm_eps):
    with pytest.raises(ValueError):
        TransformerBlock(make_block_config(rms_norm_eps=rms_norm_eps))


def test_attention_and_mlp_d_model_match_block():
    # En la config plana no es posible pasar attention.d_model o mlp.d_model
    # distintos directamente; esta prueba verifica que los subconfigs heredan d_model.
    config = make_block_config(d_model=64)

    attn_config = config.to_attention_config()
    mlp_config = config.to_mlp_config()

    assert attn_config.d_model == config.d_model
    assert mlp_config.d_model == config.d_model


# ============================================================
# B. Internal structure tests
# ============================================================

def test_block_has_expected_modules():
    block = make_block()

    assert hasattr(block, "norm1")
    assert hasattr(block, "attention")
    assert hasattr(block, "norm2")
    assert hasattr(block, "mlp")

    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.attention, CausalMultiHeadAttention)
    assert isinstance(block.norm2, RMSNorm)
    assert isinstance(block.mlp, SwiGLUMLP)


def test_norm_weights_initialized_to_ones():
    block = make_block()

    assert torch.allclose(block.norm1.weight, torch.ones_like(block.norm1.weight))
    assert torch.allclose(block.norm2.weight, torch.ones_like(block.norm2.weight))


# ============================================================
# C. Forward tests
# ============================================================

def test_block_output_shape_matches_input_shape():
    block = make_block()

    x = make_block_input(B=2, T=8, D=64)
    out = block(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(8, 64),
        torch.randn(2, 8, 64, 1),
    ],
)
def test_block_rejects_wrong_input_rank(bad_x):
    block = make_block()

    with pytest.raises(ValueError):
        block(bad_x)


def test_block_rejects_wrong_hidden_size():
    block = make_block(d_model=64)

    x = torch.randn(2, 8, 32)

    with pytest.raises(ValueError):
        block(x)


def test_returns_aux_when_need_weights_true():
    block = make_block()
    x = make_block_input(B=2, T=8, D=64)

    out, aux = block(x, need_weights=True)

    assert out.shape == (2, 8, 64)
    assert isinstance(aux, dict)
    assert "attn_weights" in aux
    assert aux["attn_weights"].shape == (2, 4, 8, 8)


# ============================================================
# D. Residual / pre-norm tests
# ============================================================

def test_block_changes_input_when_weights_nonzero():
    block = make_block()
    block.eval()

    x = make_block_input(B=2, T=8, D=64)
    out = block(x)

    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_residual_identity_when_attention_and_mlp_zeroed():
    block = make_block(
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
        use_attention_bias=True,
        use_mlp_bias=True,
    )
    block.eval()

    zero_module_parameters(block.attention)
    zero_module_parameters(block.mlp)

    x = make_block_input(B=2, T=8, D=64)
    out = block(x)

    assert torch.allclose(out, x, atol=0.0, rtol=0.0)


# ============================================================
# E. Inherited causality tests
# ============================================================

def test_block_changing_future_tokens_does_not_change_past_outputs():
    block = make_block(
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    block.eval()

    B, T, D = 2, 10, 64
    cut = 5

    x1 = make_block_input(B=B, T=T, D=D)
    x2 = x1.clone()
    x2[:, cut:, :] = torch.randn_like(x2[:, cut:, :])

    out1 = block(x1)
    out2 = block(x2)

    assert torch.allclose(
        out1[:, :cut, :],
        out2[:, :cut, :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_attention_mask_blocks_padding_keys_through_block():
    block = make_block(
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    block.eval()

    B, T, D = 2, 8, 64
    x = make_block_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T)
    attention_mask[0, 3] = 0
    attention_mask[1, 5] = 0

    _, aux = block(
        x,
        attention_mask=attention_mask,
        need_weights=True,
    )

    weights = aux["attn_weights"]

    assert torch.allclose(
        weights[0, :, :, 3],
        torch.zeros_like(weights[0, :, :, 3]),
        atol=0.0,
        rtol=0.0,
    )

    assert torch.allclose(
        weights[1, :, :, 5],
        torch.zeros_like(weights[1, :, :, 5]),
        atol=0.0,
        rtol=0.0,
    )


# ============================================================
# F. RoPE passthrough tests
# ============================================================

def test_start_pos_matches_explicit_position_ids_through_block():
    block = make_block(
        use_rope=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    block.eval()

    B, T, D = 2, 8, 64
    start_pos = 10

    x = make_block_input(B=B, T=T, D=D)

    out_start = block(x, start_pos=start_pos)
    out_explicit = block(
        x,
        position_ids=torch.arange(start_pos, start_pos + T),
    )

    assert torch.allclose(out_start, out_explicit, atol=1e-6, rtol=1e-5)


# ============================================================
# G. Dropout tests
# ============================================================

def test_block_dropout_zero_is_deterministic():
    block = make_block(
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    block.train()

    x = make_block_input(B=4, T=16, D=64)

    out1 = block(x)
    out2 = block(x)

    assert torch.equal(out1, out2)


def test_block_dropout_disabled_in_eval_mode():
    block = make_block(
        attention_dropout=0.5,
        residual_dropout=0.5,
        mlp_dropout=0.5,
    )
    block.eval()

    x = make_block_input(B=4, T=16, D=64)

    out1 = block(x)
    out2 = block(x)

    assert torch.equal(out1, out2)


def test_block_dropout_active_in_train_mode():
    block = make_block(
        attention_dropout=0.5,
        residual_dropout=0.5,
        mlp_dropout=0.5,
    )
    block.train()

    x = make_block_input(B=4, T=16, D=64)

    out1 = block(x)
    out2 = block(x)

    assert not torch.equal(out1, out2)


# ============================================================
# H. Gradient tests
# ============================================================

def test_block_backward_computes_gradients():
    block = make_block(
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
        use_attention_bias=True,
        use_mlp_bias=True,
    )

    x = make_block_input(B=2, T=8, D=64)
    x.requires_grad_(True)

    out = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ============================================================
# I. Integration tests
# ============================================================

def test_embedding_to_block_pipeline():
    B, T = 2, 16
    vocab_size = 128
    d_model = 64

    embedding = TokenEmbedding(
        EmbeddingConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_token_id=0,
            max_seq_len=T,
            embedding_dropout=0.0,
            scale_embeddings=False,
            init_std=0.02,
            tie_word_embeddings=True,
        )
    )

    block = make_block(
        d_model=d_model,
        n_heads=4,
        head_dim=16,
        rotary_dim=16,
        max_seq_len=T,
    )

    input_ids = torch.randint(1, vocab_size, (B, T), dtype=torch.long)

    hidden_states = embedding(input_ids)
    out = block(hidden_states)

    assert hidden_states.shape == (B, T, d_model)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_synthetic_dataset_batch_through_block():
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

    d_model = 64

    embedding = TokenEmbedding(
        EmbeddingConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            pad_token_id=tokenizer.pad_id,
            max_seq_len=data_cfg.block_size,
            embedding_dropout=0.0,
            scale_embeddings=False,
            init_std=0.02,
            tie_word_embeddings=True,
        )
    )

    block = make_block(
        d_model=d_model,
        n_heads=4,
        head_dim=16,
        rotary_dim=16,
        max_seq_len=data_cfg.block_size,
    )

    attention_mask = (input_ids != tokenizer.pad_id).long()

    hidden_states = embedding(input_ids)
    out = block(hidden_states, attention_mask=attention_mask)

    assert input_ids.shape == (data_cfg.batch_size, data_cfg.block_size)
    assert labels.shape == (data_cfg.batch_size, data_cfg.block_size)
    assert hidden_states.shape == (data_cfg.batch_size, data_cfg.block_size, d_model)
    assert out.shape == (data_cfg.batch_size, data_cfg.block_size, d_model)
    assert torch.isfinite(out).all()