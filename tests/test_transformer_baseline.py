# @title
# ============================================================
# MiniCausalLM baseline tests
# ============================================================

import pytest
import torch
import torch.nn.functional as F

from src.transformer_modules.transformer import *
from data.syntethic_long_context_retrieval import * 

# ============================================================
# Helpers
# ============================================================

def make_lm_config(**overrides):
    cfg = dict(
        vocab_size=128,
        d_model=64,
        n_layers=2,

        pad_token_id=0,
        max_seq_len=64,

        embedding_dropout=0.0,
        scale_embeddings=False,
        tie_word_embeddings=True,

        rms_norm_eps=1e-6,

        n_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,

        mlp_hidden_dim=256,
        mlp_expansion_factor=4.0,
        mlp_multiple_of=1,
        mlp_dropout=0.0,
        use_mlp_bias=False,

        init_std=0.02,
    )
    cfg.update(overrides)
    return MiniCausalLMConfig(**cfg)


def make_lm(**overrides):
    return MiniCausalLM(make_lm_config(**overrides))


def make_lm_batch(
    B=2,
    T=16,
    vocab_size=128,
    pad_token_id=0,
    include_pad=False,
):
    input_ids = torch.randint(
        low=1,
        high=vocab_size,
        size=(B, T),
        dtype=torch.long,
    )

    labels = torch.randint(
        low=1,
        high=vocab_size,
        size=(B, T),
        dtype=torch.long,
    )

    if include_pad:
        input_ids[0, -2:] = pad_token_id
        labels[0, -2:] = pad_token_id

    return input_ids, labels


# ============================================================
# A. Config tests
# ============================================================

def test_valid_minicausallm_config_builds():
    config = make_lm_config(
        vocab_size=128,
        d_model=256,
        n_layers=2,
        n_heads=4,
        head_dim=64,
        rotary_dim=64,
        mlp_hidden_dim=1024,
    )

    model = MiniCausalLM(config)

    assert model.vocab_size == 128
    assert model.d_model == 256
    assert model.n_layers == 2
    assert len(model.blocks) == 2


@pytest.mark.parametrize("vocab_size", [0, -1, -128])
def test_invalid_vocab_size_raises(vocab_size):
    with pytest.raises(ValueError):
        MiniCausalLM(make_lm_config(vocab_size=vocab_size))


@pytest.mark.parametrize("n_layers", [0, -1, -2])
def test_invalid_n_layers_raises(n_layers):
    with pytest.raises(ValueError):
        MiniCausalLM(make_lm_config(n_layers=n_layers))


@pytest.mark.parametrize("pad_token_id", [-1, 128])
def test_invalid_pad_token_id_raises(pad_token_id):
    with pytest.raises(ValueError):
        MiniCausalLM(make_lm_config(vocab_size=128, pad_token_id=pad_token_id))


# ============================================================
# B. Internal structure tests
# ============================================================

def test_model_has_expected_modules():
    config = make_lm_config(n_layers=3)
    model = MiniCausalLM(config)

    assert hasattr(model, "embedding")
    assert hasattr(model, "blocks")
    assert hasattr(model, "final_norm")
    assert hasattr(model, "lm_head")

    assert len(model.blocks) == config.n_layers


def test_blocks_are_transformerBlocks():
    model = make_lm(n_layers=2)

    for block in model.blocks:
        assert isinstance(block, TransformerBlock)


def test_final_norm_is_rmsnorm():
    model = make_lm()

    assert isinstance(model.final_norm, RMSNorm)


# ============================================================
# C. Weight tying tests
# ============================================================

def test_weight_tying_enabled_shares_parameter():
    model = make_lm(tie_word_embeddings=True)

    assert model.lm_head.weight is model.embedding.weight


def test_weight_tying_disabled_uses_independent_parameter():
    model = make_lm(tie_word_embeddings=False)

    assert model.lm_head.weight is not model.embedding.weight


# ============================================================
# D. Forward tests
# ============================================================

def test_logits_shape():
    config = make_lm_config(vocab_size=128, max_seq_len=16)
    model = MiniCausalLM(config)

    input_ids, _ = make_lm_batch(B=2, T=16, vocab_size=config.vocab_size)

    outputs = model(input_ids=input_ids)

    assert outputs["logits"].shape == (2, 16, config.vocab_size)


def test_forward_without_labels_returns_no_loss():
    model = make_lm()

    input_ids, _ = make_lm_batch(B=2, T=16)

    outputs = model(input_ids=input_ids, labels=None)

    assert outputs["loss"] is None


def test_forward_with_labels_returns_scalar_loss():
    model = make_lm()

    input_ids, labels = make_lm_batch(B=2, T=16)

    outputs = model(input_ids=input_ids, labels=labels)

    assert outputs["loss"] is not None
    assert outputs["loss"].dim() == 0
    assert torch.isfinite(outputs["loss"])


@pytest.mark.parametrize(
    "bad_input_ids",
    [
        torch.ones(16, dtype=torch.long),
        torch.ones(2, 16, 1, dtype=torch.long),
    ],
)
def test_rejects_wrong_input_ids_rank(bad_input_ids):
    model = make_lm()

    with pytest.raises(ValueError):
        model(input_ids=bad_input_ids)


def test_rejects_labels_wrong_shape():
    model = make_lm()

    input_ids, _ = make_lm_batch(B=2, T=16)
    bad_labels = torch.ones(2, 15, dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids=input_ids, labels=bad_labels)


def test_rejects_sequence_longer_than_max_seq_len():
    model = make_lm(max_seq_len=8)

    input_ids = torch.ones(2, 9, dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids=input_ids)


# ============================================================
# E. attention_mask tests
# ============================================================

def test_auto_attention_mask_from_pad_token():
    config = make_lm_config(
        pad_token_id=0,
        n_layers=2,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)
    model.eval()

    input_ids, labels = make_lm_batch(
        B=2,
        T=16,
        vocab_size=config.vocab_size,
        pad_token_id=config.pad_token_id,
        include_pad=True,
    )

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=None,
        need_weights=True,
    )

    pad_positions = input_ids == config.pad_token_id

    for layer_weights in outputs["aux"]["attn_weights"]:
        # layer_weights: [B, H, T, T]
        for b in range(input_ids.shape[0]):
            for s in torch.where(pad_positions[b])[0]:
                assert torch.allclose(
                    layer_weights[b, :, :, s],
                    torch.zeros_like(layer_weights[b, :, :, s]),
                    atol=0.0,
                    rtol=0.0,
                )


def test_explicit_attention_mask_is_used():
    config = make_lm_config(
        pad_token_id=0,
        n_layers=2,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)
    model.eval()

    B, T = 2, 16
    input_ids, labels = make_lm_batch(B=B, T=T, vocab_size=config.vocab_size)

    attention_mask = torch.ones(B, T, dtype=torch.long)
    blocked_position = 5
    attention_mask[:, blocked_position] = 0

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        need_weights=True,
    )

    for layer_weights in outputs["aux"]["attn_weights"]:
        assert torch.allclose(
            layer_weights[:, :, :, blocked_position],
            torch.zeros_like(layer_weights[:, :, :, blocked_position]),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.ones(16, dtype=torch.long),
        torch.ones(2, 16, 1, dtype=torch.long),
        torch.ones(2, 17, dtype=torch.long),
    ],
)
def test_invalid_attention_mask_shape_raises(bad_mask):
    model = make_lm(max_seq_len=16)

    input_ids, labels = make_lm_batch(B=2, T=16)

    with pytest.raises(ValueError):
        model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=bad_mask,
        )


# ============================================================
# F. Loss tests
# ============================================================

def test_loss_matches_manual_cross_entropy():
    config = make_lm_config(
        pad_token_id=0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)
    model.eval()

    input_ids, labels = make_lm_batch(
        B=2,
        T=16,
        vocab_size=config.vocab_size,
        pad_token_id=config.pad_token_id,
        include_pad=True,
    )

    outputs = model(input_ids=input_ids, labels=labels)

    logits = outputs["logits"]

    manual_loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        labels.reshape(-1),
        ignore_index=config.pad_token_id,
    )

    assert torch.allclose(outputs["loss"], manual_loss, atol=1e-6, rtol=1e-6)


def test_loss_ignores_pad_token_id():
    config = make_lm_config(
        pad_token_id=0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)
    model.eval()

    input_ids, labels = make_lm_batch(
        B=2,
        T=16,
        vocab_size=config.vocab_size,
    )

    labels[0, 0] = config.pad_token_id
    labels[1, 3] = config.pad_token_id

    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs["logits"]

    manual_loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        labels.reshape(-1),
        ignore_index=config.pad_token_id,
    )

    assert torch.allclose(outputs["loss"], manual_loss, atol=1e-6, rtol=1e-6)


# ============================================================
# G. Aux / attention weights tests
# ============================================================

def test_need_weights_returns_attention_weights_per_layer():
    config = make_lm_config(n_layers=3)
    model = MiniCausalLM(config)

    input_ids, _ = make_lm_batch(B=2, T=16, vocab_size=config.vocab_size)

    outputs = model(input_ids=input_ids, need_weights=True)

    assert "attn_weights" in outputs["aux"]
    assert isinstance(outputs["aux"]["attn_weights"], list)
    assert len(outputs["aux"]["attn_weights"]) == config.n_layers

    for weights in outputs["aux"]["attn_weights"]:
        assert weights.shape == (2, config.n_heads, 16, 16)


def test_need_weights_false_does_not_store_attention_weights():
    model = make_lm()

    input_ids, _ = make_lm_batch(B=2, T=16)

    outputs = model(input_ids=input_ids, need_weights=False)

    assert "attn_weights" not in outputs["aux"]


# ============================================================
# H. End-to-end causality tests
# ============================================================

def test_changing_future_tokens_does_not_change_past_logits():
    config = make_lm_config(
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)
    model.eval()

    B, T = 2, 16
    cut = 8

    input_ids_1, _ = make_lm_batch(
        B=B,
        T=T,
        vocab_size=config.vocab_size,
    )

    input_ids_2 = input_ids_1.clone()

    input_ids_2[:, cut:] = torch.randint(
        low=1,
        high=config.vocab_size,
        size=input_ids_2[:, cut:].shape,
        dtype=torch.long,
    )

    logits_1 = model(input_ids=input_ids_1)["logits"]
    logits_2 = model(input_ids=input_ids_2)["logits"]

    assert torch.allclose(
        logits_1[:, :cut, :],
        logits_2[:, :cut, :],
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# I. Gradient tests
# ============================================================

def test_minicausallm_backward_computes_gradients():
    config = make_lm_config(
        tie_word_embeddings=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)

    input_ids, labels = make_lm_batch(
        B=2,
        T=16,
        vocab_size=config.vocab_size,
    )

    outputs = model(input_ids=input_ids, labels=labels)

    loss = outputs["loss"]

    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()

    assert model.final_norm.weight.grad is not None
    assert torch.isfinite(model.final_norm.weight.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_minicausallm_backward_computes_lm_head_grad_when_not_tied():
    config = make_lm_config(
        tie_word_embeddings=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    model = MiniCausalLM(config)

    input_ids, labels = make_lm_batch(
        B=2,
        T=16,
        vocab_size=config.vocab_size,
    )

    outputs = model(input_ids=input_ids, labels=labels)
    outputs["loss"].backward()

    assert model.lm_head.weight is not model.embedding.weight
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


# ============================================================
# J. Synthetic dataset integration tests
# ============================================================

def test_synthetic_dataset_batch_forward_loss():
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
    batch = normalize_lm_batch(batch)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    config = MiniCausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=data_cfg.block_size,
        embedding_dropout=0.0,
        scale_embeddings=False,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        n_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,
        mlp_hidden_dim=256,
        mlp_expansion_factor=4.0,
        mlp_multiple_of=1,
        mlp_dropout=0.0,
        use_mlp_bias=False,
        init_std=0.02,
    )

    model = MiniCausalLM(config)

    outputs = model(input_ids=input_ids, labels=labels)

    assert outputs["logits"].shape == (
        data_cfg.batch_size,
        data_cfg.block_size,
        tokenizer.vocab_size,
    )
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])


def test_synthetic_dataset_batch_backward():
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
    batch = normalize_lm_batch(batch)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    config = MiniCausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=data_cfg.block_size,
        embedding_dropout=0.0,
        scale_embeddings=False,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        n_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,
        mlp_hidden_dim=256,
        mlp_expansion_factor=4.0,
        mlp_multiple_of=1,
        mlp_dropout=0.0,
        use_mlp_bias=False,
        init_std=0.02,
    )

    model = MiniCausalLM(config)

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]

    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"