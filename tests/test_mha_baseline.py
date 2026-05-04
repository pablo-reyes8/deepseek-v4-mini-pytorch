# @title
# ============================================================
# Causal Multi-Head Attention baseline tests
# ============================================================

import pytest
import torch

from src.transformer_modules.mha_baseline import *
from src.transformer_modules.embedding_module import * 
from src.transformer_modules.RMSNorm import *
# ============================================================
# Helpers
# ============================================================

def make_mha_config(**overrides):
    cfg = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,
        max_seq_len=128,
        init_std=0.02,
    )
    cfg.update(overrides)
    return CausalMHAConfig(**cfg)


def make_mha(**overrides):
    return CausalMultiHeadAttention(make_mha_config(**overrides))


def make_input(B=2, T=8, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)



# ============================================================
# A. Config tests
# ============================================================

def test_valid_attention_config_builds():
    config = make_mha_config(d_model=256, n_heads=4, head_dim=64)
    attn = CausalMultiHeadAttention(config)

    assert attn.d_model == 256
    assert attn.n_heads == 4
    assert attn.head_dim == 64
    assert attn.inner_dim == 256


@pytest.mark.parametrize("d_model", [0, -1, -256])
def test_invalid_d_model_raises(d_model):
    with pytest.raises(ValueError):
        CausalMultiHeadAttention(make_mha_config(d_model=d_model))


@pytest.mark.parametrize("n_heads", [0, -1, -4])
def test_invalid_n_heads_raises(n_heads):
    with pytest.raises(ValueError):
        CausalMultiHeadAttention(make_mha_config(n_heads=n_heads))


@pytest.mark.parametrize(
    "head_dim",
    [0, -1, 15],
)
def test_invalid_head_dim_raises(head_dim):
    with pytest.raises(ValueError):
        CausalMultiHeadAttention(make_mha_config(head_dim=head_dim))


@pytest.mark.parametrize(
    "field,value",
    [
        ("attention_dropout", -0.1),
        ("attention_dropout", 1.0),
        ("attention_dropout", 1.5),
        ("residual_dropout", -0.1),
        ("residual_dropout", 1.0),
        ("residual_dropout", 1.5),
    ],
)
def test_invalid_dropout_raises(field, value):
    with pytest.raises(ValueError):
        CausalMultiHeadAttention(make_mha_config(**{field: value}))


@pytest.mark.parametrize("rotary_dim", [0, -1, 17, 32])
def test_invalid_rotary_dim_raises(rotary_dim):
    with pytest.raises(ValueError):
        CausalMultiHeadAttention(make_mha_config(rotary_dim=rotary_dim))


# ============================================================
# B. Shape and forward tests
# ============================================================

def test_mha_output_shape_matches_input():
    attn = make_mha()
    x = make_input(B=2, T=8, D=64)

    out = attn(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(8, 64),
        torch.randn(2, 8, 64, 1),
    ],
)
def test_rejects_wrong_input_dim(bad_x):
    attn = make_mha()

    with pytest.raises(ValueError):
        attn(bad_x)


def test_rejects_wrong_hidden_size():
    attn = make_mha(d_model=64)
    x = torch.randn(2, 8, 32)

    with pytest.raises(ValueError):
        attn(x)


def test_returns_attention_weights_when_requested():
    attn = make_mha()
    x = make_input(B=2, T=8, D=64)

    out, weights = attn(x, need_weights=True)

    assert out.shape == (2, 8, 64)
    assert weights.shape == (2, 4, 8, 8)


def test_attention_weights_sum_to_one_without_dropout():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    x = make_input(B=2, T=8, D=64)

    _, weights = attn(x, need_weights=True)

    sums = weights.sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)


# ============================================================
# C. Causality tests
# ============================================================

def test_causal_mask_blocks_future_attention_weights():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 8, 64
    x = make_input(B=B, T=T, D=D)

    _, weights = attn(x, need_weights=True)

    future_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool),
        diagonal=1,
    )

    future_weights = weights[:, :, future_mask]

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=0.0,
        rtol=0.0,
    )


def test_changing_future_tokens_does_not_change_past_outputs():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 10, 64
    cut = 5

    x1 = make_input(B=B, T=T, D=D)
    x2 = x1.clone()

    x2[:, cut:, :] = torch.randn_like(x2[:, cut:, :])

    out1 = attn(x1)
    out2 = attn(x2)

    assert torch.allclose(
        out1[:, :cut, :],
        out2[:, :cut, :],
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# D. attention_mask / padding tests
# ============================================================

def test_attention_mask_shape_validation_accepts_BT():
    attn = make_mha()
    x = make_input(B=2, T=8, D=64)
    attention_mask = torch.ones(2, 8)

    out = attn(x, attention_mask=attention_mask)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.ones(8),
        torch.ones(2, 8, 1),
        torch.ones(2, 9),
    ],
)
def test_attention_mask_shape_validation_rejects_bad_shapes(bad_mask):
    attn = make_mha()
    x = make_input(B=2, T=8, D=64)

    with pytest.raises(ValueError):
        attn(x, attention_mask=bad_mask)


def test_attention_mask_blocks_padding_keys():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 8, 64
    x = make_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T)
    attention_mask[0, 3] = 0
    attention_mask[1, 5] = 0

    _, weights = attn(
        x,
        attention_mask=attention_mask,
        need_weights=True,
    )

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


def test_all_padding_keys_does_not_produce_nan():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 8, 64
    x = make_input(B=B, T=T, D=D)

    attention_mask = torch.zeros(B, T)

    out, weights = attn(
        x,
        attention_mask=attention_mask,
        need_weights=True,
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()


# ============================================================
# E. RoPE integration tests
# ============================================================



def test_no_rope_when_disabled():
    attn = make_mha(use_rope=False, attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 8, 64
    x = make_input(B=B, T=T, D=D)

    out1 = attn(x, position_ids=torch.arange(T), start_pos=0)
    out2 = attn(x, position_ids=torch.arange(10, 10 + T), start_pos=10)

    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)


def test_mha_start_pos_matches_explicit_position_ids():
    attn = make_mha(use_rope=True, attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 8, 64
    start_pos = 10

    x = make_input(B=B, T=T, D=D)

    out_start = attn(x, start_pos=start_pos)
    out_explicit = attn(x, position_ids=torch.arange(start_pos, start_pos + T))

    assert torch.allclose(out_start, out_explicit, atol=1e-6, rtol=1e-5)


def test_accepts_position_ids_T_and_BT():
    attn = make_mha(use_rope=True)

    B, T, D = 2, 8, 64
    x = make_input(B=B, T=T, D=D)

    position_ids_T = torch.arange(T)
    out_T = attn(x, position_ids=position_ids_T)

    position_ids_BT = torch.stack(
        [
            torch.arange(T),
            torch.arange(10, 10 + T),
        ],
        dim=0,
    )
    out_BT = attn(x, position_ids=position_ids_BT)

    assert out_T.shape == x.shape
    assert out_BT.shape == x.shape


def test_nonuniform_position_ids_change_output_when_rope_enabled():
    attn = make_mha(use_rope=True, attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    B, T, D = 2, 8, 64
    x = make_input(B=B, T=T, D=D)

    position_ids_1 = torch.arange(T)
    position_ids_2 = torch.arange(T) * 2

    out1 = attn(x, position_ids=position_ids_1)
    out2 = attn(x, position_ids=position_ids_2)

    assert not torch.allclose(out1, out2, atol=1e-7, rtol=1e-7)


# ============================================================
# F. Dropout tests
# ============================================================

def test_mha_dropout_zero_is_deterministic():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)
    attn.train()

    x = make_input(B=4, T=16, D=64)

    out1 = attn(x)
    out2 = attn(x)

    assert torch.equal(out1, out2)


def test_mha_dropout_disabled_in_eval_mode():
    attn = make_mha(attention_dropout=0.5, residual_dropout=0.5)
    attn.eval()

    x = make_input(B=4, T=16, D=64)

    out1 = attn(x)
    out2 = attn(x)

    assert torch.equal(out1, out2)


def test_mha_dropout_active_in_train_mode():
    attn = make_mha(attention_dropout=0.5, residual_dropout=0.5)
    attn.train()

    x = make_input(B=4, T=16, D=64)

    out1 = attn(x)
    out2 = attn(x)

    assert not torch.equal(out1, out2)


# ============================================================
# G. Gradient and dtype tests
# ============================================================

def test_mha_backward_computes_gradients():
    attn = make_mha(attention_dropout=0.0, residual_dropout=0.0)

    x = make_input(B=2, T=8, D=64)
    x.requires_grad_(True)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
    assert attn.out_proj.weight.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(attn.q_proj.weight.grad).all()
    assert torch.isfinite(attn.k_proj.weight.grad).all()
    assert torch.isfinite(attn.v_proj.weight.grad).all()
    assert torch.isfinite(attn.out_proj.weight.grad).all()


def test_mha_output_dtype_matches_input_dtype_float32():
    attn = make_mha()

    x = make_input(B=2, T=8, D=64, dtype=torch.float32)
    out = attn(x)

    assert out.dtype == torch.float32


def test_mha_output_dtype_matches_input_dtype_bfloat16():
    attn = make_mha().to(dtype=torch.bfloat16)

    x = make_input(B=2, T=8, D=64).to(dtype=torch.bfloat16)
    out = attn(x)

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


# ============================================================
# H. Integration tests with closed modules
# ============================================================

def test_embedding_rmsnorm_attention_pipeline():
    B, T = 2, 16
    vocab_size = 128
    d_model = 64

    emb_cfg = EmbeddingConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        pad_token_id=0,
        max_seq_len=T,
        embedding_dropout=0.0,
        scale_embeddings=False,
        init_std=0.02,
        tie_word_embeddings=True)

    embedding = TokenEmbedding(emb_cfg)
    norm = RMSNorm(dim=d_model)
    attn = make_mha(d_model=d_model, n_heads=4, head_dim=16, max_seq_len=T)

    input_ids = torch.randint(1, vocab_size, (B, T), dtype=torch.long)

    hidden_states = embedding(input_ids)
    normed = norm(hidden_states)
    out = attn(normed)

    assert hidden_states.shape == (B, T, d_model)
    assert normed.shape == (B, T, d_model)
    assert out.shape == (B, T, d_model)

    assert torch.isfinite(hidden_states).all()
    assert torch.isfinite(normed).all()
    assert torch.isfinite(out).all()


class SpyRoPE(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.call_count = 0

    def forward(self, x, position_ids=None, start_pos=0):
        self.call_count += 1
        return self.rope(x, position_ids=position_ids, start_pos=start_pos)


def test_rope_is_called_when_enabled():
    attn = make_mha(use_rope=True, attention_dropout=0.0, residual_dropout=0.0)
    attn.eval()

    attn.rope = SpyRoPE(attn.rope)

    x = make_input(B=2, T=8, D=64)

    _ = attn(x, position_ids=torch.arange(8))

    # RoPE debe llamarse para q y k. No para v.
    assert attn.rope.call_count == 2
