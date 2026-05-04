# @title
# ============================================================
# SwiGLU / MLP baseline tests
# ============================================================

import pytest
import torch
import torch.nn.functional as F

from src.transformer_modules.SwiGLU import * 
from src.transformer_modules.RMSNorm import *
from src.transformer_modules.mha_baseline import *
from src.transformer_modules.embedding_module import *
# ============================================================
# Helpers
# ============================================================

def make_mlp_config(**overrides):
    cfg = dict(
        d_model=64,
        hidden_dim=256,
        expansion_factor=4.0,
        multiple_of=1,
        dropout=0.0,
        use_bias=False,
        init_std=0.02,
    )
    cfg.update(overrides)
    return SwiGLUMLPConfig(**cfg)


def make_mlp(**overrides):
    return SwiGLUMLP(make_mlp_config(**overrides))


def make_mlp_input(B=2, T=8, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


# ============================================================
# A. Config tests
# ============================================================

def test_valid_mlp_config_builds():
    config = make_mlp_config(
        d_model=256,
        hidden_dim=1024,
        dropout=0.0,
    )
    mlp = SwiGLUMLP(config)

    assert mlp.d_model == 256
    assert mlp.hidden_dim == 1024


def test_hidden_dim_inferred_from_expansion_factor():
    config = make_mlp_config(
        d_model=256,
        hidden_dim=None,
        expansion_factor=4.0,
        multiple_of=1,
    )
    mlp = SwiGLUMLP(config)

    assert mlp.hidden_dim == 1024


def test_hidden_dim_rounds_to_multiple_of():
    config = make_mlp_config(
        d_model=250,
        hidden_dim=None,
        expansion_factor=4.0,
        multiple_of=64,
    )
    mlp = SwiGLUMLP(config)

    raw_hidden_dim = int(4.0 * 250)

    assert mlp.hidden_dim % 64 == 0
    assert mlp.hidden_dim >= raw_hidden_dim


@pytest.mark.parametrize("d_model", [0, -1, -256])
def test_invalid_d_model_raises(d_model):
    with pytest.raises(ValueError):
        SwiGLUMLP(make_mlp_config(d_model=d_model))


@pytest.mark.parametrize("hidden_dim", [0, -1, -256])
def test_invalid_hidden_dim_raises(hidden_dim):
    with pytest.raises(ValueError):
        SwiGLUMLP(make_mlp_config(hidden_dim=hidden_dim))


@pytest.mark.parametrize("expansion_factor", [0.0, -1.0, -4.0])
def test_invalid_expansion_factor_raises(expansion_factor):
    with pytest.raises(ValueError):
        SwiGLUMLP(
            make_mlp_config(
                hidden_dim=None,
                expansion_factor=expansion_factor,
            )
        )


@pytest.mark.parametrize("multiple_of", [0, -1, -64])
def test_invalid_multiple_of_raises(multiple_of):
    with pytest.raises(ValueError):
        SwiGLUMLP(make_mlp_config(multiple_of=multiple_of))


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_raises(dropout):
    with pytest.raises(ValueError):
        SwiGLUMLP(make_mlp_config(dropout=dropout))


@pytest.mark.parametrize("init_std", [0.0, -0.01])
def test_invalid_init_std_raises(init_std):
    with pytest.raises(ValueError):
        SwiGLUMLP(make_mlp_config(init_std=init_std))


# ============================================================
# B. Internal structure tests
# ============================================================

def test_projection_shapes():
    config = make_mlp_config(d_model=64, hidden_dim=256)
    mlp = SwiGLUMLP(config)

    assert mlp.gate_proj.weight.shape == (256, 64)
    assert mlp.up_proj.weight.shape == (256, 64)
    assert mlp.down_proj.weight.shape == (64, 256)


def test_bias_absent_when_use_bias_false():
    mlp = make_mlp(use_bias=False)

    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


def test_bias_present_when_use_bias_true():
    mlp = make_mlp(use_bias=True)

    assert mlp.gate_proj.bias is not None
    assert mlp.up_proj.bias is not None
    assert mlp.down_proj.bias is not None


def test_weights_are_finite_after_init():
    mlp = make_mlp()

    assert torch.isfinite(mlp.gate_proj.weight).all()
    assert torch.isfinite(mlp.up_proj.weight).all()
    assert torch.isfinite(mlp.down_proj.weight).all()


def test_bias_initialized_to_zero_when_present():
    mlp = make_mlp(use_bias=True)

    assert torch.allclose(
        mlp.gate_proj.bias,
        torch.zeros_like(mlp.gate_proj.bias),
    )
    assert torch.allclose(
        mlp.up_proj.bias,
        torch.zeros_like(mlp.up_proj.bias),
    )
    assert torch.allclose(
        mlp.down_proj.bias,
        torch.zeros_like(mlp.down_proj.bias),
    )


# ============================================================
# C. Forward tests
# ============================================================

def test_mlp_output_shape_matches_input_shape():
    mlp = make_mlp(d_model=64, hidden_dim=256)

    x = make_mlp_input(B=2, T=8, D=64)
    out = mlp(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(8, 64),
        torch.randn(2, 8, 64, 1),
    ],
)
def test_rejects_wrong_input_rank(bad_x):
    mlp = make_mlp(d_model=64, hidden_dim=256)

    with pytest.raises(ValueError):
        mlp(bad_x)


def test_mlp_rejects_wrong_hidden_size():
    mlp = make_mlp(d_model=64, hidden_dim=256)

    x = torch.randn(2, 8, 32)

    with pytest.raises(ValueError):
        mlp(x)


def test_forward_matches_manual_swiglu_computation():
    mlp = make_mlp(
        d_model=64,
        hidden_dim=256,
        dropout=0.0,
        use_bias=True,
    )
    mlp.eval()

    x = make_mlp_input(B=2, T=8, D=64)

    out = mlp(x)

    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    hidden = F.silu(gate) * up
    expected = mlp.down_proj(hidden)

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_mlp_output_is_finite():
    mlp = make_mlp()

    x = make_mlp_input(B=2, T=8, D=64)
    out = mlp(x)

    assert torch.isfinite(out).all()


# ============================================================
# D. Dropout tests
# ============================================================

def test_mlp_dropout_zero_is_deterministic():
    mlp = make_mlp(dropout=0.0)
    mlp.train()

    x = make_mlp_input(B=4, T=16, D=64)

    out1 = mlp(x)
    out2 = mlp(x)

    assert torch.equal(out1, out2)


def test_mlp_dropout_disabled_in_eval_mode():
    mlp = make_mlp(dropout=0.5)
    mlp.eval()

    x = make_mlp_input(B=4, T=16, D=64)

    out1 = mlp(x)
    out2 = mlp(x)

    assert torch.equal(out1, out2)


def test_mlp_dropout_active_in_train_mode():
    mlp = make_mlp(dropout=0.5)
    mlp.train()

    x = make_mlp_input(B=4, T=16, D=64)

    out1 = mlp(x)
    out2 = mlp(x)

    assert not torch.equal(out1, out2)


# ============================================================
# E. Gradient tests
# ============================================================

def test_mlp_backward_computes_gradients():
    mlp = make_mlp(dropout=0.0)

    x = make_mlp_input(B=2, T=8, D=64)
    x.requires_grad_(True)

    out = mlp(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(mlp.gate_proj.weight.grad).all()
    assert torch.isfinite(mlp.up_proj.weight.grad).all()
    assert torch.isfinite(mlp.down_proj.weight.grad).all()


def test_all_parameters_receive_gradients():
    mlp = make_mlp(dropout=0.0, use_bias=True)

    x = make_mlp_input(B=2, T=8, D=64)
    out = mlp(x)
    loss = out.sum()
    loss.backward()

    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ============================================================
# F. Dtype and device tests
# ============================================================

def test_mlp_output_dtype_matches_input_dtype_float32():
    mlp = make_mlp()

    x = make_mlp_input(B=2, T=8, D=64, dtype=torch.float32)
    out = mlp(x)

    assert out.dtype == torch.float32


def test_mlp_output_dtype_matches_input_dtype_bfloat16():
    mlp = make_mlp().to(dtype=torch.bfloat16)

    x = make_mlp_input(B=2, T=8, D=64).to(dtype=torch.bfloat16)
    out = mlp(x)

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_mlp_output_device_matches_input_device():
    mlp = make_mlp()

    x = make_mlp_input(B=2, T=8, D=64)
    out = mlp(x)

    assert out.device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mlp_runs_on_cuda_if_available():
    mlp = make_mlp().cuda()

    x = make_mlp_input(B=2, T=8, D=64, device="cuda")
    out = mlp(x)

    assert out.device.type == "cuda"
    assert out.shape == x.shape


# ============================================================
# Integration tests
# ============================================================

def test_embedding_rmsnorm_mlp_pipeline():
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
        tie_word_embeddings=True,
    )

    embedding = TokenEmbedding(emb_cfg)
    norm = RMSNorm(dim=d_model)
    mlp = make_mlp(d_model=d_model, hidden_dim=256)

    input_ids = torch.randint(1, vocab_size, (B, T), dtype=torch.long)

    hidden_states = embedding(input_ids)
    normed = norm(hidden_states)
    out = mlp(normed)

    assert hidden_states.shape == (B, T, d_model)
    assert normed.shape == (B, T, d_model)
    assert out.shape == (B, T, d_model)

    assert torch.isfinite(hidden_states).all()
    assert torch.isfinite(normed).all()
    assert torch.isfinite(out).all()


def test_attention_mlp_pipeline():
    B, T, d_model = 2, 16, 64

    norm1 = RMSNorm(dim=d_model)
    attn = CausalMultiHeadAttention(
        CausalMHAConfig(
            d_model=d_model,
            n_heads=4,
            head_dim=16,
            attention_dropout=0.0,
            residual_dropout=0.0,
            use_bias=False,
            use_rope=True,
            rope_theta=10000.0,
            rotary_dim=16,
            max_seq_len=T,
            init_std=0.02,
        ))

    norm2 = RMSNorm(dim=d_model)
    mlp = make_mlp(d_model=d_model, hidden_dim=256)

    x = torch.randn(B, T, d_model)

    normed_1 = norm1(x)
    attn_out = attn(normed_1)

    normed_2 = norm2(attn_out)
    mlp_out = mlp(normed_2)

    assert attn_out.shape == (B, T, d_model)
    assert mlp_out.shape == (B, T, d_model)

    assert torch.isfinite(attn_out).all()
    assert torch.isfinite(mlp_out).all()