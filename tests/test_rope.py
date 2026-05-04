from src.transformer_modules.rope import *


# ============================================================
# RoPE utilities tests
# Tests for rotate_half and RotaryEmbedding
# ============================================================

import pytest
import torch


# ============================================================
# A. rotate_half tests
# ============================================================

def test_rotate_half_shape():
    x = torch.randn(2, 4, 3, 8)
    y = rotate_half(x)

    assert y.shape == x.shape


def test_rotate_half_manual_values():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = rotate_half(x)

    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])

    assert torch.equal(y, expected)


def test_rotate_half_preserves_dtype_and_device():
    x = torch.randn(2, 4, 3, 8, dtype=torch.float32)
    y = rotate_half(x)

    assert y.dtype == x.dtype
    assert y.device == x.device


# ============================================================
# B. Config tests
# ============================================================

def test_valid_rope_config_builds():
    rope = RotaryEmbedding(dim=64, rotary_dim=64, base=10000.0)

    assert rope.dim == 64
    assert rope.rotary_dim == 64
    assert rope.base == 10000.0
    assert rope.inv_freq.shape == (32,)


def test_rotary_dim_none_defaults_to_dim():
    rope = RotaryEmbedding(dim=64, rotary_dim=None)

    assert rope.rotary_dim == 64


@pytest.mark.parametrize("dim", [0, -1, -64])
def test_invalid_dim_raises(dim):
    with pytest.raises(ValueError):
        RotaryEmbedding(dim=dim)


@pytest.mark.parametrize(
    "dim,rotary_dim",
    [
        (64, 0),
        (64, -1),
        (64, 65),
        (64, 7),
    ],
)
def test_invalid_rotary_dim_raises(dim, rotary_dim):
    with pytest.raises(ValueError):
        RotaryEmbedding(dim=dim, rotary_dim=rotary_dim)


@pytest.mark.parametrize("base", [0.0, -1.0, -10000.0])
def test_invalid_base_raises(base):
    with pytest.raises(ValueError):
        RotaryEmbedding(dim=64, rotary_dim=64, base=base)


# ============================================================
# C. Forward and shape tests
# ============================================================

def test_rope_output_shape_matches_input_shape():
    rope = RotaryEmbedding(dim=32, rotary_dim=32)

    x = torch.randn(2, 8, 4, 32)
    y = rope(x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(2, 8, 32),
        torch.randn(8, 32),
        torch.randn(2, 8, 4, 32, 1),
    ],
)
def test_rejects_non_4d_input(bad_x):
    rope = RotaryEmbedding(dim=32, rotary_dim=32)

    with pytest.raises(ValueError):
        rope(bad_x)


def test_rejects_wrong_last_dim():
    rope = RotaryEmbedding(dim=64, rotary_dim=64)

    x = torch.randn(2, 8, 4, 32)

    with pytest.raises(ValueError):
        rope(x)


# ============================================================
# D. Mathematical property tests
# ============================================================

def test_position_zero_is_identity():
    rope = RotaryEmbedding(dim=32, rotary_dim=32)

    x = torch.randn(2, 8, 4, 32)
    y = rope(x)

    assert torch.allclose(y[:, 0, :, :], x[:, 0, :, :], atol=1e-6, rtol=1e-6)


def test_rotary_preserves_norm_of_rotated_part():
    D = 32
    rotary_dim = 16

    rope = RotaryEmbedding(dim=D, rotary_dim=rotary_dim)

    x = torch.randn(2, 8, 4, D)
    y = rope(x)

    x_rot = x[..., D - rotary_dim:]
    y_rot = y[..., D - rotary_dim:]

    x_norm = torch.linalg.vector_norm(x_rot, dim=-1)
    y_norm = torch.linalg.vector_norm(y_rot, dim=-1)

    assert torch.allclose(x_norm, y_norm, atol=1e-5, rtol=1e-5)


def test_partial_rope_leaves_pass_through_dimensions_unchanged():
    D = 16
    rotary_dim = 8

    rope = RotaryEmbedding(dim=D, rotary_dim=rotary_dim)

    x = torch.randn(2, 8, 4, D)
    y = rope(x)

    pass_dim = D - rotary_dim

    assert torch.allclose(
        y[..., :pass_dim],
        x[..., :pass_dim],
        atol=0.0,
        rtol=0.0,
    )


def test_full_rope_rotates_all_dimensions():
    D = 16

    rope = RotaryEmbedding(dim=D, rotary_dim=D)

    x = torch.randn(2, 8, 4, D)
    y = rope(x)

    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)

    assert y.shape == x.shape
    assert torch.allclose(x_norm, y_norm, atol=1e-5, rtol=1e-5)


def test_forward_matches_manual_computation():
    B, T, H, D = 2, 5, 3, 16
    rotary_dim = 8
    pass_dim = D - rotary_dim

    rope = RotaryEmbedding(dim=D, rotary_dim=rotary_dim)

    x = torch.randn(B, T, H, D)
    y = rope(x)

    x_pass = x[..., :pass_dim]
    x_rot = x[..., pass_dim:]

    position_ids = torch.arange(T, dtype=torch.float32)

    inv_freq = rope.inv_freq.float()
    freqs = position_ids[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(emb)[None, :, None, :].to(dtype=x.dtype)
    sin = torch.sin(emb)[None, :, None, :].to(dtype=x.dtype)

    expected_rot = x_rot * cos + rotate_half(x_rot) * sin
    expected = torch.cat((x_pass, expected_rot), dim=-1)

    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)


# ============================================================
# E. Position tests
# ============================================================

def test_start_pos_matches_explicit_position_ids():
    B, T, H, D = 2, 6, 3, 16
    start_pos = 10

    rope = RotaryEmbedding(dim=D, rotary_dim=D)

    x = torch.randn(B, T, H, D)

    y_start = rope(x, position_ids=None, start_pos=start_pos)

    explicit_ids = torch.arange(start_pos, start_pos + T)
    y_explicit = rope(x, position_ids=explicit_ids)

    assert torch.allclose(y_start, y_explicit, atol=1e-6, rtol=1e-5)


def test_accepts_position_ids_T():
    B, T, H, D = 2, 6, 3, 16

    rope = RotaryEmbedding(dim=D, rotary_dim=D)

    x = torch.randn(B, T, H, D)
    position_ids = torch.arange(T)

    y = rope(x, position_ids=position_ids)

    assert y.shape == x.shape


def test_accepts_position_ids_BT():
    B, T, H, D = 2, 6, 3, 16

    rope = RotaryEmbedding(dim=D, rotary_dim=D)

    x = torch.randn(B, T, H, D)

    position_ids = torch.stack(
        [
            torch.arange(T),
            torch.arange(10, 10 + T),
        ],
        dim=0,
    )

    y = rope(x, position_ids=position_ids)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    "position_ids",
    [
        torch.arange(2),                       # [B], wrong when T != B
        torch.zeros(2, 6, 1, dtype=torch.long), # [B,T,1]
        torch.arange(7),                       # [T+1]
    ],
)
def test_rejects_invalid_position_ids_shape(position_ids):
    B, T, H, D = 2, 6, 3, 16

    rope = RotaryEmbedding(dim=D, rotary_dim=D)
    x = torch.randn(B, T, H, D)

    with pytest.raises(ValueError):
        rope(x, position_ids=position_ids)


def test_accepts_negative_position_ids():
    B, T, H, D = 2, 4, 3, 16

    rope = RotaryEmbedding(dim=D, rotary_dim=8)

    x = torch.randn(B, T, H, D)
    position_ids = torch.tensor([-3, -2, -1, 0])

    y = rope(x, position_ids=position_ids)

    x_rot = x[..., D - rope.rotary_dim:]
    y_rot = y[..., D - rope.rotary_dim:]

    x_norm = torch.linalg.vector_norm(x_rot, dim=-1)
    y_norm = torch.linalg.vector_norm(y_rot, dim=-1)

    assert y.shape == x.shape
    assert torch.allclose(x_norm, y_norm, atol=1e-5, rtol=1e-5)


# ============================================================
# F. Dtype and device tests
# ============================================================

def test_output_dtype_matches_input_dtype_float32():
    rope = RotaryEmbedding(dim=32, rotary_dim=32)

    x = torch.randn(2, 8, 4, 32, dtype=torch.float32)
    y = rope(x)

    assert y.dtype == torch.float32


def test_output_dtype_matches_input_dtype_bfloat16():
    rope = RotaryEmbedding(dim=32, rotary_dim=32)

    x = torch.randn(2, 8, 4, 32, dtype=torch.bfloat16)
    y = rope(x)

    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_output_dtype_matches_input_dtype_float16_cuda_if_available():
    rope = RotaryEmbedding(dim=32, rotary_dim=32).cuda()

    x = torch.randn(2, 8, 4, 32, device="cuda", dtype=torch.float16)
    y = rope(x)

    assert y.dtype == torch.float16
    assert y.device.type == "cuda"
    assert torch.isfinite(y.float()).all()


def test_output_device_matches_input_device():
    rope = RotaryEmbedding(dim=32, rotary_dim=32)

    x = torch.randn(2, 8, 4, 32)
    y = rope(x)

    assert y.device == x.device


# ============================================================
# G. Gradient tests
# ============================================================

def test_backward_computes_gradient_for_x():
    rope = RotaryEmbedding(dim=32, rotary_dim=16)

    x = torch.randn(2, 8, 4, 32, requires_grad=True)
    y = rope(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()