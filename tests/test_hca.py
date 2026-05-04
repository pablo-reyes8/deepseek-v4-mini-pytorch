# @title
# ============================================================
# HCAAttention tests
# ============================================================

import math
import pytest
import torch
import torch.nn as nn

from src.deepseek_hca_attention import *
from src.transformer_modules.RMSNorm import RMSNorm

# ============================================================
# Helpers
# ============================================================

def make_hca_config(**overrides):
    cfg = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        compression_factor=4,
        window_size=4,
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
    return HCAConfig(**cfg)


def make_hca(**overrides):
    return HCAAttention(make_hca_config(**overrides))


def make_hca_input(B=2, T=16, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


def make_compressor(compression_factor=4, head_dim=16):
    return HCATokenCompressor(
        compression_factor=compression_factor,
        head_dim=head_dim,
        init_std=0.02,
    )


# ============================================================
# A. Config tests
# ============================================================

def test_valid_hca_config_builds():
    config = HCAConfig(
        d_model=256,
        n_heads=4,
        head_dim=64,
        compression_factor=16,
        window_size=32,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=64,
        max_seq_len=512,
        init_std=0.02,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )

    hca = HCAAttention(config)

    assert hca.d_model == 256
    assert hca.n_heads == 4
    assert hca.head_dim == 64
    assert hca.compression_factor == 16
    assert hca.window_size == 32
    assert hca.use_attention_sink is True
    assert hca.use_grouped_output_projection is True


@pytest.mark.parametrize("d_model", [0, -1, -64])
def test_invalid_hca_d_model_raises(d_model):
    with pytest.raises(ValueError):
        HCAAttention(make_hca_config(d_model=d_model))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_heads": 0},
        {"n_heads": -1},
        {"head_dim": 0},
        {"head_dim": -1},
        {"head_dim": 15},  # 4 * 15 != 64
    ],
)
def test_invalid_heads_or_head_dim_raises(kwargs):
    with pytest.raises(ValueError):
        HCAAttention(make_hca_config(**kwargs))


@pytest.mark.parametrize("compression_factor", [0, -1, -4])
def test_invalid_compression_factor_raises(compression_factor):
    with pytest.raises(ValueError):
        HCAAttention(make_hca_config(compression_factor=compression_factor))


@pytest.mark.parametrize("window_size", [0, -1, -4])
def test_invalid_window_size_raises(window_size):
    with pytest.raises(ValueError):
        HCAAttention(make_hca_config(window_size=window_size))


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
def test_invalid_hca_dropout_raises(field, value):
    with pytest.raises(ValueError):
        HCAAttention(make_hca_config(**{field: value}))


@pytest.mark.parametrize("rotary_dim", [0, -1, 17, 32])
def test_invalid_hca_rotary_dim_raises(rotary_dim):
    with pytest.raises(ValueError):
        HCAAttention(make_hca_config(rotary_dim=rotary_dim))


# ============================================================
# B. Compressor tests
# ============================================================

def test_compressor_output_shape_exact_multiple():
    B, T, Dh = 2, 32, 16
    m = 8

    compressor = make_compressor(compression_factor=m, head_dim=Dh)

    C = torch.randn(B, T, Dh)
    Z = torch.randn(B, T, Dh)

    compressed_C, compressed_valid_mask, compressed_position_ids = compressor(C, Z)

    S = 4

    assert compressed_C.shape == (B, S, Dh)
    assert compressed_valid_mask.shape == (B, S)
    assert compressed_position_ids.shape == (S,)


def test_compressor_output_shape_non_exact_multiple():
    B, T, Dh = 2, 30, 16
    m = 8

    compressor = make_compressor(compression_factor=m, head_dim=Dh)

    C = torch.randn(B, T, Dh)
    Z = torch.randn(B, T, Dh)

    compressed_C, compressed_valid_mask, compressed_position_ids = compressor(C, Z)

    S = math.ceil(T / m)

    assert S == 4
    assert compressed_C.shape == (B, S, Dh)
    assert compressed_valid_mask.shape == (B, S)
    assert compressed_position_ids.shape == (S,)


def test_compressor_valid_mask_all_valid():
    B, T, Dh = 2, 16, 16
    m = 4

    compressor = make_compressor(compression_factor=m, head_dim=Dh)

    C = torch.randn(B, T, Dh)
    Z = torch.randn(B, T, Dh)

    _, compressed_valid_mask, _ = compressor(C, Z, attention_mask=None)

    assert compressed_valid_mask.all()


def test_compressor_valid_mask_with_padding():
    B, T, Dh = 2, 16, 16
    m = 4

    compressor = make_compressor(compression_factor=m, head_dim=Dh)

    C = torch.randn(B, T, Dh)
    Z = torch.randn(B, T, Dh)

    attention_mask = torch.ones(B, T, dtype=torch.long)

    # Block 2 for batch 0: positions 8,9,10,11 are all padding.
    attention_mask[0, 8:12] = 0

    compressed_C, compressed_valid_mask, _ = compressor(
        C,
        Z,
        attention_mask=attention_mask,
    )

    assert compressed_valid_mask[0, 2] == 0
    assert torch.allclose(
        compressed_C[0, 2],
        torch.zeros_like(compressed_C[0, 2]),
        atol=0.0,
        rtol=0.0,
    )

    assert compressed_valid_mask[1].all()


def test_compressor_ignores_padding_tokens():
    B, T, Dh = 1, 4, 8
    m = 4

    compressor = make_compressor(compression_factor=m, head_dim=Dh)

    C = torch.zeros(B, T, Dh)
    Z = torch.zeros(B, T, Dh)

    # Valid tokens are small.
    C[:, 0, :] = 1.0
    C[:, 1, :] = 1.0
    C[:, 2, :] = 1.0

    # Padding token has huge value. It must not dominate.
    C[:, 3, :] = 1_000_000.0

    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    compressed_C, compressed_valid_mask, _ = compressor(
        C,
        Z,
        attention_mask=attention_mask,
    )

    assert compressed_valid_mask[0, 0] == 1
    assert torch.isfinite(compressed_C).all()
    assert compressed_C.max() < 10.0


def test_compressor_backward():
    B, T, Dh = 2, 16, 16
    m = 4

    compressor = make_compressor(compression_factor=m, head_dim=Dh)

    C = torch.randn(B, T, Dh, requires_grad=True)
    Z = torch.randn(B, T, Dh, requires_grad=True)

    compressed_C, _, _ = compressor(C, Z)

    loss = compressed_C.sum()
    loss.backward()

    assert C.grad is not None
    assert Z.grad is not None
    assert compressor.compression_bias.grad is not None

    assert torch.isfinite(C.grad).all()
    assert torch.isfinite(Z.grad).all()
    assert torch.isfinite(compressor.compression_bias.grad).all()


# ============================================================
# C. Forward and shape tests
# ============================================================

def test_hca_output_shape_matches_input():
    hca = make_hca()
    x = make_hca_input(B=2, T=16, D=64)

    out = hca(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(16, 64),
        torch.randn(2, 16, 64, 1),
    ],
)
def test_hca_rejects_wrong_input_rank(bad_x):
    hca = make_hca()

    with pytest.raises(ValueError):
        hca(bad_x)


def test_hca_rejects_wrong_hidden_size():
    hca = make_hca(d_model=64)

    x = torch.randn(2, 16, 32)

    with pytest.raises(ValueError):
        hca(x)


def test_hca_rejects_too_long_sequence():
    hca = make_hca(max_seq_len=8)

    x = torch.randn(2, 9, 64)

    with pytest.raises(ValueError):
        hca(x)


def test_hca_need_weights_returns_aux():
    hca = make_hca(
        compression_factor=4,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )
    x = make_hca_input(B=2, T=16, D=64)

    out, aux = hca(x, need_weights=True)

    B, T, D = x.shape
    S = math.ceil(T / hca.compression_factor)

    assert out.shape == x.shape
    assert isinstance(aux, dict)

    assert aux["global_attn_weights"].shape == (B, hca.n_heads, T, S)
    assert aux["local_attn_weights"].shape == (B, hca.n_heads, T, T)
    assert aux["compressed_valid_mask"].shape == (B, S)
    assert aux["compressed_position_ids"].shape == (S,)

    assert "sink_attn_weights" in aux
    assert aux["sink_attn_weights"].shape == (B, hca.n_heads, T, 1)


# ============================================================
# D. Masks and global causality
# ============================================================

def test_global_compressed_attention_blocks_current_and_future_blocks():
    m = 4
    hca = make_hca(compression_factor=m, window_size=4)
    hca.eval()

    B, T, D = 2, 16, 64

    x = make_hca_input(B=B, T=T, D=D)

    _, aux = hca(x, need_weights=True)

    global_weights = aux["global_attn_weights"]
    S = global_weights.shape[-1]

    for t in range(T):
        query_block = t // m
        if query_block < S:
            blocked = global_weights[:, :, t, query_block:]
            assert torch.allclose(
                blocked,
                torch.zeros_like(blocked),
                atol=0.0,
                rtol=0.0,
            )


def test_first_block_has_no_global_attention():
    m = 4
    hca = make_hca(compression_factor=m, window_size=4)
    hca.eval()

    x = make_hca_input(B=2, T=16, D=64)

    _, aux = hca(x, need_weights=True)

    global_weights = aux["global_attn_weights"]

    assert torch.allclose(
        global_weights[:, :, :m, :],
        torch.zeros_like(global_weights[:, :, :m, :]),
        atol=0.0,
        rtol=0.0,
    )


def test_local_window_is_causal():
    hca = make_hca(window_size=4)
    hca.eval()

    B, T, D = 2, 16, 64
    x = make_hca_input(B=B, T=T, D=D)

    _, aux = hca(x, need_weights=True)

    local_weights = aux["local_attn_weights"]

    future_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool),
        diagonal=1,
    )

    future_weights = local_weights[:, :, future_mask]

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=0.0,
        rtol=0.0,
    )


def test_local_window_limits_past_context():
    W = 4
    hca = make_hca(window_size=W)
    hca.eval()

    B, T, D = 2, 16, 64
    x = make_hca_input(B=B, T=T, D=D)

    _, aux = hca(x, need_weights=True)

    local_weights = aux["local_attn_weights"]

    q_pos = torch.arange(T)[:, None]
    k_pos = torch.arange(T)[None, :]

    too_old_mask = (q_pos - k_pos) >= W

    too_old_weights = local_weights[:, :, too_old_mask]

    assert torch.allclose(
        too_old_weights,
        torch.zeros_like(too_old_weights),
        atol=0.0,
        rtol=0.0,
    )


def test_hca_changing_future_tokens_does_not_change_past_outputs():
    hca = make_hca(
        attention_dropout=0.0,
        residual_dropout=0.0,
        compression_factor=4,
        window_size=4,
    )
    hca.eval()

    B, T, D = 2, 16, 64
    cut = 8

    x1 = make_hca_input(B=B, T=T, D=D)
    x2 = x1.clone()
    x2[:, cut:, :] = torch.randn_like(x2[:, cut:, :])

    out1 = hca(x1)
    out2 = hca(x2)

    assert torch.allclose(
        out1[:, :cut, :],
        out2[:, :cut, :],
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# E. attention_mask tests
# ============================================================

def test_attention_mask_blocks_padding_local_keys():
    hca = make_hca(window_size=8)
    hca.eval()

    B, T, D = 2, 16, 64
    x = make_hca_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[0, 5] = 0
    attention_mask[1, 7] = 0

    _, aux = hca(
        x,
        attention_mask=attention_mask,
        need_weights=True,
    )

    local_weights = aux["local_attn_weights"]

    assert torch.allclose(
        local_weights[0, :, :, 5],
        torch.zeros_like(local_weights[0, :, :, 5]),
        atol=0.0,
        rtol=0.0,
    )

    assert torch.allclose(
        local_weights[1, :, :, 7],
        torch.zeros_like(local_weights[1, :, :, 7]),
        atol=0.0,
        rtol=0.0,
    )


def test_attention_mask_blocks_padding_compressed_blocks():
    m = 4
    hca = make_hca(compression_factor=m, window_size=4)
    hca.eval()

    B, T, D = 2, 16, 64
    x = make_hca_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T, dtype=torch.long)

    # Block 2 for batch 0 is fully padding.
    attention_mask[0, 8:12] = 0

    _, aux = hca(
        x,
        attention_mask=attention_mask,
        need_weights=True,
    )

    compressed_valid_mask = aux["compressed_valid_mask"]
    global_weights = aux["global_attn_weights"]

    assert compressed_valid_mask[0, 2] == 0

    assert torch.allclose(
        global_weights[0, :, :, 2],
        torch.zeros_like(global_weights[0, :, :, 2]),
        atol=0.0,
        rtol=0.0,
    )


def test_hca_attention_mask_shape_validation_accepts_BT():
    hca = make_hca()

    x = make_hca_input(B=2, T=16, D=64)
    attention_mask = torch.ones(2, 16)

    out = hca(x, attention_mask=attention_mask)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.ones(16),
        torch.ones(2, 16, 1),
        torch.ones(2, 17),
    ],
)
def test_hca_attention_mask_shape_validation_rejects_bad_shapes(bad_mask):
    hca = make_hca()

    x = make_hca_input(B=2, T=16, D=64)

    with pytest.raises(ValueError):
        hca(x, attention_mask=bad_mask)


# ============================================================
# F. RoPE tests
# ============================================================

def test_hca_start_pos_matches_explicit_position_ids():
    hca = make_hca(
        use_rope=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    hca.eval()

    B, T, D = 2, 16, 64
    start_pos = 10

    x = make_hca_input(B=B, T=T, D=D)

    out_start = hca(x, start_pos=start_pos)
    out_explicit = hca(
        x,
        position_ids=torch.arange(start_pos, start_pos + T),
    )

    assert torch.allclose(out_start, out_explicit, atol=1e-5, rtol=1e-5)


def test_hca_no_rope_when_disabled():
    hca = make_hca(
        use_rope=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    hca.eval()

    B, T, D = 2, 16, 64
    x = make_hca_input(B=B, T=T, D=D)

    out1 = hca(x, start_pos=0)
    out2 = hca(x, position_ids=torch.arange(10, 10 + T), start_pos=10)

    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)


# ============================================================
# G. Attention weights tests
# ============================================================

def test_sink_plus_global_plus_local_weights_sum_to_one():
    torch.manual_seed(0)

    config = HCAConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        compression_factor=4,
        window_size=8,
        max_seq_len=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_rope=True,
        rotary_dim=16,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )

    hca = HCAAttention(config)
    hca.eval()

    B, T, D = 2, 17, 64
    x = torch.randn(B, T, D)

    out, aux = hca(x, need_weights=True)

    sink_sum = aux["sink_attn_weights"].sum(dim=-1)      # [B, H, T]
    global_sum = aux["global_attn_weights"].sum(dim=-1)  # [B, H, T]
    local_sum = aux["local_attn_weights"].sum(dim=-1)    # [B, H, T]

    total = sink_sum + global_sum + local_sum

    assert torch.allclose(
        total,
        torch.ones_like(total),
        atol=1e-5,
        rtol=1e-5,
    )

    assert out.shape == (B, T, D)


def test_no_nan_when_no_global_blocks_available():
    m = 4
    hca = make_hca(
        compression_factor=m,
        window_size=4,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_sink=True,
    )
    hca.eval()

    x = make_hca_input(B=2, T=16, D=64)

    out, aux = hca(x, need_weights=True)

    assert torch.isfinite(out).all()
    assert torch.isfinite(aux["global_attn_weights"]).all()
    assert torch.isfinite(aux["local_attn_weights"]).all()
    assert torch.isfinite(aux["sink_attn_weights"]).all()

    assert torch.allclose(
        aux["global_attn_weights"][:, :, :m, :],
        torch.zeros_like(aux["global_attn_weights"][:, :, :m, :]),
        atol=0.0,
        rtol=0.0,
    )

    # In the first compression block, there are no global compressed blocks.
    # The probability mass should be assigned to sink + local attention.
    total_first_block = (
        aux["sink_attn_weights"][:, :, :m, :].sum(dim=-1)
        + aux["local_attn_weights"][:, :, :m, :].sum(dim=-1)
    )

    assert torch.allclose(
        total_first_block,
        torch.ones_like(total_first_block),
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# H. Gradient tests
# ============================================================

def test_hca_backward_computes_gradients():
    hca = make_hca(
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=True,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )

    x = make_hca_input(B=2, T=16, D=64)
    x.requires_grad_(True)

    out = hca(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert hca.q_proj.weight.grad is not None
    assert hca.kv_proj.weight.grad is not None
    assert hca.z_proj.weight.grad is not None
    assert hca.compressor.compression_bias.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(hca.q_proj.weight.grad).all()
    assert torch.isfinite(hca.kv_proj.weight.grad).all()
    assert torch.isfinite(hca.z_proj.weight.grad).all()
    assert torch.isfinite(hca.compressor.compression_bias.grad).all()

    # Attention sink parameters should receive gradients.
    assert hca.sink_k.grad is not None
    assert hca.sink_v.grad is not None
    assert torch.isfinite(hca.sink_k.grad).all()
    assert torch.isfinite(hca.sink_v.grad).all()

    # Grouped output projection parameters should receive gradients.
    assert hasattr(hca.out_proj, "group_projs")

    for proj in hca.out_proj.group_projs:
        assert proj.weight.grad is not None
        assert torch.isfinite(proj.weight.grad).all()

        if proj.bias is not None:
            assert proj.bias.grad is not None
            assert torch.isfinite(proj.bias.grad).all()


# ============================================================
# I. Integration with block-like interface
# ============================================================

def test_hca_can_replace_attention_in_block_interface():
    B, T, D = 2, 16, 64

    norm1 = RMSNorm(dim=D)
    attention = make_hca(
        d_model=D,
        n_heads=4,
        head_dim=16,
        compression_factor=4,
        window_size=4,
        max_seq_len=T,
    )

    x = torch.randn(B, T, D)

    residual = x
    h = norm1(x)
    attn_out = attention(h)
    out = residual + attn_out

    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "output_projection_groups",
    [0, -1, 3],  # 3 no divide n_heads=4
)
def test_invalid_output_projection_groups_raises(output_projection_groups):
    with pytest.raises(ValueError):
        HCAAttention(
            make_hca_config(
                use_grouped_output_projection=True,
                output_projection_groups=output_projection_groups,
            )
        )

def test_global_plus_local_weights_sum_to_one_without_sink():
    torch.manual_seed(0)

    config = HCAConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        compression_factor=4,
        window_size=8,
        max_seq_len=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_rope=True,
        rotary_dim=16,
        use_attention_sink=False,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )

    hca = HCAAttention(config)
    hca.eval()

    B, T, D = 2, 17, 64
    x = torch.randn(B, T, D)

    out, aux = hca(x, need_weights=True)

    assert "sink_attn_weights" not in aux

    global_sum = aux["global_attn_weights"].sum(dim=-1)
    local_sum = aux["local_attn_weights"].sum(dim=-1)

    total = global_sum + local_sum

    assert torch.allclose(
        total,
        torch.ones_like(total),
        atol=1e-5,
        rtol=1e-5,
    )

    assert out.shape == (B, T, D)


def test_grouped_output_projection_shape_and_gradients():
    torch.manual_seed(0)

    proj = GroupedOutputProjection(
        n_heads=4,
        head_dim=16,
        num_groups=4,
        bias=True,
        init_std=0.02,
    )

    x = torch.randn(2, 8, 4, 16, requires_grad=True)

    out = proj(x)

    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for group_proj in proj.group_projs:
        assert group_proj.weight.grad is not None
        assert torch.isfinite(group_proj.weight.grad).all()

        assert group_proj.bias.grad is not None
        assert torch.isfinite(group_proj.bias.grad).all()


def test_attention_sink_is_present_and_receives_valid_mass():
    torch.manual_seed(0)

    hca = make_hca(
        use_attention_sink=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    hca.eval()

    x = make_hca_input(B=2, T=16, D=64)

    _, aux = hca(x, need_weights=True)

    assert "sink_attn_weights" in aux

    sink_weights = aux["sink_attn_weights"]

    assert sink_weights.shape == (2, hca.n_heads, 16, 1)
    assert torch.isfinite(sink_weights).all()
    assert (sink_weights >= 0).all()

    # At least some probability mass should be assigned to the sink.
    assert sink_weights.sum() > 0



def test_hca_aux_returns_compressed_position_ids():
    m = 4
    hca = make_hca(
        compression_factor=m,
        use_attention_sink=True,
    )
    hca.eval()

    B, T, D = 2, 17, 64
    start_pos = 10

    x = make_hca_input(B=B, T=T, D=D)

    _, aux = hca(
        x,
        start_pos=start_pos,
        need_weights=True,
    )

    # Blocks:
    # [0,1,2,3]       -> position 13
    # [4,5,6,7]       -> position 17
    # [8,9,10,11]     -> position 21
    # [12,13,14,15]   -> position 25
    # [16]            -> position 26
    expected = torch.tensor(
        [13, 17, 21, 25, 26],
        device=x.device,
        dtype=torch.long)

    assert "compressed_position_ids" in aux
    assert torch.equal(aux["compressed_position_ids"], expected)


def test_sink_global_plus_local_weights_sum_to_one():
    hca = make_hca(
        compression_factor=4,
        window_size=4,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_sink=True,
    )
    hca.eval()

    x = make_hca_input(B=2, T=16, D=64)

    _, aux = hca(x, need_weights=True)

    sink_w = aux["sink_attn_weights"]
    global_w = aux["global_attn_weights"]
    local_w = aux["local_attn_weights"]

    total = (
        sink_w.sum(dim=-1)
        + global_w.sum(dim=-1)
        + local_w.sum(dim=-1)
    )

    assert torch.allclose(
        total,
        torch.ones_like(total),
        atol=1e-5,
        rtol=1e-5,
    )
