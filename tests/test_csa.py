
# ============================================================
# Helpers - REPLACE make_csa_config with this version
# ============================================================

from src.deepseek_csa_attention import *
import pytest 

def make_csa_config(**overrides):
    cfg = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,

        compression_factor=4,
        top_k=3,
        window_size=4,

        indexer_dim=8,
        n_indexer_heads=2,
        query_compression_dim=16,

        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=True,

        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=16,

        max_seq_len=128,
        init_std=0.02,

        # Canonical additions
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
        use_indexer_score_bias=False,
        use_separate_local_kv=True,
    )
    cfg.update(overrides)
    return CSAConfig(**cfg)


def make_csa(**overrides):
    return CSAAttention(make_csa_config(**overrides))


def make_csa_input(B=2, T=16, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


def make_overlapped_compressor(m=4, dim=8):
    return CSAOverlappedCompressor(
        compression_factor=m,
        dim=dim,
        init_std=0.02,
    )


def make_indexer(m=4, top_k=3):
    return CSALightningIndexer(
        compression_factor=m,
        top_k=top_k,
    )


# ============================================================
# A. Config tests
# ============================================================

def test_valid_csa_config_builds():
    config = CSAConfig(
        d_model=256,
        n_heads=4,
        head_dim=64,
        compression_factor=8,
        top_k=8,
        window_size=32,
        indexer_dim=32,
        n_indexer_heads=2,
        query_compression_dim=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=64,
        max_seq_len=512,
        init_std=0.02,

        # Canonical additions
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
        use_indexer_score_bias=False,
        use_separate_local_kv=True,
    )

    csa = CSAAttention(config)

    assert csa.d_model == 256
    assert csa.n_heads == 4
    assert csa.head_dim == 64
    assert csa.compression_factor == 8
    assert csa.top_k == 8
    assert csa.window_size == 32
    assert csa.indexer_dim == 32
    assert csa.n_indexer_heads == 2
    assert csa.query_compression_dim == 64

    assert csa.use_attention_sink is True
    assert csa.use_grouped_output_projection is True
    assert csa.use_indexer_score_bias is False
    assert csa.use_separate_local_kv is True

@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_projection_groups": 0},
        {"output_projection_groups": -1},
        {"output_projection_groups": 3},  # 3 does not divide n_heads=4
    ],
)
def test_invalid_csa_output_projection_groups_raise(kwargs):
    with pytest.raises(ValueError):
        CSAAttention(
            make_csa_config(
                use_grouped_output_projection=True,
                **kwargs,
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"d_model": -1},
        {"n_heads": 0},
        {"n_heads": -1},
        {"head_dim": 0},
        {"head_dim": -1},
        {"head_dim": 15},  # 4 * 15 != 64
    ],
)
def test_invalid_model_dims_raise(kwargs):
    with pytest.raises(ValueError):
        CSAAttention(make_csa_config(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compression_factor": 0},
        {"compression_factor": -1},
        {"top_k": 0},
        {"top_k": -1},
        {"window_size": 0},
        {"window_size": -1},
        {"indexer_dim": 0},
        {"indexer_dim": -1},
        {"n_indexer_heads": 0},
        {"n_indexer_heads": -1},
        {"query_compression_dim": 0},
        {"query_compression_dim": -1},
    ],
)
def test_invalid_csa_hyperparams_raise(kwargs):
    with pytest.raises(ValueError):
        CSAAttention(make_csa_config(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_dropout": -0.1},
        {"attention_dropout": 1.0},
        {"attention_dropout": 1.5},
        {"residual_dropout": -0.1},
        {"residual_dropout": 1.0},
        {"residual_dropout": 1.5},
        {"init_std": 0.0},
        {"init_std": -0.01},
        {"rope_theta": 0.0},
        {"rope_theta": -1.0},
        {"rotary_dim": 0},
        {"rotary_dim": -1},
        {"rotary_dim": 17},
        {"rotary_dim": 32},
    ],
)
def test_invalid_dropout_rope_init_raise(kwargs):
    with pytest.raises(ValueError):
        CSAAttention(make_csa_config(**kwargs))


# ============================================================
# B. CSAOverlappedCompressor tests
# ============================================================

def test_overlapped_compressor_output_shape_exact_multiple():
    B, T, D = 2, 32, 8
    m = 8

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    C_comp, valid_mask, pos = compressor(C_a, C_b, Z_a, Z_b)

    assert C_comp.shape == (B, 4, D)
    assert valid_mask.shape == (B, 4)
    assert pos.shape == (4,)


def test_overlapped_compressor_output_shape_non_exact_multiple():
    B, T, D = 2, 30, 8
    m = 8

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    C_comp, valid_mask, pos = compressor(C_a, C_b, Z_a, Z_b)

    S = math.ceil(T / m)

    assert S == 4
    assert C_comp.shape == (B, S, D)
    assert valid_mask.shape == (B, S)
    assert pos.shape == (S,)


def test_first_block_uses_only_a_branch():
    B, T, D = 1, 16, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b_1 = torch.zeros(B, T, D)
    C_b_2 = torch.full((B, T, D), 1_000_000.0)

    Z_a = torch.zeros(B, T, D)
    Z_b = torch.zeros(B, T, D)

    out_1, _, _ = compressor(C_a, C_b_1, Z_a, Z_b)
    out_2, _, _ = compressor(C_a, C_b_2, Z_a, Z_b)

    assert torch.allclose(out_1[:, 0, :], out_2[:, 0, :], atol=1e-6, rtol=1e-6)


def test_second_block_uses_current_a_and_previous_b():
    B, T, D = 1, 16, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    C_a_2 = C_a.clone()
    C_b_2 = C_b.clone()
    Z_a_2 = Z_a.clone()
    Z_b_2 = Z_b.clone()

    # Block i=1 uses A current [4:8] and B previous [0:4].
    # Modify outside those regions.
    C_a_2[:, :4, :] += 1000.0
    C_a_2[:, 8:, :] += 1000.0
    Z_a_2[:, :4, :] += 1000.0
    Z_a_2[:, 8:, :] += 1000.0

    C_b_2[:, 4:, :] += 1000.0
    Z_b_2[:, 4:, :] += 1000.0

    out_1, _, _ = compressor(C_a, C_b, Z_a, Z_b)
    out_2, _, _ = compressor(C_a_2, C_b_2, Z_a_2, Z_b_2)

    assert torch.allclose(out_1[:, 1, :], out_2[:, 1, :], atol=1e-5, rtol=1e-5)


def test_overlapped_compressor_padding_block_zero():
    B, T, D = 1, 16, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    attention_mask = torch.ones(B, T, dtype=torch.long)

    # For block i=2:
    # A = [8:12], B = [4:8]. Make both all padding.
    attention_mask[:, 4:12] = 0

    C_comp, valid_mask, _ = compressor(
        C_a,
        C_b,
        Z_a,
        Z_b,
        attention_mask=attention_mask,
    )

    assert valid_mask[0, 2] == 0
    assert torch.allclose(C_comp[0, 2], torch.zeros_like(C_comp[0, 2]), atol=0.0, rtol=0.0)


def test_overlapped_compressor_ignores_padding_tokens():
    B, T, D = 1, 8, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.ones(B, T, D)
    C_b = torch.ones(B, T, D)
    Z_a = torch.zeros(B, T, D)
    Z_b = torch.zeros(B, T, D)

    C_a[:, 3, :] = 1_000_000.0
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[:, 3] = 0

    C_comp, valid_mask, _ = compressor(
        C_a,
        C_b,
        Z_a,
        Z_b,
        attention_mask=attention_mask,
    )

    assert valid_mask[0, 0] == 1
    assert C_comp[0, 0].max() < 10.0


def test_overlapped_compressor_position_ids_none():
    B, T, D = 1, 16, 4
    m = 4
    start_pos = 10

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    _, _, pos = compressor(
        C_a,
        C_b,
        Z_a,
        Z_b,
        start_pos=start_pos,
    )

    expected = torch.tensor([13, 17, 21, 25], dtype=torch.long)

    assert torch.equal(pos.cpu(), expected)


def test_overlapped_compressor_position_ids_T():
    B, T, D = 1, 16, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    position_ids = torch.arange(100, 100 + T)

    _, _, pos = compressor(
        C_a,
        C_b,
        Z_a,
        Z_b,
        position_ids=position_ids,
    )

    expected = position_ids[torch.tensor([3, 7, 11, 15])]

    assert torch.equal(pos.cpu(), expected.cpu())


def test_overlapped_compressor_position_ids_BT():
    B, T, D = 2, 16, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)

    position_ids = torch.stack(
        [
            torch.arange(100, 100 + T),
            torch.arange(200, 200 + T),
        ],
        dim=0,
    )

    _, _, pos = compressor(
        C_a,
        C_b,
        Z_a,
        Z_b,
        position_ids=position_ids,
    )

    expected = position_ids[:, torch.tensor([3, 7, 11, 15])]

    assert pos.shape == (B, 4)
    assert torch.equal(pos.cpu(), expected.cpu())


def test_overlapped_compressor_backward():
    B, T, D = 2, 16, 4
    m = 4

    compressor = make_overlapped_compressor(m=m, dim=D)

    C_a = torch.randn(B, T, D, requires_grad=True)
    C_b = torch.randn(B, T, D, requires_grad=True)
    Z_a = torch.randn(B, T, D, requires_grad=True)
    Z_b = torch.randn(B, T, D, requires_grad=True)

    C_comp, _, _ = compressor(C_a, C_b, Z_a, Z_b)

    loss = C_comp.sum()
    loss.backward()

    assert C_a.grad is not None
    assert C_b.grad is not None
    assert Z_a.grad is not None
    assert Z_b.grad is not None
    assert compressor.bias_a.grad is not None
    assert compressor.bias_b.grad is not None

    assert torch.isfinite(C_a.grad).all()
    assert torch.isfinite(C_b.grad).all()
    assert torch.isfinite(Z_a.grad).all()
    assert torch.isfinite(Z_b.grad).all()
    assert torch.isfinite(compressor.bias_a.grad).all()
    assert torch.isfinite(compressor.bias_b.grad).all()


# ============================================================
# C. CSALightningIndexer tests
# ============================================================

def test_indexer_output_shapes():
    B, T, H_i, I, S = 2, 16, 2, 8, 5
    top_k = 3

    indexer = make_indexer(m=4, top_k=top_k)

    index_q = torch.randn(B, T, H_i, I)
    index_weights = torch.randn(B, T, H_i)
    I_comp = torch.randn(B, S, I)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    topk_indices, topk_scores, topk_mask = indexer(
        index_q,
        index_weights,
        I_comp,
        valid_mask,
    )

    K = min(top_k, S)

    assert topk_indices.shape == (B, T, K)
    assert topk_scores.shape == (B, T, K)
    assert topk_mask.shape == (B, T, K)


def test_indexer_scores_shape_when_requested():
    B, T, H_i, I, S = 2, 16, 2, 8, 5

    indexer = make_indexer(m=4, top_k=3)

    index_q = torch.randn(B, T, H_i, I)
    index_weights = torch.randn(B, T, H_i)
    I_comp = torch.randn(B, S, I)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    topk_indices, topk_scores, topk_mask, index_scores = indexer(
        index_q,
        index_weights,
        I_comp,
        valid_mask,
        need_scores=True,
    )

    assert index_scores.shape == (B, T, S)


def test_indexer_respects_causality():
    B, T, H_i, I, S = 2, 16, 2, 8, 5
    m = 4

    indexer = make_indexer(m=m, top_k=3)

    index_q = torch.randn(B, T, H_i, I)
    index_weights = torch.randn(B, T, H_i)
    I_comp = torch.randn(B, S, I)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    topk_indices, _, topk_mask = indexer(
        index_q,
        index_weights,
        I_comp,
        valid_mask,
    )

    for t in range(T):
        query_block = t // m
        selected = topk_indices[:, t, :]
        valid = topk_mask[:, t, :]

        if valid.any():
            assert (selected[valid] < query_block).all()


def test_indexer_first_block_has_no_valid_topk():
    B, T, H_i, I, S = 2, 16, 2, 8, 5
    m = 4

    indexer = make_indexer(m=m, top_k=3)

    index_q = torch.randn(B, T, H_i, I)
    index_weights = torch.randn(B, T, H_i)
    I_comp = torch.randn(B, S, I)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    _, _, topk_mask = indexer(index_q, index_weights, I_comp, valid_mask)

    assert not topk_mask[:, :m, :].any()


def test_indexer_respects_compressed_valid_mask():
    B, T, H_i, I, S = 2, 16, 2, 8, 5
    m = 4

    indexer = make_indexer(m=m, top_k=3)

    index_q = torch.randn(B, T, H_i, I)
    index_weights = torch.randn(B, T, H_i)
    I_comp = torch.randn(B, S, I)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    valid_mask[:, 1] = False

    topk_indices, _, topk_mask = indexer(
        index_q,
        index_weights,
        I_comp,
        valid_mask,
    )

    valid_selected = topk_indices[topk_mask]

    assert not (valid_selected == 1).any()


def test_indexer_topk_larger_than_num_blocks():
    B, T, H_i, I, S = 2, 16, 2, 8, 3

    indexer = make_indexer(m=4, top_k=10)

    index_q = torch.randn(B, T, H_i, I)
    index_weights = torch.randn(B, T, H_i)
    I_comp = torch.randn(B, S, I)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    topk_indices, topk_scores, topk_mask = indexer(
        index_q,
        index_weights,
        I_comp,
        valid_mask,
    )

    assert topk_indices.shape[-1] == S
    assert topk_scores.shape[-1] == S
    assert topk_mask.shape[-1] == S


def test_indexer_backward():
    B, T, H_i, I, S = 2, 16, 2, 8, 5

    indexer = make_indexer(m=4, top_k=3)

    index_q = torch.randn(B, T, H_i, I, requires_grad=True)
    index_weights = torch.randn(B, T, H_i, requires_grad=True)
    I_comp = torch.randn(B, S, I, requires_grad=True)
    valid_mask = torch.ones(B, S, dtype=torch.bool)

    _, topk_scores, topk_mask = indexer(
        index_q,
        index_weights,
        I_comp,
        valid_mask,
    )

    loss = topk_scores[topk_mask].sum()
    loss.backward()

    assert index_q.grad is not None
    assert index_weights.grad is not None
    assert I_comp.grad is not None

    assert torch.isfinite(index_q.grad).all()
    assert torch.isfinite(index_weights.grad).all()
    assert torch.isfinite(I_comp.grad).all()


# ============================================================
# D. CSAAttention forward tests
# ============================================================

def test_csa_output_shape_matches_input():
    csa = make_csa()
    x = make_csa_input()

    out = csa(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(16, 64),
        torch.randn(2, 16, 64, 1),
    ],
)
def test_csa_rejects_wrong_input_rank(bad_x):
    csa = make_csa()

    with pytest.raises(ValueError):
        csa(bad_x)


def test_csa_rejects_wrong_hidden_size():
    csa = make_csa(d_model=64)

    x = torch.randn(2, 16, 32)

    with pytest.raises(ValueError):
        csa(x)


def test_csa_rejects_too_long_sequence():
    csa = make_csa(max_seq_len=8)

    x = torch.randn(2, 9, 64)

    with pytest.raises(ValueError):
        csa(x)


def test_csa_need_weights_returns_aux():
    csa = make_csa(
        compression_factor=4,
        top_k=3,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )
    x = make_csa_input(B=2, T=16, D=64)

    out, aux = csa(x, need_weights=True)

    B, T, D = x.shape
    S = math.ceil(T / csa.compression_factor)
    K = min(csa.top_k, S)

    assert out.shape == x.shape

    assert aux["global_attn_weights"].shape == (B, csa.n_heads, T, K)
    assert aux["local_attn_weights"].shape == (B, csa.n_heads, T, T)
    assert aux["topk_indices"].shape == (B, T, K)
    assert aux["topk_scores"].shape == (B, T, K)
    assert aux["topk_mask"].shape == (B, T, K)
    assert aux["compressed_valid_mask"].shape == (B, S)
    assert aux["index_scores"].shape == (B, T, S)

    assert "sink_attn_weights" in aux
    assert aux["sink_attn_weights"].shape == (B, csa.n_heads, T, 1)

    assert "compressed_position_ids" in aux
    assert aux["compressed_position_ids"].shape == (S,)


# ============================================================
# E. Compression inside CSAAttention
# ============================================================

def test_csa_uses_overlapped_compressor_for_kv():
    csa = make_csa()

    assert hasattr(csa.kv_compressor, "bias_a")
    assert hasattr(csa.kv_compressor, "bias_b")


def test_csa_uses_separate_index_compressor():
    csa = make_csa(head_dim=16, indexer_dim=8)

    assert csa.kv_compressor is not csa.index_compressor
    assert csa.kv_compressor.dim == csa.head_dim
    assert csa.index_compressor.dim == csa.indexer_dim

def test_csa_has_separate_local_kv_projection_when_enabled():
    csa = make_csa(use_separate_local_kv=True)

    assert hasattr(csa, "local_kv_proj")
    assert csa.local_kv_proj is not None
    assert csa.local_kv_proj.out_features == csa.head_dim


def test_csa_can_disable_separate_local_kv_projection():
    csa = make_csa(use_separate_local_kv=False)

    assert hasattr(csa, "local_kv_proj")
    assert csa.local_kv_proj is None

# ============================================================
# F. Global sparse causality tests
# ============================================================

def test_csa_topk_blocks_current_and_future_blocks():
    m = 4
    csa = make_csa(compression_factor=m, top_k=3)
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    _, aux = csa(x, need_weights=True)

    topk_indices = aux["topk_indices"]
    topk_mask = aux["topk_mask"]

    for t in range(x.shape[1]):
        query_block = t // m
        selected = topk_indices[:, t, :]
        valid = topk_mask[:, t, :]

        if valid.any():
            assert (selected[valid] < query_block).all()


def test_csa_first_block_has_zero_global_weights():
    m = 4
    csa = make_csa(compression_factor=m, top_k=3)
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    _, aux = csa(x, need_weights=True)

    assert not aux["topk_mask"][:, :m, :].any()

    assert torch.allclose(
        aux["global_attn_weights"][:, :, :m, :],
        torch.zeros_like(aux["global_attn_weights"][:, :, :m, :]),
        atol=0.0,
        rtol=0.0,
    )


def test_csa_global_weights_zero_for_invalid_topk():
    csa = make_csa(compression_factor=4, top_k=3)
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    _, aux = csa(x, need_weights=True)

    invalid = ~aux["topk_mask"]  # [B,T,K]
    weights = aux["global_attn_weights"]  # [B,H,T,K]

    invalid_weights = weights.masked_select(invalid[:, None, :, :].expand_as(weights))

    assert torch.allclose(
        invalid_weights,
        torch.zeros_like(invalid_weights),
        atol=0.0,
        rtol=0.0,
    )


# ============================================================
# G. Local sliding-window tests
# ============================================================

def test_csa_local_window_is_causal():
    csa = make_csa(window_size=4)
    csa.eval()

    B, T, D = 2, 16, 64
    x = make_csa_input(B=B, T=T, D=D)

    _, aux = csa(x, need_weights=True)

    local_weights = aux["local_attn_weights"]

    future_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    future_weights = local_weights[:, :, future_mask]

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=0.0,
        rtol=0.0,
    )


def test_csa_local_window_limits_past_context():
    W = 4
    csa = make_csa(window_size=W)
    csa.eval()

    B, T, D = 2, 16, 64
    x = make_csa_input(B=B, T=T, D=D)

    _, aux = csa(x, need_weights=True)

    local_weights = aux["local_attn_weights"]

    q_pos = torch.arange(T)[:, None]
    k_pos = torch.arange(T)[None, :]
    too_old = (q_pos - k_pos) >= W

    too_old_weights = local_weights[:, :, too_old]

    assert torch.allclose(
        too_old_weights,
        torch.zeros_like(too_old_weights),
        atol=0.0,
        rtol=0.0,
    )


def test_csa_changing_future_tokens_does_not_change_past_outputs():
    m = 4
    csa = make_csa(
        compression_factor=m,
        window_size=4,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    csa.eval()

    B, T, D = 2, 16, 64
    cut = 8

    x1 = make_csa_input(B=B, T=T, D=D)
    x2 = x1.clone()
    x2[:, cut:, :] = torch.randn_like(x2[:, cut:, :])

    out1 = csa(x1)
    out2 = csa(x2)

    assert torch.allclose(
        out1[:, :cut, :],
        out2[:, :cut, :],
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# H. attention_mask tests
# ============================================================

def test_csa_attention_mask_blocks_padding_local_keys():
    csa = make_csa(window_size=8)
    csa.eval()

    B, T, D = 2, 16, 64
    x = make_csa_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[0, 5] = 0
    attention_mask[1, 7] = 0

    _, aux = csa(x, attention_mask=attention_mask, need_weights=True)

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


def test_csa_attention_mask_blocks_padding_compressed_blocks():
    m = 4
    csa = make_csa(compression_factor=m, top_k=3)
    csa.eval()

    B, T, D = 2, 16, 64
    x = make_csa_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T, dtype=torch.long)

    # For compressed block 2:
    # A = [8:12], B = [4:8]. Make both invalid.
    attention_mask[0, 4:12] = 0

    _, aux = csa(x, attention_mask=attention_mask, need_weights=True)

    assert aux["compressed_valid_mask"][0, 2] == 0

    selected = aux["topk_indices"][0]
    valid = aux["topk_mask"][0]

    assert not (selected[valid] == 2).any()


def test_csa_attention_mask_shape_validation_accepts_BT():
    csa = make_csa()

    x = make_csa_input(B=2, T=16, D=64)
    attention_mask = torch.ones(2, 16)

    out = csa(x, attention_mask=attention_mask)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.ones(16),
        torch.ones(2, 16, 1),
        torch.ones(2, 17),
    ],
)
def test_csa_attention_mask_shape_validation_rejects_bad_shapes(bad_mask):
    csa = make_csa()

    x = make_csa_input(B=2, T=16, D=64)

    with pytest.raises(ValueError):
        csa(x, attention_mask=bad_mask)


# ============================================================
# I. Combined softmax tests
# ============================================================

def test_csa_sink_plus_global_plus_local_weights_sum_to_one():
    csa = make_csa(
        attention_dropout=0.0,
        residual_dropout=0.0,
        window_size=4,
        use_attention_sink=True,
    )
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    _, aux = csa(x, need_weights=True)

    sink_sum = aux["sink_attn_weights"].sum(dim=-1)
    global_sum = aux["global_attn_weights"].sum(dim=-1)
    local_sum = aux["local_attn_weights"].sum(dim=-1)

    total = sink_sum + global_sum + local_sum

    assert torch.allclose(
        total,
        torch.ones_like(total),
        atol=1e-6,
        rtol=1e-6,
    )


def test_csa_global_plus_local_weights_sum_to_one_without_sink():
    csa = make_csa(
        attention_dropout=0.0,
        residual_dropout=0.0,
        window_size=4,
        use_attention_sink=False,
    )
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    _, aux = csa(x, need_weights=True)

    assert "sink_attn_weights" not in aux

    global_sum = aux["global_attn_weights"].sum(dim=-1)
    local_sum = aux["local_attn_weights"].sum(dim=-1)

    total = global_sum + local_sum

    assert torch.allclose(
        total,
        torch.ones_like(total),
        atol=1e-6,
        rtol=1e-6,
    )


def test_csa_no_nan_when_no_global_blocks_available():
    m = 4
    csa = make_csa(
        compression_factor=m,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_sink=True,
    )
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    out, aux = csa(x, need_weights=True)

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

    # In the first compression block, sparse global attention is empty.
    # Probability mass should be assigned to sink + local.
    total_first_block = (
        aux["sink_attn_weights"][:, :, :m, :].sum(dim=-1)
        + aux["local_attn_weights"][:, :, :m, :].sum(dim=-1)
    )

    assert torch.allclose(
        total_first_block,
        torch.ones_like(total_first_block),
        atol=1e-6,
        rtol=1e-6,
    )

def test_csa_attention_sink_is_present_and_receives_mass():
    torch.manual_seed(0)

    csa = make_csa(
        use_attention_sink=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    csa.eval()

    x = make_csa_input(B=2, T=16, D=64)

    _, aux = csa(x, need_weights=True)

    assert "sink_attn_weights" in aux

    sink_weights = aux["sink_attn_weights"]

    assert sink_weights.shape == (2, csa.n_heads, 16, 1)
    assert torch.isfinite(sink_weights).all()
    assert (sink_weights >= 0).all()
    assert sink_weights.sum() > 0


def test_csa_aux_returns_compressed_position_ids():
    m = 4
    csa = make_csa(
        compression_factor=m,
        use_attention_sink=True,
    )
    csa.eval()

    B, T, D = 2, 17, 64
    start_pos = 10

    x = make_csa_input(B=B, T=T, D=D)

    _, aux = csa(
        x,
        start_pos=start_pos,
        need_weights=True,
    )

    expected = torch.tensor(
        [13, 17, 21, 25, 26],
        device=x.device,
        dtype=torch.long,
    )

    assert "compressed_position_ids" in aux
    assert torch.equal(aux["compressed_position_ids"], expected)


def test_csa_grouped_output_projection_shape_and_gradients():
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

# ============================================================
# J. RoPE tests
# ============================================================

def test_csa_start_pos_matches_explicit_position_ids():
    csa = make_csa(
        use_rope=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    csa.eval()

    B, T, D = 2, 16, 64
    start_pos = 10

    x = make_csa_input(B=B, T=T, D=D)

    out_start = csa(x, start_pos=start_pos)
    out_explicit = csa(
        x,
        position_ids=torch.arange(start_pos, start_pos + T),
    )

    assert torch.allclose(out_start, out_explicit, atol=1e-5, rtol=1e-5)


def test_csa_no_rope_when_disabled():
    csa = make_csa(
        use_rope=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    csa.eval()

    B, T, D = 2, 16, 64
    x = make_csa_input(B=B, T=T, D=D)

    out1 = csa(x, start_pos=0)
    out2 = csa(x, position_ids=torch.arange(10, 10 + T), start_pos=10)

    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)


# ============================================================
# K. Gradient tests
# ============================================================

def test_csa_backward_computes_gradients_canonical_no_indexer_score_bias():
    """
    Canonical mode:
        use_indexer_score_bias=False

    In this mode the indexer is used for discrete top-k selection.
    Since top-k indices are not differentiable, the main loss is not expected
    to send gradients into the indexer query/key path. This is architecturally
    more faithful, but less useful for tiny end-to-end training.
    """
    csa = make_csa(
        use_bias=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
        use_indexer_score_bias=False,
        use_separate_local_kv=True,
    )

    x = make_csa_input(B=2, T=16, D=64)
    x.requires_grad_(True)

    out = csa(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    params = dict(csa.named_parameters())

    expected_grad_params = [
        "q_down_proj.weight",
        "q_up_proj.weight",
        "a_kv_proj.weight",
        "b_kv_proj.weight",
        "a_z_proj.weight",
        "b_z_proj.weight",
        "local_kv_proj.weight",
        "kv_compressor.bias_a",
        "kv_compressor.bias_b",
        "sink_k",
        "sink_v",
    ]

    for name in expected_grad_params:
        assert name in params, f"Missing parameter {name}"
        assert params[name].grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(params[name].grad).all(), f"Non-finite grad for {name}"

    # Grouped output projection gradients
    assert hasattr(csa.out_proj, "group_projs")

    for proj in csa.out_proj.group_projs:
        assert proj.weight.grad is not None
        assert torch.isfinite(proj.weight.grad).all()

        if proj.bias is not None:
            assert proj.bias.grad is not None
            assert torch.isfinite(proj.bias.grad).all()


def test_csa_backward_computes_indexer_gradients_with_score_bias_enabled():
    """
    Training-friendly mode:
        use_indexer_score_bias=True

    In this mode selected top-k scores are added to global attention logits.
    This allows gradients to flow into the indexer score path.
    """
    csa = make_csa(
        use_bias=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
        use_indexer_score_bias=True,
        use_separate_local_kv=True,
    )

    x = make_csa_input(B=2, T=16, D=64)
    x.requires_grad_(True)

    out = csa(x)
    loss = out.sum()
    loss.backward()

    params = dict(csa.named_parameters())

    expected_indexer_grad_params = [
        "index_q_up_proj.weight",
        "index_weight_proj.weight",
        "a_index_kv_proj.weight",
        "b_index_kv_proj.weight",
        "a_index_z_proj.weight",
        "b_index_z_proj.weight",
        "index_compressor.bias_a",
        "index_compressor.bias_b",
    ]

    for name in expected_indexer_grad_params:
        assert name in params, f"Missing parameter {name}"
        assert params[name].grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(params[name].grad).all(), f"Non-finite grad for {name}"


# ============================================================
# L. Interface integration
# ============================================================

def test_csa_can_replace_attention_interface():
    csa = make_csa(
        use_attention_sink=True,
        use_grouped_output_projection=True,
        output_projection_groups=4,
    )

    B, T, D = 2, 16, 64
    x = make_csa_input(B=B, T=T, D=D)

    attention_mask = torch.ones(B, T, dtype=torch.long)
    position_ids = torch.arange(T)

    out, aux = csa(
        x,
        attention_mask=attention_mask,
        position_ids=position_ids,
        start_pos=0,
        need_weights=True,
    )

    assert out.shape == x.shape

    required_keys = [
        "global_attn_weights",
        "local_attn_weights",
        "topk_indices",
        "topk_scores",
        "topk_mask",
        "compressed_valid_mask",
        "compressed_position_ids",
        "index_scores",
        "sink_attn_weights",
    ]

    for key in required_keys:
        assert key in aux

    assert torch.isfinite(out).all()