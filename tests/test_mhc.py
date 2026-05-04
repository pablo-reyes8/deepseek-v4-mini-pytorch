# @title
# ============================================================
# ManifoldHyperConnection / mHC tests
# ============================================================

import pytest
import torch
import torch.nn as nn

from src.mHC_residuals import *  
from src.deepseek_csa_attention import CSAAttention, CSAConfig
from src.deepseek_hca_attention import HCAAttention, HCAConfig
from src.transformer_modules.SwiGLU import SwiGLUMLP, SwiGLUMLPConfig
from src.transformer_modules.mha_baseline import CausalMHAConfig, CausalMultiHeadAttention

# ============================================================
# Helpers
# ============================================================

def make_mhc_config(**overrides):
    cfg = dict(
        d_model=64,
        n_hc=4,
        sinkhorn_iters=30,
        eps=1e-6,

        # New canonical Sinkhorn controls
        use_log_sinkhorn=False,
        sinkhorn_fp32=True,

        # New bounded-alpha dynamic controls
        init_alpha=1e-3,
        alpha_max=1.0,
        bounded_alpha=True,

        dynamic=True,
        static_a_stream0=4.0,
        static_a_other=-4.0,
        static_b_diag=6.0,
        static_b_offdiag=-6.0,
        static_c_stream0=0.0,
        static_c_other=-8.0,
        init_std=0.02,
    )
    cfg.update(overrides)
    return ManifoldHyperConnectionConfig(**cfg)


def make_mhc(**overrides):
    return ManifoldHyperConnection(make_mhc_config(**overrides))


def make_X(B=2, T=8, n_hc=4, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, n_hc, D, dtype=dtype, device=device)


def make_x(B=2, T=8, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


class TinyLinearSublayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(x)


# ============================================================
# A. Config tests
# ============================================================

def test_valid_mhc_config_builds():
    config = ManifoldHyperConnectionConfig(
        d_model=256,
        n_hc=4,
        sinkhorn_iters=20,
        eps=1e-6,
        use_log_sinkhorn=False,
        sinkhorn_fp32=True,
        init_alpha=1e-3,
        alpha_max=1.0,
        bounded_alpha=True,
        dynamic=True,
    )

    mhc = ManifoldHyperConnection(config)

    assert mhc.d_model == 256
    assert mhc.n_hc == 4
    assert mhc.sinkhorn_iters == 20
    assert mhc.eps == 1e-6
    assert mhc.dynamic is True
    assert mhc.use_log_sinkhorn is False
    assert mhc.sinkhorn_fp32 is True
    assert mhc.bounded_alpha is True
    assert mhc.alpha_max == 1.0


@pytest.mark.parametrize("d_model", [0, -1, -64])
def test_invalid_mhc_d_model_raises(d_model):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(make_mhc_config(d_model=d_model))


@pytest.mark.parametrize("n_hc", [0, -1, 1])
def test_invalid_n_hc_raises(n_hc):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(make_mhc_config(n_hc=n_hc))


@pytest.mark.parametrize("sinkhorn_iters", [0, -1])
def test_invalid_sinkhorn_iters_raises(sinkhorn_iters):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(make_mhc_config(sinkhorn_iters=sinkhorn_iters))


@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_invalid_eps_raises(eps):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(make_mhc_config(eps=eps))


# ============================================================
# B. Sinkhorn tests
# ============================================================

def test_sinkhorn_output_shape():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = sinkhorn(logits, n_iters=30, eps=1e-6)

    assert B_mat.shape == logits.shape


def test_sinkhorn_non_negative():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = sinkhorn(logits, n_iters=30, eps=1e-6)

    assert (B_mat >= 0).all()


def test_sinkhorn_rows_sum_to_one():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = sinkhorn(logits, n_iters=50, eps=1e-6)

    row_sums = B_mat.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)


def test_sinkhorn_cols_sum_to_one():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = sinkhorn(logits, n_iters=50, eps=1e-6)

    col_sums = B_mat.sum(dim=-2)

    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)


def test_sinkhorn_finite_for_large_logits():
    logits = torch.randn(2, 8, 4, 4) * 1_000.0

    B_mat = sinkhorn(logits, n_iters=50, eps=1e-6)

    assert torch.isfinite(B_mat).all()
    assert (B_mat >= 0).all()


# ============================================================
# C. A/B/C generation tests
# ============================================================

def test_abc_shapes():
    B, T, N, D = 2, 8, 4, 64

    mhc = make_mhc(d_model=D, n_hc=N)
    X = make_X(B=B, T=T, n_hc=N, D=D)

    A, B_mat, C = mhc.compute_ABC(X)

    assert A.shape == (B, T, 1, N)
    assert B_mat.shape == (B, T, N, N)
    assert C.shape == (B, T, N, 1)


def test_A_bounds():
    mhc = make_mhc()
    X = make_X()

    A, _, _ = mhc.compute_ABC(X)

    assert A.min() >= 0
    assert A.max() <= 1


def test_C_bounds():
    mhc = make_mhc()
    X = make_X()

    _, _, C = mhc.compute_ABC(X)

    assert C.min() >= 0
    assert C.max() <= 2


def test_B_doubly_stochastic():
    mhc = make_mhc(sinkhorn_iters=50)
    X = make_X()

    _, B_mat, _ = mhc.compute_ABC(X)

    assert (B_mat >= 0).all()

    row_sums = B_mat.sum(dim=-1)
    col_sums = B_mat.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)


def test_dynamic_false_same_abc_for_different_X():
    mhc = make_mhc(dynamic=False)

    X1 = make_X()
    X2 = make_X() * 10.0 + 5.0

    A1, B1, C1 = mhc.compute_ABC(X1)
    A2, B2, C2 = mhc.compute_ABC(X2)

    assert torch.allclose(A1, A2, atol=0.0, rtol=0.0)
    assert torch.allclose(B1, B2, atol=0.0, rtol=0.0)
    assert torch.allclose(C1, C2, atol=0.0, rtol=0.0)


def test_dynamic_true_abc_changes_with_X():
    mhc = make_mhc(dynamic=True, init_alpha=1e-1)

    X1 = make_X()
    X2 = make_X() * 10.0 + 5.0

    A1, B1, C1 = mhc.compute_ABC(X1)
    A2, B2, C2 = mhc.compute_ABC(X2)

    diff = (
        (A1 - A2).abs().sum()
        + (B1 - B2).abs().sum()
        + (C1 - C2).abs().sum()
    )

    assert diff > 0


# ============================================================
# D. Forward tests
# ============================================================

def test_forward_output_shape_matches_input():
    mhc = make_mhc()
    X = make_X()

    X_next = mhc(X, sublayer=lambda x: x)

    assert X_next.shape == X.shape


@pytest.mark.parametrize(
    "bad_X",
    [
        torch.randn(2, 8, 64),
        torch.randn(2, 8, 4, 64, 1),
    ],
)
def test_mhc_rejects_wrong_input_rank(bad_X):
    mhc = make_mhc()

    with pytest.raises(ValueError):
        mhc(bad_X, sublayer=lambda x: x)


def test_rejects_wrong_n_hc():
    mhc = make_mhc(n_hc=4)

    X = torch.randn(2, 8, 3, 64)

    with pytest.raises(ValueError):
        mhc(X, sublayer=lambda x: x)


def test_rejects_wrong_d_model():
    mhc = make_mhc(d_model=64)

    X = torch.randn(2, 8, 4, 32)

    with pytest.raises(ValueError):
        mhc(X, sublayer=lambda x: x)


def test_sublayer_receives_BTD_tensor():
    B, T, N, D = 2, 8, 4, 64

    mhc = make_mhc(d_model=D, n_hc=N)
    X = make_X(B=B, T=T, n_hc=N, D=D)

    seen = {}

    def sublayer(x_sub):
        seen["shape"] = x_sub.shape
        return x_sub

    _ = mhc(X, sublayer=sublayer)

    assert seen["shape"] == (B, T, D)


def test_sublayer_output_must_match_shape():
    mhc = make_mhc()
    X = make_X()

    def bad_sublayer(x_sub):
        B, T, D = x_sub.shape
        return torch.randn(B, T, D + 1)

    with pytest.raises(ValueError):
        mhc(X, sublayer=bad_sublayer)


# ============================================================
# E. Mathematical update tests
# ============================================================

def test_forward_matches_manual_computation():
    mhc = make_mhc(dynamic=True)

    X = make_X()

    def sublayer(x_sub):
        return x_sub ** 2

    X_next, aux = mhc(X, sublayer=sublayer, return_aux=True)

    A = aux["A"]
    B_mat = aux["B"]
    C = aux["C"]

    x_sub = torch.einsum("btan,btnd->btad", A, X).squeeze(2)
    y_sub = sublayer(x_sub)
    mixed_X = torch.einsum("btij,btjd->btid", B_mat, X)
    expected = mixed_X + C * y_sub[:, :, None, :]

    assert torch.allclose(X_next, expected, atol=1e-6, rtol=1e-5)

    assert "alpha_A" in aux
    assert "alpha_B" in aux
    assert "alpha_C" in aux


def test_zero_sublayer_reduces_to_B_mixing():
    mhc = make_mhc()
    X = make_X()

    def zero_sublayer(x_sub):
        return torch.zeros_like(x_sub)

    X_next, aux = mhc(X, sublayer=zero_sublayer, return_aux=True)

    expected = torch.einsum("btij,btjd->btid", aux["B"], X)

    assert torch.allclose(X_next, expected, atol=1e-6, rtol=1e-5)


def test_zero_input_zero_sublayer_returns_zero():
    mhc = make_mhc()

    X = torch.zeros(2, 8, 4, 64)

    def zero_sublayer(x_sub):
        return torch.zeros_like(x_sub)

    X_next = mhc(X, sublayer=zero_sublayer)

    assert torch.allclose(X_next, torch.zeros_like(X_next), atol=1e-7, rtol=1e-7)


# ============================================================
# F. Residual-like initialization tests
# ============================================================

def test_initial_A_prefers_stream_zero():
    mhc = make_mhc(dynamic=False)
    X = make_X()

    A, _, _ = mhc.compute_ABC(X)

    stream0 = A[..., 0]
    others = A[..., 1:]

    assert (stream0[..., None] > others).all()


def test_initial_C_prefers_stream_zero():
    mhc = make_mhc(dynamic=False)
    X = make_X()

    _, _, C = mhc.compute_ABC(X)

    stream0 = C[..., 0, 0]
    others = C[..., 1:, 0]

    assert (stream0[..., None] > others).all()


def test_initial_B_close_to_identity():
    mhc = make_mhc(dynamic=False, sinkhorn_iters=100)
    X = make_X()

    _, B_mat, _ = mhc.compute_ABC(X)

    B_mean = B_mat.mean(dim=(0, 1))

    diag = torch.diagonal(B_mean)
    offdiag = B_mean[~torch.eye(mhc.n_hc, dtype=torch.bool)]

    assert diag.mean() > offdiag.mean()


def test_initial_forward_approximates_residual_on_stream_zero():
    B, T, D, N = 2, 8, 64, 4

    mhc = make_mhc(
        d_model=D,
        n_hc=N,
        dynamic=False,
        sinkhorn_iters=100,
        static_a_stream0=8.0,
        static_a_other=-8.0,
        static_b_diag=10.0,
        static_b_offdiag=-10.0,
        static_c_stream0=0.0,
        static_c_other=-10.0,
    )

    x = make_x(B=B, T=T, D=D)
    X = expand_residual_stream(x, n_hc=N, mode="first")

    def sublayer(x_sub):
        return 0.1 * x_sub

    X_next = mhc(X, sublayer=sublayer)

    expected_stream0 = x + sublayer(x)

    assert torch.allclose(
        X_next[:, :, 0, :],
        expected_stream0,
        atol=3e-3,
        rtol=3e-3,
    )


# ============================================================
# G. Expand / collapse tests
# ============================================================

def test_expand_residual_stream_shape():
    x = make_x(B=2, T=8, D=64)

    X = expand_residual_stream(x, n_hc=4, mode="first")

    assert X.shape == (2, 8, 4, 64)


def test_expand_residual_stream_puts_x_in_stream_zero():
    x = make_x(B=2, T=8, D=64)

    X = expand_residual_stream(x, n_hc=4, mode="first")

    assert torch.allclose(X[:, :, 0, :], x, atol=0.0, rtol=0.0)
    assert torch.allclose(X[:, :, 1:, :], torch.zeros_like(X[:, :, 1:, :]), atol=0.0, rtol=0.0)


def test_collapse_residual_stream_mean_shape():
    X = make_X(B=2, T=8, n_hc=4, D=64)

    x = collapse_residual_stream(X, mode="mean")

    assert x.shape == (2, 8, 64)


def test_collapse_residual_stream_first_shape():
    X = make_X(B=2, T=8, n_hc=4, D=64)

    x = collapse_residual_stream(X, mode="first")

    assert x.shape == (2, 8, 64)
    assert torch.allclose(x, X[:, :, 0, :], atol=0.0, rtol=0.0)


# ============================================================
# H. Gradient / numerical tests
# ============================================================

def test_mhc_backward_computes_gradients():
    B, T, D, N = 2, 8, 64, 4

    mhc = make_mhc(d_model=D, n_hc=N, dynamic=True, init_alpha=1e-2)

    sublayer = TinyLinearSublayer(D)

    X = make_X(B=B, T=T, n_hc=N, D=D)
    X.requires_grad_(True)

    X_next, aux = mhc(X, sublayer=sublayer, return_aux=True)

    loss = X_next.sum()
    loss.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()

    expected_params = [
        "static_A",
        "static_B",
        "static_C",
        "alpha_A_raw",
        "alpha_B_raw",
        "alpha_C_raw",
        "dynamic_A.weight",
        "dynamic_B.weight",
        "dynamic_C.weight",
        "param_norm.weight",
    ]

    params = dict(mhc.named_parameters())

    for name in expected_params:
        assert name in params, f"Missing parameter {name}"
        assert params[name].grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(params[name].grad).all(), f"Non-finite grad for {name}"

    for name, param in sublayer.named_parameters():
        assert param.grad is not None, f"Missing sublayer grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite sublayer grad for {name}"

    assert "alpha_A" in aux
    assert "alpha_B" in aux
    assert "alpha_C" in aux


def test_no_nan_large_inputs():
    mhc = make_mhc(dynamic=True)

    X = make_X() * 1_000.0

    X_next = mhc(X, sublayer=lambda x: 0.1 * x)

    assert torch.isfinite(X_next).all()


def test_no_nan_small_inputs():
    mhc = make_mhc(dynamic=True)

    X = make_X() * 1e-8

    X_next = mhc(X, sublayer=lambda x: 0.1 * x)

    assert torch.isfinite(X_next).all()


# ============================================================
# I. Integration with real sublayers
# ============================================================

def test_mhc_wraps_swiglu_mlp():
    B, T, D, N = 2, 8, 64, 4

    mhc = make_mhc(d_model=D, n_hc=N)
    mlp = SwiGLUMLP(
        SwiGLUMLPConfig(
            d_model=D,
            hidden_dim=256,
            dropout=0.0,
            use_bias=False,
            init_std=0.02,
        )
    )

    X = make_X(B=B, T=T, n_hc=N, D=D)

    X_next = mhc(X, sublayer=mlp)

    assert X_next.shape == X.shape
    assert torch.isfinite(X_next).all()


def test_mhc_wraps_causal_mha_with_lambda():
    B, T, D, N = 2, 8, 64, 4

    mhc = make_mhc(d_model=D, n_hc=N)

    attn = CausalMultiHeadAttention(
        CausalMHAConfig(
            d_model=D,
            n_heads=4,
            head_dim=16,
            attention_dropout=0.0,
            residual_dropout=0.0,
            use_bias=False,
            use_rope=True,
            rotary_dim=16,
            max_seq_len=T,
            init_std=0.02,
        )
    )

    X = make_X(B=B, T=T, n_hc=N, D=D)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    position_ids = torch.arange(T)

    X_next = mhc(
        X,
        sublayer=lambda x_sub: attn(
            x_sub,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ),
    )

    assert X_next.shape == X.shape
    assert torch.isfinite(X_next).all()


def test_mhc_wraps_hca_attention():
    B, T, D, N = 2, 8, 64, 4

    mhc = make_mhc(d_model=D, n_hc=N)

    hca = HCAAttention(
        HCAConfig(
            d_model=D,
            n_heads=4,
            head_dim=16,
            compression_factor=4,
            window_size=4,
            attention_dropout=0.0,
            residual_dropout=0.0,
            use_bias=False,
            use_rope=True,
            rotary_dim=16,
            max_seq_len=T,
            init_std=0.02,
        )
    )

    X = make_X(B=B, T=T, n_hc=N, D=D)
    attention_mask = torch.ones(B, T, dtype=torch.long)

    X_next = mhc(
        X,
        sublayer=lambda x_sub: hca(
            x_sub,
            attention_mask=attention_mask,
            start_pos=0,
        ),
    )

    assert X_next.shape == X.shape
    assert torch.isfinite(X_next).all()


def test_mhc_wraps_csa_attention():
    B, T, D, N = 2, 8, 64, 4

    mhc = make_mhc(d_model=D, n_hc=N)

    csa = CSAAttention(
        CSAConfig(
            d_model=D,
            n_heads=4,
            head_dim=16,
            compression_factor=4,
            top_k=2,
            window_size=4,
            indexer_dim=8,
            n_indexer_heads=2,
            query_compression_dim=16,
            attention_dropout=0.0,
            residual_dropout=0.0,
            use_bias=False,
            use_rope=True,
            rotary_dim=16,
            max_seq_len=T,
            init_std=0.02,
        )
    )

    X = make_X(B=B, T=T, n_hc=N, D=D)
    attention_mask = torch.ones(B, T, dtype=torch.long)

    X_next = mhc(
        X,
        sublayer=lambda x_sub: csa(
            x_sub,
            attention_mask=attention_mask,
            start_pos=0,
        ),
    )

    assert X_next.shape == X.shape
    assert torch.isfinite(X_next).all()


def test_log_sinkhorn_output_shape():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = log_sinkhorn(logits, n_iters=30)

    assert B_mat.shape == logits.shape


def test_log_sinkhorn_non_negative():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = log_sinkhorn(logits, n_iters=30)

    assert (B_mat >= 0).all()


def test_log_sinkhorn_rows_cols_sum_to_one():
    logits = torch.randn(2, 8, 4, 4)

    B_mat = log_sinkhorn(logits, n_iters=50)

    row_sums = B_mat.sum(dim=-1)
    col_sums = B_mat.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)


def test_compute_ABC_with_log_sinkhorn():
    mhc = make_mhc(use_log_sinkhorn=True, sinkhorn_iters=50)
    X = make_X()

    _, B_mat, _ = mhc.compute_ABC(X)

    assert torch.isfinite(B_mat).all()
    assert (B_mat >= 0).all()

    row_sums = B_mat.sum(dim=-1)
    col_sums = B_mat.sum(dim=-2)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)

def test_bounded_alpha_initializes_near_init_alpha():
    init_alpha = 1e-3

    mhc = make_mhc(
        bounded_alpha=True,
        alpha_max=1.0,
        init_alpha=init_alpha,
    )

    alpha_A, alpha_B, alpha_C = mhc.get_alpha_values()

    expected = torch.tensor(init_alpha, dtype=alpha_A.dtype)

    assert torch.allclose(alpha_A, expected, atol=1e-7, rtol=1e-5)
    assert torch.allclose(alpha_B, expected, atol=1e-7, rtol=1e-5)
    assert torch.allclose(alpha_C, expected, atol=1e-7, rtol=1e-5)


def test_bounded_alpha_is_within_bounds_after_manual_large_raw():
    mhc = make_mhc(
        bounded_alpha=True,
        alpha_max=0.25,
        init_alpha=1e-3,
    )

    with torch.no_grad():
        mhc.alpha_A_raw.fill_(100.0)
        mhc.alpha_B_raw.fill_(-100.0)
        mhc.alpha_C_raw.fill_(50.0)

    alpha_A, alpha_B, alpha_C = mhc.get_alpha_values()

    assert alpha_A <= 0.25
    assert alpha_A >= -0.25

    assert alpha_B <= 0.25
    assert alpha_B >= -0.25

    assert alpha_C <= 0.25

def test_pre_mix_output_shape():
    B, T, N, D = 2, 8, 4, 64

    mhc = make_mhc(d_model=D, n_hc=N)
    X = make_X(B=B, T=T, n_hc=N, D=D)

    x_sub = mhc.pre_mix(X)

    assert x_sub.shape == (B, T, D)
    assert torch.isfinite(x_sub).all()


def test_pre_mix_with_precomputed_A_matches_forward_aux_x_sub():
    mhc = make_mhc(dynamic=True)
    X = make_X()

    A, B_mat, C = mhc.compute_ABC(X)

    x_sub_1 = mhc.pre_mix(X, A=A)

    _, aux = mhc(
        X,
        sublayer=lambda x: x,
        return_aux=True,
    )

    x_sub_2 = aux["x_sub"]

    assert torch.allclose(x_sub_1, x_sub_2, atol=1e-6, rtol=1e-5)


def test_pre_mix_return_aux_contains_ABC_when_A_not_provided():
    mhc = make_mhc()
    X = make_X()

    x_sub, aux = mhc.pre_mix(X, return_aux=True)

    assert x_sub.shape == (X.shape[0], X.shape[1], X.shape[-1])
    assert "A" in aux
    assert "B" in aux
    assert "C" in aux

    assert aux["A"].shape == (X.shape[0], X.shape[1], 1, X.shape[2])
    assert aux["B"].shape == (X.shape[0], X.shape[1], X.shape[2], X.shape[2])
    assert aux["C"].shape == (X.shape[0], X.shape[1], X.shape[2], 1)


def test_update_output_shape():
    B, T, N, D = 2, 8, 4, 64

    mhc = make_mhc(d_model=D, n_hc=N)
    X = make_X(B=B, T=T, n_hc=N, D=D)
    y_sub = torch.randn(B, T, D)

    X_next = mhc.update(X, y_sub)

    assert X_next.shape == X.shape
    assert torch.isfinite(X_next).all()


def test_update_with_precomputed_BC_matches_manual_computation():
    mhc = make_mhc(dynamic=True)
    X = make_X()

    A, B_mat, C = mhc.compute_ABC(X)
    y_sub = torch.randn(X.shape[0], X.shape[1], X.shape[-1])

    X_next = mhc.update(X, y_sub, B_mat=B_mat, C=C)

    expected = (
        torch.einsum("btij,btjd->btid", B_mat, X)
        + C * y_sub[:, :, None, :]
    )

    assert torch.allclose(X_next, expected, atol=1e-6, rtol=1e-5)


def test_forward_matches_modular_pre_mix_update():
    mhc = make_mhc(dynamic=True)
    X = make_X()

    def sublayer(x_sub):
        return x_sub ** 2

    X_forward = mhc(X, sublayer=sublayer)

    A, B_mat, C = mhc.compute_ABC(X)
    x_sub = mhc.pre_mix(X, A=A)
    y_sub = sublayer(x_sub)
    X_modular = mhc.update(X, y_sub, B_mat=B_mat, C=C)

    assert torch.allclose(X_forward, X_modular, atol=1e-6, rtol=1e-5)


def test_post_and_residual_mix_alias_matches_update():
    mhc = make_mhc()
    X = make_X()
    y_sub = torch.randn(X.shape[0], X.shape[1], X.shape[-1])

    _, B_mat, C = mhc.compute_ABC(X)

    out_update = mhc.update(X, y_sub, B_mat=B_mat, C=C)
    out_alias = mhc.post_and_residual_mix(X, y_sub, B_mat=B_mat, C=C)

    assert torch.allclose(out_update, out_alias, atol=0.0, rtol=0.0)


def test_update_rejects_bad_y_sub_shape():
    mhc = make_mhc()
    X = make_X()

    bad_y = torch.randn(X.shape[0], X.shape[1], X.shape[-1] + 1)

    with pytest.raises(ValueError):
        mhc.update(X, bad_y)


def test_pre_mix_rejects_bad_A_shape():
    mhc = make_mhc()
    X = make_X()

    bad_A = torch.randn(X.shape[0], X.shape[1], mhc.n_hc)

    with pytest.raises(ValueError):
        mhc.pre_mix(X, A=bad_A)


def test_update_rejects_bad_B_shape():
    mhc = make_mhc()
    X = make_X()
    y_sub = torch.randn(X.shape[0], X.shape[1], X.shape[-1])

    bad_B = torch.randn(X.shape[0], X.shape[1], mhc.n_hc, mhc.n_hc + 1)

    with pytest.raises(ValueError):
        mhc.update(X, y_sub, B_mat=bad_B)


def test_update_rejects_bad_C_shape():
    mhc = make_mhc()
    X = make_X()
    y_sub = torch.randn(X.shape[0], X.shape[1], X.shape[-1])

    bad_C = torch.randn(X.shape[0], X.shape[1], mhc.n_hc)

    with pytest.raises(ValueError):
        mhc.update(X, y_sub, C=bad_C)

def test_collapse_residual_stream_sum_shape():
    X = make_X(B=2, T=8, n_hc=4, D=64)

    x = collapse_residual_stream(X, mode="sum")

    assert x.shape == (2, 8, 64)
    assert torch.allclose(x, X.sum(dim=2), atol=0.0, rtol=0.0)

@pytest.mark.parametrize("init_alpha", [-1e-3, -1.0])
def test_invalid_init_alpha_raises(init_alpha):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(make_mhc_config(init_alpha=init_alpha))


@pytest.mark.parametrize("alpha_max", [0.0, -1.0])
def test_invalid_alpha_max_raises(alpha_max):
    with pytest.raises(ValueError):
        ManifoldHyperConnection(make_mhc_config(alpha_max=alpha_max))


def test_invalid_bounded_alpha_when_init_alpha_ge_alpha_max_raises():
    with pytest.raises(ValueError):
        ManifoldHyperConnection(
            make_mhc_config(
                bounded_alpha=True,
                init_alpha=1.0,
                alpha_max=1.0,
            )
        )


def test_unbounded_alpha_allows_init_alpha_ge_alpha_max():
    mhc = ManifoldHyperConnection(
        make_mhc_config(
            bounded_alpha=False,
            init_alpha=2.0,
            alpha_max=1.0,
        )
    )

    alpha_A, alpha_B, alpha_C = mhc.get_alpha_values()

    expected = torch.tensor(2.0, device=alpha_A.device, dtype=alpha_A.dtype)

    assert torch.allclose(alpha_A, expected)
    assert torch.allclose(alpha_B, expected)
    assert torch.allclose(alpha_C, expected)
