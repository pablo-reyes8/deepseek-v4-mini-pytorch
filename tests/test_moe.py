
# ============================================================
# DeepSeekMoE-style FFN tests
# ============================================================

import pytest
import torch
import torch.nn as nn

from src.deepseek_moe import * 
from src.mHC_residuals import ManifoldHyperConnection, ManifoldHyperConnectionConfig
from src.transformer_modules.RMSNorm import RMSNorm


def make_X(B=2, T=8, n_hc=4, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, n_hc, D, dtype=dtype, device=device)

# ============================================================
# Helpers
# ============================================================

def make_moe_config(**overrides):
    cfg = dict(
        d_model=64,

        num_experts=4,
        top_k=2,

        expert_hidden_dim=128,
        expert_expansion_factor=4.0,
        expert_multiple_of=1,

        shared_experts=1,
        shared_hidden_dim=None,
        shared_expansion_factor=4.0,

        router_score_fn="sqrt_softplus",
        normalize_topk_weights=True,
        topk_weight_scale=1.0,

        router_type="learned",  # "learned" | "hash"
        router_jitter_noise=0.0,

        routed_scale=1.0,
        shared_scale=1.0,

        dropout=0.0,
        use_bias=True,
        init_std=0.02,

        balance_loss_weight=0.01,
        sequence_balance_loss_weight=0.01,

        eps=1e-9,
    )
    cfg.update(overrides)
    return DeepSeekMoEConfig(**cfg)


def make_moe(**overrides):
    return DeepSeekMoE(make_moe_config(**overrides))


def make_moe_input(B=2, T=8, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


def zero_router(moe: DeepSeekMoE):
    with torch.no_grad():
        moe.router.weight.zero_()
        if moe.router.bias is not None:
            moe.router.bias.zero_()


def force_router_bias(moe: DeepSeekMoE, bias_values):
    """
    Forces routing to be independent of x by zeroing router weights
    and setting router bias.
    Requires use_bias=True.
    """
    assert moe.router.bias is not None, "This helper requires router bias."

    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.copy_(torch.tensor(
            bias_values,
            dtype=moe.router.bias.dtype,
            device=moe.router.bias.device,
        ))


# ============================================================
# A. Config tests
# ============================================================

def test_valid_moe_config_builds():
    config = DeepSeekMoEConfig(
        d_model=256,
        num_experts=8,
        top_k=2,
        expert_expansion_factor=4.0,
        shared_experts=2,
        router_score_fn="sqrt_softplus",
        normalize_topk_weights=True,
        topk_weight_scale=1.0,
        router_type="learned",
        routed_scale=1.0,
        shared_scale=1.0,
        balance_loss_weight=0.01,
        sequence_balance_loss_weight=0.01,
    )

    moe = DeepSeekMoE(config)

    assert moe.d_model == 256
    assert moe.num_experts == 8
    assert moe.top_k == 2
    assert moe.router_score_fn == "sqrt_softplus"
    assert moe.router_type == "learned"
    assert moe.routed_scale == 1.0
    assert moe.shared_scale == 1.0
    assert moe.topk_weight_scale == 1.0

    assert len(moe.experts) == 8
    assert hasattr(moe, "shared_experts")
    assert len(moe.shared_experts) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"d_model": -1},
        {"num_experts": 0},
        {"num_experts": -1},
        {"top_k": 0},
        {"top_k": -1},
        {"num_experts": 4, "top_k": 5},
    ],
)
def test_invalid_moe_model_dims_raise(kwargs):
    with pytest.raises(ValueError):
        DeepSeekMoE(make_moe_config(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expert_hidden_dim": 0},
        {"expert_hidden_dim": -1},
        {"expert_expansion_factor": 0.0},
        {"expert_expansion_factor": -1.0},
        {"expert_multiple_of": 0},
        {"expert_multiple_of": -1},
        {"shared_experts": -1},
        {"shared_hidden_dim": 0},
        {"shared_hidden_dim": -1},
        {"shared_expansion_factor": 0.0},
        {"shared_expansion_factor": -1.0},
    ],
)
def test_invalid_expert_config_raises(kwargs):
    with pytest.raises(ValueError):
        DeepSeekMoE(make_moe_config(**kwargs))




@pytest.mark.parametrize("router_score_fn", ["bad", "relu", "topk_softmax"])
def test_invalid_router_score_fn_raises(router_score_fn):
    with pytest.raises(ValueError):
        DeepSeekMoE(make_moe_config(router_score_fn=router_score_fn))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": 1.5},
        {"init_std": 0.0},
        {"init_std": -0.01},
        {"balance_loss_weight": -0.1},
        {"sequence_balance_loss_weight": -0.1},
        {"router_jitter_noise": -0.1},
        {"topk_weight_scale": 0.0},
        {"topk_weight_scale": -1.0},
        {"routed_scale": -1.0},
        {"shared_scale": -1.0},
        {"eps": 0.0},
        {"eps": -1e-9},
    ],
)
def test_invalid_dropout_init_balance_raise(kwargs):
    with pytest.raises(ValueError):
        DeepSeekMoE(make_moe_config(**kwargs))


@pytest.mark.parametrize("router_type", ["bad", "dense", "random"])
def test_invalid_router_type_raises(router_type):
    with pytest.raises(ValueError):
        DeepSeekMoE(make_moe_config(router_type=router_type))

# ============================================================
# B. Internal structure tests
# ============================================================

def test_moe_has_router_and_experts():
    config = make_moe_config(num_experts=4)
    moe = DeepSeekMoE(config)

    assert hasattr(moe, "router")
    assert hasattr(moe, "experts")
    assert len(moe.experts) == config.num_experts


def test_all_experts_are_swiglu_mlp():
    moe = make_moe(num_experts=4)

    for expert in moe.experts:
        assert isinstance(expert, SwiGLUMLP)


def test_shared_experts_exist_when_enabled():
    moe = make_moe(shared_experts=2)

    assert hasattr(moe, "shared_experts")
    assert isinstance(moe.shared_experts, nn.ModuleList)
    assert len(moe.shared_experts) == 2

    for expert in moe.shared_experts:
        assert isinstance(expert, SwiGLUMLP)


def test_shared_experts_empty_when_disabled():
    moe = make_moe(shared_experts=0)

    assert hasattr(moe, "shared_experts")
    assert isinstance(moe.shared_experts, nn.ModuleList)
    assert len(moe.shared_experts) == 0


# ============================================================
# C. Router tests
# ============================================================

def test_router_logits_shape():
    B, T, D = 2, 8, 64
    E = 4

    moe = make_moe(d_model=D, num_experts=E)
    x = make_moe_input(B=B, T=T, D=D)

    router_logits = moe._compute_router_logits(x)

    assert router_logits.shape == (B, T, E)


@pytest.mark.parametrize("router_score_fn", ["softmax", "sigmoid", "sqrt_softplus"])
def test_router_scores_positive_for_all_score_fns(router_score_fn):
    moe = make_moe(router_score_fn=router_score_fn)

    x = make_moe_input()
    logits = moe._compute_router_logits(x)
    scores = moe._router_scores(logits)

    assert scores.shape == logits.shape
    assert (scores >= 0).all()
    assert torch.isfinite(scores).all()


def test_softmax_router_scores_sum_to_one():
    moe = make_moe(router_score_fn="softmax")

    x = make_moe_input()
    logits = moe._compute_router_logits(x)
    scores = moe._router_scores(logits)

    assert torch.allclose(
        scores.sum(dim=-1),
        torch.ones_like(scores.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_topk_shapes():
    B, T, D = 2, 8, 64
    E, K = 4, 2

    moe = make_moe(num_experts=E, top_k=K)

    x = make_moe_input(B=B, T=T, D=D)
    logits = moe._compute_router_logits(x)
    scores = moe._router_scores(logits)

    topk_scores, topk_indices, topk_weights = moe._topk_routing(scores)

    assert topk_scores.shape == (B, T, K)
    assert topk_indices.shape == (B, T, K)
    assert topk_weights.shape == (B, T, K)


def test_topk_indices_in_valid_range():
    E = 4
    moe = make_moe(num_experts=E, top_k=2)

    x = make_moe_input()
    scores = moe._router_scores(moe._compute_router_logits(x))
    _, topk_indices, _ = moe._topk_routing(scores)

    assert (topk_indices >= 0).all()
    assert (topk_indices < E).all()


def test_topk_weights_sum_to_scale_when_normalized():
    scale = 1.25

    moe = make_moe(
        normalize_topk_weights=True,
        topk_weight_scale=scale,
    )

    x = make_moe_input()
    scores = moe._router_scores(moe._compute_router_logits(x))
    _, _, topk_weights = moe._topk_routing(scores)

    expected = torch.full_like(
        topk_weights.sum(dim=-1),
        fill_value=scale,
    )

    assert torch.allclose(
        topk_weights.sum(dim=-1),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_topk_weights_not_necessarily_sum_to_one_when_not_normalized():
    moe = make_moe(
        normalize_topk_weights=False,
        router_score_fn="sqrt_softplus",
    )

    x = make_moe_input()
    scores = moe._router_scores(moe._compute_router_logits(x))
    _, _, topk_weights = moe._topk_routing(scores)

    assert (topk_weights >= 0).all()
    assert torch.isfinite(topk_weights).all()


# ============================================================
# D. Forward tests
# ============================================================

def test_moe_output_shape_matches_input():
    moe = make_moe()

    x = make_moe_input()
    out = moe(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "bad_x",
    [
        torch.randn(8, 64),
        torch.randn(2, 8, 64, 1),
    ],
)
def test_moe_rejects_wrong_input_rank(bad_x):
    moe = make_moe()

    with pytest.raises(ValueError):
        moe(bad_x)


def test_moe_rejects_wrong_hidden_size():
    moe = make_moe(d_model=64)

    x = torch.randn(2, 8, 32)

    with pytest.raises(ValueError):
        moe(x)


def test_forward_returns_aux_when_requested():
    B, T, D = 2, 8, 64
    E, K = 4, 2

    moe = make_moe(d_model=D, num_experts=E, top_k=K)
    x = make_moe_input(B=B, T=T, D=D)

    out, aux = moe(x, return_aux=True)

    assert out.shape == x.shape
    assert isinstance(aux, dict)

    required_keys = [
        "router_logits",
        "router_scores",
        "topk_indices",
        "topk_scores",
        "topk_weights",

        "expert_counts",
        "expert_fraction",
        "sequence_expert_counts",
        "sequence_expert_fraction",

        "router_entropy",

        "balance_loss",
        "raw_balance_loss",
        "sequence_balance_loss",
        "sequence_raw_balance_loss",
        "total_balance_loss",

        "routed_out",
        "shared_out",
        "routed_scale",
        "shared_scale",
        "router_type",
    ]

    for key in required_keys:
        assert key in aux, f"Missing aux key: {key}. Available keys: {list(aux.keys())}"

    assert aux["router_logits"].shape == (B, T, E)
    assert aux["router_scores"].shape == (B, T, E)
    assert aux["topk_indices"].shape == (B, T, K)
    assert aux["topk_scores"].shape == (B, T, K)
    assert aux["topk_weights"].shape == (B, T, K)

    assert aux["expert_counts"].shape == (E,)
    assert aux["expert_fraction"].shape == (E,)

    assert aux["sequence_expert_counts"].shape == (B, E)
    assert aux["sequence_expert_fraction"].shape == (B, E)

    assert aux["routed_out"].shape == x.shape
    assert aux["shared_out"].shape == x.shape

    assert aux["balance_loss"].dim() == 0
    assert aux["raw_balance_loss"].dim() == 0
    assert aux["sequence_balance_loss"].dim() == 0
    assert aux["sequence_raw_balance_loss"].dim() == 0
    assert aux["total_balance_loss"].dim() == 0

    assert aux["router_type"] == moe.router_type


def test_forward_without_aux_returns_tensor():
    moe = make_moe()
    x = make_moe_input()

    out = moe(x, return_aux=False)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


# ============================================================
# E. Dispatch / combine tests
# ============================================================

def test_only_selected_experts_contribute():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=1,
        shared_experts=0,
        dropout=0.0,
        use_bias=True,
        router_score_fn="softmax",
        normalize_topk_weights=True,
    )
    moe.eval()

    # Force expert 0 to always win.
    force_router_bias(moe, [10.0, 0.0, -1.0, -2.0])

    x = make_moe_input(B=B, T=T, D=D)

    out = moe(x)
    expected = moe.experts[0](x)

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_top2_combination_matches_manual():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=2,
        shared_experts=0,
        dropout=0.0,
        use_bias=True,
        router_score_fn="softmax",
        normalize_topk_weights=True,
    )
    moe.eval()

    # Force top-2: expert 0 then expert 1.
    force_router_bias(moe, [4.0, 2.0, -10.0, -20.0])

    x = make_moe_input(B=B, T=T, D=D)

    out, aux = moe(x, return_aux=True)

    router_scores = aux["router_scores"]
    top2_scores = router_scores[..., [0, 1]]
    weights = top2_scores / top2_scores.sum(dim=-1, keepdim=True)

    expected = (
        weights[..., 0:1] * moe.experts[0](x)
        + weights[..., 1:2] * moe.experts[1](x)
    )

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_expert_with_no_tokens_is_skipped_safely():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=1,
        shared_experts=0,
        use_bias=True,
        router_score_fn="softmax",
    )
    moe.eval()

    # Expert 0 always wins; experts 1,2,3 get no tokens.
    force_router_bias(moe, [10.0, 0.0, -1.0, -2.0])

    x = make_moe_input(B=B, T=T, D=D)

    out, aux = moe(x, return_aux=True)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert aux["expert_counts"][0] == B * T
    assert aux["expert_counts"][1:].sum() == 0


def test_shared_experts_add_to_routed_output():
    B, T, D = 2, 8, 64

    base_kwargs = dict(
        d_model=D,
        num_experts=4,
        top_k=1,
        dropout=0.0,
        use_bias=True,
        router_score_fn="softmax",
        normalize_topk_weights=True,
        routed_scale=1.0,
        shared_scale=1.0,
    )

    moe = make_moe(**base_kwargs, shared_experts=2)
    moe.eval()

    force_router_bias(moe, [10.0, 0.0, -1.0, -2.0])

    x = make_moe_input(B=B, T=T, D=D)

    out = moe(x)

    routed = moe.experts[0](x)
    shared = sum(shared_expert(x) for shared_expert in moe.shared_experts)

    expected = routed + shared

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


# ============================================================
# F. Aux stats tests
# ============================================================

def test_expert_counts_sum_to_total_routes():
    B, T, D = 2, 8, 64
    E, K = 4, 2

    moe = make_moe(d_model=D, num_experts=E, top_k=K)
    x = make_moe_input(B=B, T=T, D=D)

    _, aux = moe(x, return_aux=True)

    assert aux["expert_counts"].sum() == B * T * K


def test_expert_fraction_sums_to_one():
    moe = make_moe()

    x = make_moe_input()
    _, aux = moe(x, return_aux=True)

    assert torch.allclose(
        aux["expert_fraction"].sum(),
        torch.tensor(1.0, dtype=aux["expert_fraction"].dtype),
        atol=1e-6,
        rtol=1e-6,
    )


def test_router_entropy_is_finite():
    moe = make_moe()

    x = make_moe_input()
    _, aux = moe(x, return_aux=True)

    assert torch.isfinite(aux["router_entropy"])
    assert aux["router_entropy"] >= 0


def test_balance_loss_is_scalar_and_finite():
    moe = make_moe(balance_loss_weight=0.01)

    x = make_moe_input()
    _, aux = moe(x, return_aux=True)

    assert aux["balance_loss"].dim() == 0
    assert aux["raw_balance_loss"].dim() == 0
    assert torch.isfinite(aux["balance_loss"])
    assert torch.isfinite(aux["raw_balance_loss"])


def test_balance_loss_zero_weight_returns_zero_weighted_loss():
    moe = make_moe(balance_loss_weight=0.0)

    x = make_moe_input()
    _, aux = moe(x, return_aux=True)

    assert torch.allclose(
        aux["balance_loss"],
        torch.zeros_like(aux["balance_loss"]),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.isfinite(aux["raw_balance_loss"])


# ============================================================
# G. Router jitter tests
# ============================================================

def test_router_jitter_disabled_is_deterministic():
    moe = make_moe(router_jitter_noise=0.0, dropout=0.0)
    moe.train()

    x = make_moe_input(B=4, T=16, D=64)

    _, aux1 = moe(x, return_aux=True)
    _, aux2 = moe(x, return_aux=True)

    assert torch.equal(aux1["router_logits"], aux2["router_logits"])
    assert torch.equal(aux1["topk_indices"], aux2["topk_indices"])


def test_router_jitter_disabled_in_eval():
    moe = make_moe(router_jitter_noise=1.0, dropout=0.0)
    moe.eval()

    x = make_moe_input(B=4, T=16, D=64)

    _, aux1 = moe(x, return_aux=True)
    _, aux2 = moe(x, return_aux=True)

    assert torch.equal(aux1["router_logits"], aux2["router_logits"])
    assert torch.equal(aux1["topk_indices"], aux2["topk_indices"])


def test_router_jitter_active_in_train_changes_routes_or_logits():
    moe = make_moe(router_jitter_noise=1.0, dropout=0.0)
    moe.train()

    x = make_moe_input(B=8, T=32, D=64)

    _, aux1 = moe(x, return_aux=True)
    _, aux2 = moe(x, return_aux=True)

    logits_changed = not torch.equal(aux1["router_logits"], aux2["router_logits"])
    routes_changed = not torch.equal(aux1["topk_indices"], aux2["topk_indices"])

    assert logits_changed or routes_changed


# ============================================================
# H. Gradient tests
# ============================================================

def test_backward_computes_gradients_for_router_and_used_experts():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=2,
        shared_experts=2,
        balance_loss_weight=0.01,
        sequence_balance_loss_weight=0.01,
        dropout=0.0,
        use_bias=True,
    )

    x = make_moe_input(B=B, T=T, D=D)
    x.requires_grad_(True)

    out, aux = moe(x, return_aux=True)
    loss = out.sum() + aux["balance_loss"] + aux["sequence_balance_loss"]
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()

    used_experts = aux["topk_indices"].unique().tolist()

    at_least_one_expert_grad = False

    for expert_id in used_experts:
        expert = moe.experts[expert_id]

        for name, param in expert.named_parameters():
            if param.grad is not None:
                at_least_one_expert_grad = True
                assert torch.isfinite(param.grad).all(), f"Non-finite grad expert {expert_id}.{name}"

    assert at_least_one_expert_grad

    for shared_id, shared_expert in enumerate(moe.shared_experts):
        for name, param in shared_expert.named_parameters():
            assert param.grad is not None, f"Missing shared expert grad for shared_experts.{shared_id}.{name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite shared expert grad for shared_experts.{shared_id}.{name}"


def test_unused_expert_may_have_no_grad():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=1,
        shared_experts=0,
        use_bias=True,
        router_score_fn="softmax",
        dropout=0.0,
    )

    moe.train()

    # Only expert 0 used.
    force_router_bias(moe, [10.0, 0.0, -1.0, -2.0])

    x = make_moe_input(B=B, T=T, D=D)
    x.requires_grad_(True)

    out, aux = moe(x, return_aux=True)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert moe.router.weight.grad is not None

    # Expert 0 should get grad.
    assert any(
        param.grad is not None
        for param in moe.experts[0].parameters()
    )

    # Other experts may have no grad. This should not crash.
    for expert_id in [1, 2, 3]:
        for param in moe.experts[expert_id].parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()


# ============================================================
# I. Integration tests
# ============================================================

def test_moe_can_replace_mlp_interface():
    B, T, D = 2, 8, 64

    moe = make_moe(d_model=D)

    x = make_moe_input(B=B, T=T, D=D)

    out = moe(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_moe_inside_transformer_like_residual():
    B, T, D = 2, 8, 64

    norm = RMSNorm(dim=D)
    moe = make_moe(d_model=D)

    x = make_moe_input(B=B, T=T, D=D)

    out = x + moe(norm(x))

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mhc_wraps_moe():
    B, T, D, N = 2, 8, 64, 4

    mhc = ManifoldHyperConnection(
        ManifoldHyperConnectionConfig(
            d_model=D,
            n_hc=N,
            sinkhorn_iters=30,
            eps=1e-6,
            init_alpha=1e-3,
            dynamic=True,
        )
    )

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=2,
        shared_experts=2,
        dropout=0.0,
        router_type="learned",
    )

    X = make_X(B=B, T=T, n_hc=N, D=D)

    X_next = mhc(
        X,
        sublayer=lambda x_sub: moe(x_sub),
    )

    assert X_next.shape == X.shape
    assert torch.isfinite(X_next).all()



def test_topk_weights_sum_to_one_when_normalized_default_scale():
    moe = make_moe(
        normalize_topk_weights=True,
        topk_weight_scale=1.0,
    )

    x = make_moe_input()
    scores = moe._router_scores(moe._compute_router_logits(x))
    _, _, topk_weights = moe._topk_routing(scores)

    assert torch.allclose(
        topk_weights.sum(dim=-1),
        torch.ones_like(topk_weights.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )

def test_routed_and_shared_scales_are_applied():
    B, T, D = 2, 8, 64

    routed_scale = 0.5
    shared_scale = 2.0

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=1,
        shared_experts=2,
        dropout=0.0,
        use_bias=True,
        router_score_fn="softmax",
        normalize_topk_weights=True,
        routed_scale=routed_scale,
        shared_scale=shared_scale,
    )
    moe.eval()

    force_router_bias(moe, [10.0, 0.0, -1.0, -2.0])

    x = make_moe_input(B=B, T=T, D=D)

    out = moe(x)

    routed = moe.experts[0](x)
    shared = sum(shared_expert(x) for shared_expert in moe.shared_experts)

    expected = routed_scale * routed + shared_scale * shared

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_sequence_expert_fraction_sums_to_one_per_sequence():
    B, T, D = 3, 8, 64
    E, K = 4, 2

    moe = make_moe(d_model=D, num_experts=E, top_k=K)
    x = make_moe_input(B=B, T=T, D=D)

    _, aux = moe(x, return_aux=True)

    seq_frac = aux["sequence_expert_fraction"]

    assert seq_frac.shape == (B, E)

    assert torch.allclose(
        seq_frac.sum(dim=-1),
        torch.ones(B, dtype=seq_frac.dtype, device=seq_frac.device),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sequence_balance_loss_is_scalar_and_finite():
    moe = make_moe(sequence_balance_loss_weight=0.01)

    x = make_moe_input()
    _, aux = moe(x, return_aux=True)

    assert aux["sequence_balance_loss"].dim() == 0
    assert aux["sequence_raw_balance_loss"].dim() == 0
    assert torch.isfinite(aux["sequence_balance_loss"])
    assert torch.isfinite(aux["sequence_raw_balance_loss"])



def test_sequence_balance_loss_zero_weight_returns_zero_weighted_loss():
    moe = make_moe(sequence_balance_loss_weight=0.0)

    x = make_moe_input()
    _, aux = moe(x, return_aux=True)

    assert torch.allclose(
        aux["sequence_balance_loss"],
        torch.zeros_like(aux["sequence_balance_loss"]),
        atol=0.0,
        rtol=0.0,
    )

    assert torch.isfinite(aux["sequence_raw_balance_loss"])


def test_hash_router_requires_input_ids():
    moe = make_moe(router_type="hash")

    x = make_moe_input()

    with pytest.raises(ValueError):
        moe(x)


def test_hash_router_forward_with_input_ids():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=2,
        router_type="hash",
        shared_experts=1,
    )

    x = make_moe_input(B=B, T=T, D=D)
    input_ids = torch.arange(B * T).view(B, T)

    out, aux = moe(
        x,
        input_ids=input_ids,
        return_aux=True,
    )

    assert out.shape == x.shape
    assert aux["topk_indices"].shape == (B, T, 2)
    assert torch.isfinite(out).all()


def test_hash_router_is_deterministic():
    B, T, D = 2, 8, 64

    moe = make_moe(
        d_model=D,
        num_experts=4,
        top_k=2,
        router_type="hash",
        shared_experts=0,
        dropout=0.0,
    )
    moe.eval()

    x = make_moe_input(B=B, T=T, D=D)
    input_ids = torch.arange(B * T).view(B, T)

    _, aux1 = moe(x, input_ids=input_ids, return_aux=True)
    _, aux2 = moe(x, input_ids=input_ids, return_aux=True)

    assert torch.equal(aux1["topk_indices"], aux2["topk_indices"])
    assert torch.equal(aux1["topk_weights"], aux2["topk_weights"])

