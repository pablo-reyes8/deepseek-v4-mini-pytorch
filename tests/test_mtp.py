# @title
# ============================================================
# MultiTokenPredictionHead / MTP tests
# ============================================================

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.deepseek_mtp import * 
from data.syntethic_long_context_retrieval import * 
# ============================================================
# Helpers
# ============================================================

def make_mtp_config(**overrides):
    cfg = dict(
        d_model=64,
        vocab_size=128,
        mtp_depth=2,

        hidden_dim=64,
        use_mtp_transform=True,
        activation="silu",

        dropout=0.0,
        use_bias=True,
        init_std=0.02,

        tie_with_lm_head=False,
        mtp_loss_weight=0.3,

        ignore_index=-100,
        pad_token_id=0,

        depth_loss_weights=None,
        validate_label_range=True,
    )
    cfg.update(overrides)
    return MTPConfig(**cfg)


def make_mtp(**overrides):
    return MultiTokenPredictionHead(make_mtp_config(**overrides))


def make_hidden_states(B=2, T=8, D=64, dtype=torch.float32, device="cpu"):
    return torch.randn(B, T, D, dtype=dtype, device=device)


def make_mtp_labels(
    B=2,
    K=2,
    T=8,
    V=128,
    ignore_index=-100,
    include_ignore=False,
):
    labels = torch.randint(
        low=1,
        high=V,
        size=(B, K, T),
        dtype=torch.long,
    )

    if include_ignore:
        labels[:, :, -2:] = ignore_index

    return labels


# ============================================================
# A. Config tests
# ============================================================

def test_valid_mtp_config_builds():
    config = MTPConfig(
        d_model=256,
        vocab_size=3000,
        mtp_depth=2,
        use_mtp_transform=True,
        hidden_dim=256,
        activation="silu",
        dropout=0.0,
        use_bias=False,
        init_std=0.02,
        mtp_loss_weight=0.3,
        ignore_index=-100,
        pad_token_id=0,
        depth_loss_weights=None,
        validate_label_range=True,
    )

    mtp = MultiTokenPredictionHead(config)

    assert mtp.d_model == 256
    assert mtp.vocab_size == 3000
    assert mtp.mtp_depth == 2
    assert mtp.ignore_index == -100
    assert len(mtp.heads) == 2
    assert len(mtp.transforms) == 2


@pytest.mark.parametrize("d_model", [0, -1, -64])
def test_invalid_mtp_d_model_raises(d_model):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(make_mtp_config(d_model=d_model))


@pytest.mark.parametrize("vocab_size", [0, -1, -128])
def test_invalid_vocab_size_raises(vocab_size):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(make_mtp_config(vocab_size=vocab_size))


@pytest.mark.parametrize("mtp_depth", [0, -1, -2])
def test_invalid_mtp_depth_raises(mtp_depth):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(make_mtp_config(mtp_depth=mtp_depth))


@pytest.mark.parametrize("hidden_dim", [0, -1, -64])
def test_invalid_hidden_dim_raises(hidden_dim):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(make_mtp_config(hidden_dim=hidden_dim))


@pytest.mark.parametrize("activation", ["bad", "tanh", "swiglu"])
def test_invalid_activation_raises(activation):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(make_mtp_config(activation=activation))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": 1.5},
        {"init_std": 0.0},
        {"init_std": -0.01},
        {"mtp_loss_weight": -0.1},
    ],
)
def test_invalid_dropout_init_loss_weight_raises(kwargs):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(make_mtp_config(**kwargs))


@pytest.mark.parametrize("pad_token_id", [-1, 128])
def test_invalid_pad_token_id_raises(pad_token_id):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(
            make_mtp_config(
                vocab_size=128,
                pad_token_id=pad_token_id,
            )
        )

@pytest.mark.parametrize(
    "depth_loss_weights",
    [
        (1.0,),              # wrong length for mtp_depth=2
        (1.0, -1.0),         # negative
        (0.0, 0.0),          # all zero
    ],
)
def test_invalid_depth_loss_weights_raise(depth_loss_weights):
    with pytest.raises(ValueError):
        MultiTokenPredictionHead(
            make_mtp_config(
                mtp_depth=2,
                depth_loss_weights=depth_loss_weights,
            )
        )


def test_valid_depth_loss_weights_builds_and_normalizes():
    mtp = make_mtp(
        mtp_depth=2,
        depth_loss_weights=(0.25, 0.75),
    )

    expected = torch.tensor([0.25, 0.75], dtype=mtp.depth_loss_weights.dtype)

    assert torch.allclose(mtp.depth_loss_weights.cpu(), expected)
    assert torch.allclose(
        mtp.depth_loss_weights.sum(),
        torch.tensor(1.0, dtype=mtp.depth_loss_weights.dtype),
    )


# ============================================================
# B. Internal structure tests
# ============================================================

def test_mtp_has_correct_number_of_heads():
    mtp = make_mtp(mtp_depth=3)

    assert len(mtp.heads) == 3
    assert len(mtp.transforms) == 3


def test_heads_have_expected_shape():
    D, V, K = 64, 128, 3

    mtp = make_mtp(d_model=D, vocab_size=V, mtp_depth=K)

    for head in mtp.heads:
        assert head.weight.shape == (V, D)


def test_transform_identity_when_disabled():
    mtp = make_mtp(use_mtp_transform=False, mtp_depth=3)

    for transform in mtp.transforms:
        assert isinstance(transform, nn.Identity)

    x = make_hidden_states()
    y = mtp.transforms[0](x)

    assert torch.equal(x, y)


def test_transform_exists_when_enabled():
    B, T, D = 2, 8, 64

    mtp = make_mtp(
        d_model=D,
        hidden_dim=128,
        use_mtp_transform=True,
        mtp_depth=2,
    )

    x = make_hidden_states(B=B, T=T, D=D)

    for transform in mtp.transforms:
        y = transform(x)
        assert y.shape == (B, T, D)
        assert torch.isfinite(y).all()

        has_params = any(p.requires_grad for p in transform.parameters())
        assert has_params


def test_build_mtp_labels_shape_and_shifts():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.long,
    )

    labels = build_mtp_labels(
        input_ids=input_ids,
        mtp_depth=2,
        ignore_index=-100,
    )

    expected = torch.tensor(
        [
            [
                [12, 13, 14, 15, -100, -100],  # x_{t+2}
                [13, 14, 15, -100, -100, -100],  # x_{t+3}
            ],
            [
                [22, 23, 24, 25, -100, -100],
                [23, 24, 25, -100, -100, -100],
            ],
        ],
        dtype=torch.long,
    )

    assert labels.shape == (2, 2, 6)
    assert torch.equal(labels, expected)


def test_build_mtp_labels_converts_future_pad_to_ignore_index():
    input_ids = torch.tensor(
        [[5, 6, 0, 7, 0, 8]],
        dtype=torch.long,
    )

    labels = build_mtp_labels(
        input_ids=input_ids,
        mtp_depth=1,
        ignore_index=-100,
        pad_token_id=0,
    )

    # depth 0 predicts x_{t+2}:
    # future targets: [0, 7, 0, 8, ignore, ignore]
    # pad targets become ignore_index.
    expected = torch.tensor(
        [[[-100, 7, -100, 8, -100, -100]]],
        dtype=torch.long,
    )

    assert torch.equal(labels, expected)


def test_build_mtp_labels_rejects_bad_input_rank():
    input_ids = torch.ones(2, 3, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        build_mtp_labels(input_ids, mtp_depth=2)


def test_build_mtp_labels_rejects_float_input_ids():
    input_ids = torch.randn(2, 8)

    with pytest.raises(TypeError):
        build_mtp_labels(input_ids, mtp_depth=2)

def test_rejects_out_of_range_mtp_labels():
    mtp = make_mtp(
        vocab_size=128,
        mtp_depth=2,
        ignore_index=-100,
        validate_label_range=True,
    )

    hidden_states = make_hidden_states(B=2, T=8, D=64)
    mtp_labels = make_mtp_labels(B=2, K=2, T=8, V=128)

    mtp_labels[0, 0, 0] = 999  # invalid

    with pytest.raises(ValueError):
        mtp(hidden_states, mtp_labels=mtp_labels)


def test_allows_ignore_index_in_mtp_labels():
    mtp = make_mtp(
        vocab_size=128,
        mtp_depth=2,
        ignore_index=-100,
        validate_label_range=True,
    )

    hidden_states = make_hidden_states(B=2, T=8, D=64)
    mtp_labels = make_mtp_labels(
        B=2,
        K=2,
        T=8,
        V=128,
        ignore_index=-100,
        include_ignore=True,
    )

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    assert outputs["mtp_loss"] is not None
    assert torch.isfinite(outputs["mtp_loss"])



# ============================================================
# C. Forward tests
# ============================================================

def test_mtp_logits_shape():
    B, T, D, V, K = 2, 8, 64, 128, 3

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
    )

    hidden_states = make_hidden_states(B=B, T=T, D=D)

    outputs = mtp(hidden_states)

    assert outputs["mtp_logits"].shape == (B, K, T, V)


def test_forward_without_labels_returns_no_loss():
    mtp = make_mtp()

    hidden_states = make_hidden_states()

    outputs = mtp(hidden_states, mtp_labels=None)

    assert outputs["mtp_loss"] is None


def test_forward_with_labels_returns_scalar_loss():
    B, T, D, V, K = 2, 8, 64, 128, 2

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        pad_token_id=0,
    )

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    mtp_labels = make_mtp_labels(B=B, K=K, T=T, V=V)

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    assert outputs["mtp_loss"] is not None
    assert outputs["mtp_loss"].dim() == 0
    assert torch.isfinite(outputs["mtp_loss"])


@pytest.mark.parametrize(
    "bad_hidden_states",
    [
        torch.randn(8, 64),
        torch.randn(2, 8, 64, 1),
    ],
)
def test_rejects_wrong_hidden_states_rank(bad_hidden_states):
    mtp = make_mtp()

    with pytest.raises(ValueError):
        mtp(bad_hidden_states)


def test_rejects_wrong_hidden_size():
    mtp = make_mtp(d_model=64)

    hidden_states = torch.randn(2, 8, 32)

    with pytest.raises(ValueError):
        mtp(hidden_states)


@pytest.mark.parametrize(
    "bad_labels",
    [
        torch.ones(3, 2, 8, dtype=torch.long).permute(1, 0, 2),  # [K,B,T]
        torch.ones(2, 8, dtype=torch.long),                     # [B,T]
        torch.ones(2, 2, 8, 1, dtype=torch.long),                # [B,K,T,1]
        torch.ones(2, 2, 9, dtype=torch.long),                   # [B,K,T+1]
    ],
)
def test_rejects_wrong_mtp_labels_shape(bad_labels):
    mtp = make_mtp(
        d_model=64,
        vocab_size=128,
        mtp_depth=2,
    )

    hidden_states = make_hidden_states(B=2, T=8, D=64)

    with pytest.raises(ValueError):
        mtp(hidden_states, mtp_labels=bad_labels)


def test_rejects_float_mtp_labels():
    mtp = make_mtp()

    hidden_states = make_hidden_states()
    mtp_labels = torch.randn(2, 2, 8)

    with pytest.raises(TypeError):
        mtp(hidden_states, mtp_labels=mtp_labels)


# ============================================================
# D. Loss tests
# ============================================================

def test_mtp_loss_matches_manual_cross_entropy():
    B, T, D, V, K = 2, 8, 64, 128, 3
    ignore_index = -100
    loss_weight = 0.3

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        dropout=0.0,
        mtp_loss_weight=loss_weight,
        ignore_index=ignore_index,
        pad_token_id=0,
    )
    mtp.eval()

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    mtp_labels = make_mtp_labels(
        B=B,
        K=K,
        T=T,
        V=V,
        ignore_index=ignore_index,
        include_ignore=True,
    )

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    mtp_logits = outputs["mtp_logits"]

    manual_losses = []

    for k in range(K):
        loss_k = F.cross_entropy(
            mtp_logits[:, k, :, :].reshape(B * T, V),
            mtp_labels[:, k, :].reshape(B * T),
            ignore_index=ignore_index,
        )
        manual_losses.append(loss_k)

    manual_per_depth = torch.stack(manual_losses)
    manual_raw = manual_per_depth.mean()
    manual_weighted = loss_weight * manual_raw

    assert torch.allclose(
        outputs["aux"]["raw_mtp_loss"],
        manual_raw,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        outputs["mtp_loss"],
        manual_weighted,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        outputs["aux"]["mtp_loss_per_depth"],
        manual_per_depth,
        atol=1e-6,
        rtol=1e-6,
    )

    assert "depth_loss_weights" in outputs["aux"]




def test_mtp_loss_per_depth_shape():
    B, T, D, V, K = 2, 8, 64, 128, 4

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
    )

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    mtp_labels = make_mtp_labels(B=B, K=K, T=T, V=V)

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    assert outputs["aux"]["mtp_loss_per_depth"].shape == (K,)


def test_mtp_loss_ignores_ignore_index():
    B, T, D, V, K = 2, 8, 64, 128, 2
    ignore_index = -100

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        ignore_index=ignore_index,
        pad_token_id=0,
    )
    mtp.eval()

    hidden_states = make_hidden_states(B=B, T=T, D=D)

    mtp_labels = make_mtp_labels(
        B=B,
        K=K,
        T=T,
        V=V,
        ignore_index=ignore_index,
        include_ignore=True,
    )

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    logits = outputs["mtp_logits"]

    manual_losses = []

    for k in range(K):
        manual_losses.append(
            F.cross_entropy(
                logits[:, k, :, :].reshape(B * T, V),
                mtp_labels[:, k, :].reshape(B * T),
                ignore_index=ignore_index,
            )
        )

    manual_raw = torch.stack(manual_losses).mean()

    assert torch.allclose(
        outputs["aux"]["raw_mtp_loss"],
        manual_raw,
        atol=1e-6,
        rtol=1e-6,
    )

def test_mtp_loss_uses_depth_loss_weights():
    B, T, D, V, K = 2, 8, 64, 128, 2
    ignore_index = -100
    depth_weights = (0.25, 0.75)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        mtp_loss_weight=1.0,
        ignore_index=ignore_index,
        depth_loss_weights=depth_weights,
    )
    mtp.eval()

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    mtp_labels = make_mtp_labels(
        B=B,
        K=K,
        T=T,
        V=V,
        ignore_index=ignore_index,
        include_ignore=True,
    )

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    logits = outputs["mtp_logits"]

    manual_losses = []

    for k in range(K):
        manual_losses.append(
            F.cross_entropy(
                logits[:, k, :, :].reshape(B * T, V),
                mtp_labels[:, k, :].reshape(B * T),
                ignore_index=ignore_index,
            )
        )

    manual_per_depth = torch.stack(manual_losses)
    weights = torch.tensor(depth_weights, dtype=manual_per_depth.dtype)
    weights = weights / weights.sum()

    manual_raw = (weights * manual_per_depth).sum()

    assert torch.allclose(
        outputs["aux"]["raw_mtp_loss"],
        manual_raw,
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_mtp_loss_weight_returns_zero_weighted_loss():
    B, T, D, V, K = 2, 8, 64, 128, 2

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        mtp_loss_weight=0.0,
    )

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    mtp_labels = make_mtp_labels(B=B, K=K, T=T, V=V)

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    assert torch.allclose(
        outputs["mtp_loss"],
        torch.zeros_like(outputs["mtp_loss"]),
        atol=0.0,
        rtol=0.0,
    )

    assert torch.isfinite(outputs["aux"]["raw_mtp_loss"])


# ============================================================
# E. Weight tying tests
# ============================================================

def test_tie_weights_shares_parameter_across_heads():
    D, V, K = 64, 128, 3

    lm_head = nn.Linear(D, V, bias=False)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        use_mtp_transform=False,
    )

    mtp.tie_weights(lm_head.weight)

    for head in mtp.heads:
        assert head.weight is lm_head.weight


def test_tied_heads_change_when_lm_head_changes():
    D, V, K = 64, 128, 3

    lm_head = nn.Linear(D, V, bias=False)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        use_mtp_transform=False,
    )

    mtp.tie_weights(lm_head.weight)

    old_value = lm_head.weight[0, 0].detach().clone()

    with torch.no_grad():
        lm_head.weight[0, 0] += 1.0

    expected = old_value + 1.0

    for head in mtp.heads:
        assert head.weight is lm_head.weight
        assert torch.allclose(
            head.weight[0, 0],
            expected,
            atol=1e-6,
            rtol=1e-6,
        )


def test_tie_weights_rejects_wrong_shape():
    D, V = 64, 128

    bad_lm_head = nn.Linear(D + 1, V, bias=False)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=2,
    )

    with pytest.raises(ValueError):
        mtp.tie_weights(bad_lm_head.weight)


# ============================================================
# F. Dropout / determinism tests
# ============================================================

def test_mtp_dropout_zero_is_deterministic():
    mtp = make_mtp(dropout=0.0)
    mtp.train()

    hidden_states = make_hidden_states()

    out1 = mtp(hidden_states)["mtp_logits"]
    out2 = mtp(hidden_states)["mtp_logits"]

    assert torch.equal(out1, out2)


def test_mtp_dropout_disabled_in_eval_mode():
    mtp = make_mtp(dropout=0.5)
    mtp.eval()

    hidden_states = make_hidden_states()

    out1 = mtp(hidden_states)["mtp_logits"]
    out2 = mtp(hidden_states)["mtp_logits"]

    assert torch.equal(out1, out2)


def test_mtp_dropout_active_in_train_mode():
    mtp = make_mtp(dropout=0.5, use_mtp_transform=True)
    mtp.train()

    hidden_states = make_hidden_states(B=4, T=16, D=64)

    out1 = mtp(hidden_states)["mtp_logits"]
    out2 = mtp(hidden_states)["mtp_logits"]

    assert not torch.equal(out1, out2)


# ============================================================
# G. Gradient tests
# ============================================================

def test_mtp_backward_computes_gradients():
    B, T, D, V, K = 2, 8, 64, 128, 2

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        use_mtp_transform=True,
        dropout=0.0,
    )

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    hidden_states.requires_grad_(True)

    mtp_labels = make_mtp_labels(B=B, K=K, T=T, V=V)

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    loss = outputs["mtp_loss"]
    loss.backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()

    for name, param in mtp.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_backward_with_tied_lm_head():
    B, T, D, V, K = 2, 8, 64, 128, 2

    lm_head = nn.Linear(D, V, bias=False)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        use_mtp_transform=False,
    )

    mtp.tie_weights(lm_head.weight)

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    hidden_states.requires_grad_(True)

    mtp_labels = make_mtp_labels(B=B, K=K, T=T, V=V)

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    outputs["mtp_loss"].backward()

    assert lm_head.weight.grad is not None
    assert torch.isfinite(lm_head.weight.grad).all()


# ============================================================
# H. Integration tests
# ============================================================

def test_mtp_integrates_with_minicausallm_hidden_states():
    B, T, D, V, K = 2, 8, 64, 128, 2

    hidden_states = make_hidden_states(B=B, T=T, D=D)
    mtp_labels = make_mtp_labels(B=B, K=K, T=T, V=V)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        pad_token_id=0,
    )

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    assert outputs["mtp_logits"].shape == (B, K, T, V)
    assert outputs["mtp_loss"] is not None
    assert torch.isfinite(outputs["mtp_loss"])


def test_mtp_labels_from_synthetic_dataset_format():
    """
    This test assumes your synthetic retrieval dataloader supports use_mtp=True
    and returns either:

        input_ids, labels, mtp_labels

    or a dict with key "mtp_labels".

    It is skipped if the synthetic MTP dataset functions are not in memory.
    """

    required_names = [
        "SyntheticRetrievalConfig",
        "create_synthetic_retrieval_dataloaders"]

    for name in required_names:
        if name not in globals():
            pytest.skip(f"{name} is not defined in this notebook/session.")

    data_cfg = SyntheticRetrievalConfig(
        num_train_examples=64,
        num_val_examples=16,
        block_size=32,
        min_filler_tokens=8,
        max_filler_tokens=16,
        num_keys_per_example=4,
        vocab_filler_size=100,
        num_key_types=32,
        num_value_types=64,
        batch_size=4,
        num_workers=0,
        seed=123)

    train_loader, _, tokenizer = create_synthetic_retrieval_dataloaders(
        cfg=data_cfg,
        use_mtp=True)

    batch = next(iter(train_loader))

    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        mtp_labels = batch["mtp_labels"]
    else:
        assert len(batch) == 3, (
            "Expected synthetic MTP batch to be either dict or "
            "(input_ids, labels, mtp_labels).")
        input_ids, _, mtp_labels = batch

    # Official MTP format must be [B,K,T].
    if mtp_labels.dim() == 3 and mtp_labels.shape[0] != input_ids.shape[0]:
        # Convert [K,B,T] -> [B,K,T] if dataset returns depth first.
        mtp_labels = mtp_labels.permute(1, 0, 2).contiguous()

    assert mtp_labels.dim() == 3
    assert mtp_labels.shape[0] == input_ids.shape[0]
    assert mtp_labels.shape[-1] == input_ids.shape[-1]

    B, K, T = mtp_labels.shape
    D = 64
    V = tokenizer.vocab_size

    hidden_states = torch.randn(B, T, D)

    mtp = make_mtp(
        d_model=D,
        vocab_size=V,
        mtp_depth=K,
        pad_token_id=tokenizer.pad_id)

    outputs = mtp(
        hidden_states,
        mtp_labels=mtp_labels,
        return_aux=True)

    assert outputs["mtp_logits"].shape == (B, K, T, V)
    assert outputs["mtp_loss"] is not None
    assert torch.isfinite(outputs["mtp_loss"])

