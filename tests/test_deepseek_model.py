
# ============================================================
# DeepSeekV4LM tests
# Tiny configs only: CPU/Colab-safe
# ============================================================

import math
import pytest
import torch
import torch.nn.functional as F

from src.mini_deepseek_class import * 
from data.syntethic_long_context_retrieval import *  

# ============================================================
# Helpers
# ============================================================

def make_dsv4_config(**overrides):
    cfg = dict(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        max_seq_len=32,
        pad_token_id=0,
        ignore_index=-100,

        embedding_dropout=0.0,
        scale_embeddings=False,
        tie_word_embeddings=True,

        rms_norm_eps=1e-6,

        attention_type="mha",
        n_heads=4,
        head_dim=8,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_bias=False,
        use_rope=True,
        rope_theta=10000.0,
        rotary_dim=8,

        compression_factor=4,
        hca_compression_factor=4,
        window_size=4,

        top_k_blocks=2,
        indexer_dim=8,
        n_indexer_heads=2,
        query_compression_dim=8,

        use_attention_sink=True,
        use_grouped_output_projection=False,
        output_projection_groups=None,
        use_indexer_score_bias=True,   # useful for indexer gradients
        use_separate_local_kv=True,

        ffn_type="dense",

        mlp_hidden_dim=64,
        mlp_expansion_factor=2.0,
        mlp_multiple_of=1,
        mlp_dropout=0.0,
        use_mlp_bias=False,

        num_experts=4,
        top_k_experts=2,
        expert_hidden_dim=64,
        expert_expansion_factor=2.0,
        expert_multiple_of=1,
        shared_experts=1,
        shared_hidden_dim=64,
        shared_expansion_factor=2.0,

        router_type="learned",
        router_score_fn="sqrt_softplus",
        normalize_topk_weights=True,
        topk_weight_scale=1.0,
        router_jitter_noise=0.0,
        hash_routing_stride=1,

        routed_scale=1.0,
        shared_scale=1.0,

        balance_loss_weight=0.0,
        sequence_balance_loss_weight=0.0,

        use_mhc=False,
        n_hc=2,
        mhc_sinkhorn_iters=5,
        mhc_eps=1e-6,
        mhc_dynamic=True,
        mhc_expand_mode="first",
        mhc_collapse_mode="readout",
        mhc_use_log_sinkhorn=False,
        mhc_sinkhorn_fp32=True,
        mhc_init_alpha=1e-3,
        mhc_alpha_max=1.0,
        mhc_bounded_alpha=True,

        use_mtp=False,
        mtp_depth=2,
        mtp_hidden_dim=32,
        use_mtp_transform=True,
        mtp_activation="silu",
        mtp_dropout=0.0,
        mtp_loss_weight=0.3,
        mtp_tie_with_lm_head=False,
        mtp_depth_loss_weights=None,
        mtp_validate_label_range=True,

        init_std=0.02,
    )
    cfg.update(overrides)
    return DeepSeekV4LMConfig(**cfg)


def make_dsv4_model(**overrides):
    return DeepSeekV4LM(make_dsv4_config(**overrides))


def make_ids(B=2, T=12, V=128, pad_token_id=0, include_pad=False):
    input_ids = torch.randint(
        low=1,
        high=V,
        size=(B, T),
        dtype=torch.long,
    )

    if include_pad:
        input_ids[0, -2:] = pad_token_id

    return input_ids


def make_next_token_labels(input_ids, ignore_index=-100):
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    return labels


def assert_finite_grads(model):
    any_grad = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            continue

        any_grad = True
        assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    assert any_grad, "No gradients were produced."


def get_embedding_weight_for_test(model):
    if hasattr(model.embedding, "weight"):
        return model.embedding.weight
    if hasattr(model.embedding, "embedding") and hasattr(model.embedding.embedding, "weight"):
        return model.embedding.embedding.weight
    if hasattr(model.embedding, "token_embedding") and hasattr(model.embedding.token_embedding, "weight"):
        return model.embedding.token_embedding.weight
    raise AssertionError("Could not find embedding weight.")


# ============================================================
# A. Config and construction
# ============================================================

def test_valid_deepseekv4lm_config_builds():
    cfg = make_dsv4_config(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=4,
        head_dim=8,
    )

    model = DeepSeekV4LM(cfg)

    assert model.vocab_size == 128
    assert model.d_model == 32
    assert model.n_layers == 1
    assert len(model.blocks) == 1


@pytest.mark.parametrize("attention_type", ["bad", "mla", "flash"])
def test_invalid_attention_type_raises(attention_type):
    with pytest.raises(ValueError):
        DeepSeekV4LM(make_dsv4_config(attention_type=attention_type))


@pytest.mark.parametrize("ffn_type", ["bad", "mlp", "expert"])
def test_invalid_ffn_type_raises(ffn_type):
    with pytest.raises(ValueError):
        DeepSeekV4LM(make_dsv4_config(ffn_type=ffn_type))


# ============================================================
# B. Forward basic
# ============================================================

@pytest.mark.parametrize("attention_type", ["mha", "hca", "csa"])
def test_logits_shape_all_attention_types(attention_type):
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type=attention_type,
        ffn_type="dense",
        vocab_size=V,
        max_seq_len=32,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V)

    outputs = model(input_ids=input_ids)

    assert outputs["logits"].shape == (B, T, V)
    assert torch.isfinite(outputs["logits"]).all()


@pytest.mark.parametrize("ffn_type", ["dense", "moe"])
def test_forward_dense_and_moe_ffn(ffn_type):
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="csa",
        ffn_type=ffn_type,
        vocab_size=V,
        max_seq_len=32,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V)

    outputs = model(input_ids=input_ids)

    assert outputs["logits"].shape == (B, T, V)
    assert torch.isfinite(outputs["logits"]).all()


@pytest.mark.parametrize("use_mhc", [False, True])
def test_forward_with_and_without_mhc(use_mhc):
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="hca",
        ffn_type="dense",
        use_mhc=use_mhc,
        vocab_size=V,
        max_seq_len=32,
        n_hc=2,
        mhc_sinkhorn_iters=5,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V)

    outputs = model(input_ids=input_ids)

    assert outputs["logits"].shape == (B, T, V)
    assert torch.isfinite(outputs["logits"]).all()


@pytest.mark.parametrize("use_mtp", [False, True])
def test_forward_with_and_without_mtp(use_mtp):
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        use_mtp=use_mtp,
        vocab_size=V,
        max_seq_len=32,
        mtp_depth=2,
    )

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        return_aux=True,
    )

    assert outputs["logits"].shape == (B, T, V)
    assert outputs["loss"] is not None
    assert outputs["lm_loss"] is not None
    assert torch.isfinite(outputs["loss"])

    if use_mtp:
        assert outputs["mtp_loss"] is not None
        assert torch.isfinite(outputs["mtp_loss"])
    else:
        assert outputs["mtp_loss"] is None


# ============================================================
# C. Loss
# ============================================================

def test_lm_loss_matches_manual_ce():
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        use_mtp=False,
        vocab_size=V,
        max_seq_len=32,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    logits = outputs["logits"]

    manual = F.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=model.ignore_index,
    )

    assert torch.allclose(outputs["lm_loss"], manual, atol=1e-6, rtol=1e-6)


def test_total_loss_includes_mtp_loss():
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        use_mtp=True,
        mtp_depth=2,
        mtp_loss_weight=0.3,
        vocab_size=V,
        max_seq_len=32,
    )

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        mtp_labels=None,
    )

    assert outputs["lm_loss"] is not None
    assert outputs["mtp_loss"] is not None
    assert outputs["moe_aux_loss"] is None

    expected = outputs["lm_loss"] + outputs["mtp_loss"]

    assert torch.allclose(outputs["loss"], expected, atol=1e-6, rtol=1e-6)


def test_total_loss_includes_moe_aux_loss():
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="csa",
        ffn_type="moe",
        use_mtp=False,
        vocab_size=V,
        max_seq_len=32,
        balance_loss_weight=0.01,
        sequence_balance_loss_weight=0.01,
    )

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        return_aux=True,
    )

    assert outputs["lm_loss"] is not None
    assert outputs["moe_aux_loss"] is not None
    assert outputs["moe_aux_loss"] >= 0

    expected = outputs["lm_loss"] + outputs["moe_aux_loss"]

    assert torch.allclose(outputs["loss"], expected, atol=1e-6, rtol=1e-6)


# ============================================================
# D. Weight tying
# ============================================================

def test_lm_head_tied_to_embedding():
    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        tie_word_embeddings=True,
    )

    embedding_weight = get_embedding_weight_for_test(model)

    assert model.lm_head.weight is embedding_weight


def test_mtp_heads_tied_to_lm_head_when_enabled():
    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        use_mtp=True,
        mtp_depth=2,
        tie_word_embeddings=True,
        mtp_tie_with_lm_head=True,
    )

    assert model.mtp_head is not None

    for head in model.mtp_head.heads:
        assert head.weight is model.lm_head.weight


# ============================================================
# E. Aux
# ============================================================

def test_return_aux_contains_blocks():
    model = make_dsv4_model(
        attention_type="csa",
        ffn_type="dense",
        n_layers=2,
    )
    model.eval()

    input_ids = make_ids(B=2, T=12, V=model.vocab_size)

    outputs = model(
        input_ids=input_ids,
        return_aux=True,
        need_weights=True,
    )

    assert "blocks" in outputs["aux"]
    assert len(outputs["aux"]["blocks"]) == 2


@pytest.mark.parametrize("attention_type", ["mha", "hca", "csa"])
def test_need_weights_attention_aux_shapes(attention_type):
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type=attention_type,
        ffn_type="dense",
        vocab_size=V,
        max_seq_len=32,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V)

    outputs = model(
        input_ids=input_ids,
        return_aux=True,
        need_weights=True,
    )

    block_aux = outputs["aux"]["blocks"][0]

    if attention_type == "mha":
        if "attention" in block_aux:
            attn_aux = block_aux["attention"]

            if torch.is_tensor(attn_aux):
                assert attn_aux.shape == (B, model.config.n_heads, T, T)
                assert torch.isfinite(attn_aux).all()

            elif isinstance(attn_aux, dict):
                assert "attn_weights" in attn_aux
                assert attn_aux["attn_weights"].shape == (B, model.config.n_heads, T, T)
                assert torch.isfinite(attn_aux["attn_weights"]).all()

            else:
                raise TypeError(
                    "MHA attention aux must be either a tensor or a dict, "
                    f"got {type(attn_aux)}"
                )

        return

    assert "attention" in block_aux
    attn_aux = block_aux["attention"]

    assert isinstance(attn_aux, dict)
    assert "global_attn_weights" in attn_aux
    assert "local_attn_weights" in attn_aux

    assert attn_aux["local_attn_weights"].shape == (B, model.config.n_heads, T, T)

    if attention_type == "hca":
        S = math.ceil(T / model.config.hca_compression_factor)
        assert attn_aux["global_attn_weights"].shape == (B, model.config.n_heads, T, S)

    if attention_type == "csa":
        S = math.ceil(T / model.config.compression_factor)
        K = min(model.config.top_k_blocks, S)

        assert attn_aux["global_attn_weights"].shape == (B, model.config.n_heads, T, K)
        assert attn_aux["topk_indices"].shape == (B, T, K)


def test_moe_aux_present_when_ffn_type_moe():
    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="moe",
        n_layers=2,
    )
    model.eval()

    input_ids = make_ids(B=2, T=12, V=model.vocab_size)

    outputs = model(
        input_ids=input_ids,
        return_aux=True,
    )

    assert len(outputs["aux"]["blocks"]) == 2

    for block_aux in outputs["aux"]["blocks"]:
        assert "moe" in block_aux
        assert isinstance(block_aux["moe"], dict)


def test_mhc_aux_present_when_use_mhc():
    model = make_dsv4_model(
        attention_type="hca",
        ffn_type="dense",
        use_mhc=True,
        n_hc=2,
        mhc_sinkhorn_iters=5,
    )
    model.eval()

    input_ids = make_ids(B=2, T=12, V=model.vocab_size)

    outputs = model(
        input_ids=input_ids,
        return_aux=True,
        need_weights=True,
    )

    block_aux = outputs["aux"]["blocks"][0]

    assert "mhc_attn" in block_aux
    assert "mhc_ffn" in block_aux

    for key in ["A", "B", "C"]:
        assert key in block_aux["mhc_attn"]
        assert key in block_aux["mhc_ffn"]


# ============================================================
# F. Causality
# ============================================================

@pytest.mark.parametrize("attention_type", ["mha", "hca", "csa"])
def test_changing_future_tokens_does_not_change_past_logits(attention_type):
    B, T, V = 2, 16, 128
    cut = 8

    model = make_dsv4_model(
        attention_type=attention_type,
        ffn_type="dense",
        vocab_size=V,
        max_seq_len=32,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
        use_mtp=False,
    )
    model.eval()

    input_ids_1 = make_ids(B=B, T=T, V=V)
    input_ids_2 = input_ids_1.clone()

    future = torch.randint(1, V, input_ids_2[:, cut:].shape)
    input_ids_2[:, cut:] = future

    with torch.no_grad():
        logits_1 = model(input_ids=input_ids_1)["logits"]
        logits_2 = model(input_ids=input_ids_2)["logits"]

    assert torch.allclose(
        logits_1[:, :cut, :],
        logits_2[:, :cut, :],
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# G. Attention mask
# ============================================================

@pytest.mark.parametrize("attention_type", ["hca", "csa"])
def test_auto_attention_mask_from_pad_token(attention_type):
    B, T, V = 2, 12, 128
    pad = 0

    model = make_dsv4_model(
        attention_type=attention_type,
        ffn_type="dense",
        vocab_size=V,
        pad_token_id=pad,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V, pad_token_id=pad, include_pad=True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=None,
        return_aux=True,
        need_weights=True,
    )

    assert "attention_mask" in outputs["aux"]
    assert torch.equal(outputs["aux"]["attention_mask"], (input_ids != pad).long())

    attn_aux = outputs["aux"]["blocks"][0]["attention"]
    local_weights = attn_aux["local_attn_weights"]

    # pad keys at positions -2, -1 for batch 0 must receive zero local attention.
    assert torch.allclose(
        local_weights[0, :, :, -2:],
        torch.zeros_like(local_weights[0, :, :, -2:]),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("attention_type", ["hca", "csa"])
def test_explicit_attention_mask_overrides_auto_mask(attention_type):
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type=attention_type,
        ffn_type="dense",
        vocab_size=V,
        pad_token_id=0,
    )
    model.eval()

    input_ids = make_ids(B=B, T=T, V=V)

    attention_mask = torch.ones(B, T, dtype=torch.long)
    blocked_key = 5
    attention_mask[:, blocked_key] = 0

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_aux=True,
        need_weights=True,
    )

    assert torch.equal(outputs["aux"]["attention_mask"], attention_mask)

    local_weights = outputs["aux"]["blocks"][0]["attention"]["local_attn_weights"]

    assert torch.allclose(
        local_weights[:, :, :, blocked_key],
        torch.zeros_like(local_weights[:, :, :, blocked_key]),
        atol=0.0,
        rtol=0.0,
    )


# ============================================================
# H. MTP labels
# ============================================================

def test_auto_build_mtp_labels_when_missing():
    B, T, V = 2, 12, 128

    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        use_mtp=True,
        mtp_depth=2,
        vocab_size=V,
    )

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        mtp_labels=None,
        return_aux=True,
    )

    assert outputs["mtp_loss"] is not None
    assert torch.isfinite(outputs["mtp_loss"])
    assert "mtp" in outputs["aux"]


def test_explicit_mtp_labels_used():
    B, T, V, K = 2, 12, 128, 2

    model = make_dsv4_model(
        attention_type="mha",
        ffn_type="dense",
        use_mtp=True,
        mtp_depth=K,
        vocab_size=V,
    )

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    mtp_labels = build_mtp_labels(
        input_ids=input_ids,
        mtp_depth=K,
        ignore_index=model.ignore_index,
        pad_token_id=model.pad_token_id,
    )

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        mtp_labels=mtp_labels,
        return_aux=True,
    )

    assert outputs["mtp_loss"] is not None
    assert torch.isfinite(outputs["mtp_loss"])


# ============================================================
# I. Backward
# ============================================================

@pytest.mark.parametrize(
    "overrides",
    [
        dict(attention_type="mha", ffn_type="dense", use_mhc=False, use_mtp=False),
        dict(attention_type="hca", ffn_type="dense", use_mhc=False, use_mtp=False),
        dict(attention_type="csa", ffn_type="dense", use_mhc=False, use_mtp=False),
        dict(attention_type="csa", ffn_type="moe", use_mhc=False, use_mtp=False, balance_loss_weight=0.01),
        dict(attention_type="csa", ffn_type="moe", use_mhc=False, use_mtp=True, balance_loss_weight=0.01),
        dict(attention_type="csa", ffn_type="moe", use_mhc=True, use_mtp=True, balance_loss_weight=0.01, n_hc=2),
    ],
)
def test_backward_all_major_configs(overrides):
    B, T, V = 2, 8, 128

    cfg = make_dsv4_config(
        vocab_size=V,
        max_seq_len=16,
        n_layers=1,
        d_model=32,
        n_heads=4,
        head_dim=8,
        rotary_dim=8,
        mlp_hidden_dim=64,
        expert_hidden_dim=64,
        shared_hidden_dim=64,
        num_experts=4,
        top_k_experts=2,
        compression_factor=4,
        top_k_blocks=2,
        window_size=4,
        indexer_dim=8,
        query_compression_dim=8,
        mhc_sinkhorn_iters=5,
        **overrides,
    )

    model = DeepSeekV4LM(cfg)

    input_ids = make_ids(B=B, T=T, V=V)
    labels = make_next_token_labels(input_ids)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        return_aux=True,
    )

    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])

    outputs["loss"].backward()

    assert_finite_grads(model)


# ============================================================
# J. Synthetic dataset
# ============================================================

def test_synthetic_dataset_forward_loss():
    required = ["SyntheticRetrievalConfig", "create_synthetic_retrieval_dataloaders"]

    for name in required:
        if name not in globals():
            pytest.skip(f"{name} is not defined.")

    data_cfg = SyntheticRetrievalConfig(
        num_train_examples=32,
        num_val_examples=8,
        block_size=32,
        min_filler_tokens=8,
        max_filler_tokens=16,
        num_keys_per_example=4,
        vocab_filler_size=100,
        num_key_types=32,
        num_value_types=64,
        batch_size=4,
        num_workers=0,
        seed=123,
    )

    train_loader, _, tokenizer = create_synthetic_retrieval_dataloaders(
        cfg=data_cfg,
        use_mtp=False,
    )

    batch = next(iter(train_loader))

    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
    else:
        input_ids, labels = batch[:2]

    model = make_dsv4_model(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=data_cfg.block_size,
        attention_type="csa",
        ffn_type="dense",
        use_mtp=False,
    )

    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    assert outputs["logits"].shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        tokenizer.vocab_size,
    )
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])


def test_synthetic_dataset_backward():
    required = ["SyntheticRetrievalConfig", "create_synthetic_retrieval_dataloaders"]

    for name in required:
        if name not in globals():
            pytest.skip(f"{name} is not defined.")

    data_cfg = SyntheticRetrievalConfig(
        num_train_examples=32,
        num_val_examples=8,
        block_size=32,
        min_filler_tokens=8,
        max_filler_tokens=16,
        num_keys_per_example=4,
        vocab_filler_size=100,
        num_key_types=32,
        num_value_types=64,
        batch_size=4,
        num_workers=0,
        seed=123,
    )

    train_loader, _, tokenizer = create_synthetic_retrieval_dataloaders(
        cfg=data_cfg,
        use_mtp=False,
    )

    batch = next(iter(train_loader))

    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
    else:
        input_ids, labels = batch[:2]

    model = make_dsv4_model(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=data_cfg.block_size,
        attention_type="csa",
        ffn_type="moe",
        balance_loss_weight=0.01,
        use_mtp=False,
    )

    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])

    outputs["loss"].backward()

    assert_finite_grads(model)