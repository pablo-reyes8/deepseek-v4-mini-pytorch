import torch
from torch.utils.data import DataLoader, TensorDataset

from data.data_utils import normalize_lm_batch
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig
from training.adam_optmizer import build_adamw_optimizer, build_adamw_parameter_groups
from training.autocast import setup_device_and_precision
from training.scheduler import WarmupCosineLR
from training.train_one_epoch import train_one_epoch


def make_tiny_lm(**overrides):
    cfg = dict(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        max_seq_len=16,
        pad_token_id=0,
        attention_type="mha",
        n_heads=2,
        head_dim=8,
        rotary_dim=8,
        ffn_type="dense",
        mlp_hidden_dim=32,
        use_mhc=False,
        use_mtp=False,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )
    cfg.update(overrides)
    return DeepSeekV4LM(DeepSeekV4LMConfig(**cfg))


def make_tiny_loader(batch_size=2, seq_len=12, vocab_size=64, n_batches=3):
    torch.manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (batch_size * n_batches, seq_len), dtype=torch.long)
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = -100
    return DataLoader(TensorDataset(input_ids, labels), batch_size=batch_size, shuffle=False)


def test_normalize_lm_batch_accepts_tuple_and_dict():
    input_ids = torch.ones(2, 4, dtype=torch.long)
    labels = torch.zeros(2, 4, dtype=torch.long)

    tuple_batch = normalize_lm_batch((input_ids, labels))
    assert tuple_batch["input_ids"] is input_ids
    assert tuple_batch["labels"] is labels

    dict_batch = normalize_lm_batch({"input_ids": input_ids})
    assert dict_batch["labels"] is input_ids


def test_warmup_cosine_scheduler_state_roundtrip():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2, min_lr=0.1)

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.5
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0

    state = scheduler.state_dict()
    restored = WarmupCosineLR(optimizer, total_steps=99, warmup_steps=0, min_lr=0.0)
    restored.load_state_dict(state)
    assert restored.state_dict()["step_num"] == 2
    assert restored.state_dict()["warmup_steps"] == 2


def test_adamw_parameter_groups_cover_each_trainable_param_once():
    model = make_tiny_lm()
    groups, info = build_adamw_parameter_groups(model, weight_decay=0.1)

    grouped_ids = [id(param) for group in groups for param in group["params"]]
    trainable_ids = [id(param) for param in model.parameters() if param.requires_grad]

    assert sorted(grouped_ids) == sorted(trainable_ids)
    assert len(grouped_ids) == len(set(grouped_ids))
    assert info["num_decay_tensors"] > 0
    assert info["num_no_decay_tensors"] > 0


def test_train_one_epoch_cpu_tiny_model_updates_and_steps_scheduler():
    model = make_tiny_lm()
    loader = make_tiny_loader()
    optimizer, _ = build_adamw_optimizer(model, learning_rate=1e-3, weight_decay=0.01)
    scheduler = WarmupCosineLR(optimizer, total_steps=3, warmup_steps=1, min_lr=1e-4)
    precision = setup_device_and_precision(device="cpu", amp_enabled=False)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    stats, global_step = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        precision=precision,
        grad_clip=1.0,
        grad_accum_steps=1,
        max_batches=2,
        log_every=0,
        module_metrics_every=None,
        print_module_diagnostics=False,
        is_main_process=False,
    )

    assert global_step == 2
    assert stats["n_seen_batches"] == 2.0
    assert stats["n_optimizer_steps"] == 2.0
    assert torch.isfinite(torch.tensor(stats["loss"]))
    assert scheduler.state_dict()["step_num"] == 2

    changed = False
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.allclose(before[name], param.detach()):
            changed = True
            break

    assert changed, "Expected at least one trainable parameter to update."
