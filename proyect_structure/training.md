# Training stack recomendado para Mini DeepSeek-V4

Objetivo: construir entrenamiento modular, trazable y extensible.

Estructura deseada:

```python
set_seed(...)
setup_device_and_precision(...)
build_optimizer(...)
build_scheduler(...)
init_ema(...)
save_checkpoint(...)
load_checkpoint(...)
compute_train_metrics(...)
train_step(...)
eval_step(...)
train_one_epoch(...)
evaluate(...)
train_deepseekv4(...)
```

La regla central:

```python
outputs = model(...)
loss = outputs["loss"]
```

El trainer NO debe reconstruir la loss principal afuera, porque el modelo ya combina:

```python
loss = lm_loss + mtp_loss + moe_aux_loss
```

según configuración.

---

# 1. set_seed

Responsabilidad:

```python
def set_seed(seed: int, deterministic: bool = False) -> None:
    ...
```

Debe fijar:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Si `deterministic=True`:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Para entrenamiento rápido normalmente:

```python
deterministic = False
```

Para reproducibilidad de tests:

```python
deterministic = True
```

---

# 2. setup_device_and_precision

Responsabilidad:

```python
def setup_device_and_precision(training_config):
    ...
```

Debe resolver:

```python
device = "cuda" if available else "cpu"
amp_dtype = torch.bfloat16 or torch.float16
use_amp = True/False
GradScaler only if fp16
```

Recomendación:

```python
bf16 > fp16
```

Para CSA/HCA/mHC, `bf16` es más estable que `fp16`.

Config sugerida:

```python
@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "auto"

    max_epochs: int = 1
    max_steps: Optional[int] = None
    grad_accum_steps: int = 1

    optimizer_type: str = "adamw"  # "adamw", "muon_adamw"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    muon_lr: Optional[float] = None
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    scheduler_type: str = "warmup_cosine"
    warmup_steps: int = 100

    grad_clip_norm: float = 1.0

    use_amp: bool = True
    amp_dtype: str = "bf16"  # "bf16", "fp16"

    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after_step: int = 10
    ema_update_every: int = 1

    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500

    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3

    progressive_training: bool = False
```

---

# 3. move_batch_to_device

Responsabilidad:

```python
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    ...
```

Debe mover si existen:

```python
input_ids
labels
attention_mask
mtp_labels
position_ids
```

No asumir que todas las keys existen.

Formato batch esperado:

```python
batch = {
    "input_ids": [B,T],
    "labels": [B,T],
    "attention_mask": optional [B,T],
    "mtp_labels": optional [B,mtp_depth,T],
}
```

---

# 4. build_parameter_groups

Responsabilidad:

```python
def build_parameter_groups(model, config):
    ...
```

Primera versión: grupos AdamW estándar.

```python
decay:
    Linear weights grandes

no_decay:
    bias
    RMSNorm weights
    embedding weights
    lm_head if tied or not
    scalar params
```

Regla práctica:

```python
if param.ndim >= 2 and "embedding" not in name and "lm_head" not in name:
    decay
else:
    no_decay
```

Pero para este proyecto conviene más explícito.

---

# 5. build_muon_adamw_parameter_groups

Responsabilidad:

```python
def build_muon_adamw_parameter_groups(model, config):
    ...
```

Para acercarnos a DeepSeek-style:

```python
Muon group:
    pesos 2D principales de Linear:
        attention projections
        HCA/CSA projections
        MLP projections
        MoE expert projections
        router matrix si es learned
        MTP transform linears

AdamW group:
    embeddings
    lm_head
    MTP vocab heads
    RMSNorm weights
    biases
    scalar params
    mHC alpha params
    mHC static_A/static_B/static_C
    compressor bias_a / bias_b
    Sinkhorn-related small params
```

Criterio conservador:

```python
Muon solo para params 2D de módulos nn.Linear.
AdamW para todo lo demás.
```

Esto evita meter en Muon parámetros que no son matrices de transformación estándar.

Pseudo-lógica:

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    is_linear_weight = param.ndim == 2 and name.endswith(".weight")
    is_embedding = "embedding" in name
    is_lm_head = "lm_head" in name
    is_norm = "norm" in name.lower()
    is_bias = name.endswith(".bias")
    is_mhc_static = "static_" in name or "alpha_" in name
    is_compressor_bias = "bias_a" in name or "bias_b" in name
    is_mtp_vocab_head = "mtp_head.heads" in name

    if optimizer_type == "muon_adamw" and is_linear_weight and not any_exclusion:
        muon_params.append(param)
    else:
        adamw_params.append(param)
```

---

# 6. build_optimizer

Responsabilidad:

```python
def build_optimizer(model, config):
    ...
```

Debe soportar:

```python
optimizer_type = "adamw"
optimizer_type = "muon_adamw"
```

## AdamW baseline

```python
torch.optim.AdamW(
    parameter_groups,
    lr=config.learning_rate,
    betas=config.betas,
    eps=config.eps,
    weight_decay=config.weight_decay,
)
```

## Muon + AdamW

Ideal:

```python
optimizer = HybridMuonAdamW(
    muon_groups=...,
    adamw_groups=...,
)
```

Pero si todavía no tenemos Muon implementado/importado:

```python
raise ImportError explícito
```

o fallback controlado:

```python
if config.optimizer_type == "muon_adamw" and Muon unavailable:
    raise RuntimeError("Muon optimizer requested but not available.")
```

No hacer fallback silencioso porque confunde resultados.

---

# 7. build_scheduler / get_lr

Responsabilidad:

```python
def get_lr(step: int, config: TrainingConfig) -> float:
    ...
```

Scheduler recomendado:

```python
warmup + cosine decay
```

Fórmula:

```python
if step < warmup_steps:
    lr = max_lr * (step + 1) / warmup_steps
else:
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

Función:

```python
def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr
```

Si usamos Muon + AdamW, permitir:

```python
group["lr_scale"]
```

o separar:

```python
adamw_lr = lr
muon_lr = config.muon_lr or lr
```

---

# 8. Progressive schedules

Esto no debe ser obligatorio para el primer smoke training, pero sí conviene diseñarlo.

Responsabilidad:

```python
def apply_progressive_schedule(model, step, config):
    ...
```

Cosas razonables para progresivo:

## 8.1 MTP loss schedule

DeepSeek-style mini:

```python
mtp_loss_weight = 0.3 durante la fase principal
mtp_loss_weight = 0.1 durante decay final
```

Implementación:

```python
if model.use_mtp:
    if step < decay_start_step:
        model.mtp_head.mtp_loss_weight = 0.3
    else:
        model.mtp_head.mtp_loss_weight = 0.1
```

Si el atributo vive en config, actualizar:

```python
model.mtp_head.mtp_loss_weight = current_weight
```

## 8.2 MoE balance warmup

Evitar que aux loss domine al inicio:

```python
balance_scale = min(1.0, step / balance_warmup_steps)
effective_balance_weight = base_weight * balance_scale
```

Como tu MoE guarda pesos en config, lo más limpio es que el trainer pueda modificar:

```python
module.balance_loss_weight
module.sequence_balance_loss_weight
```

si existen.

## 8.3 Router jitter schedule

Al inicio más exploración:

```python
router_jitter_noise high early
router_jitter_noise -> 0 later
```

## 8.4 Sequence length curriculum

Para datasets largos:

```python
current_seq_len = schedule(step)
crop batch to current_seq_len
```

Función:

```python
def crop_batch_to_seq_len(batch, seq_len):
    for key in ["input_ids", "labels", "attention_mask"]:
        batch[key] = batch[key][:, :seq_len]
    if "mtp_labels" in batch:
        batch["mtp_labels"] = batch["mtp_labels"][:, :, :seq_len]
```

Mini schedule:

```python
0% - 30% steps: 128 tokens
30% - 70% steps: 256 tokens
70% - 100% steps: 512 tokens
```

## 8.5 Dense-to-sparse attention schedule

No lo haría dentro del mismo modelo todavía.

Cambiar `attention_type="mha"` a `"csa"` requiere cambiar módulos y pesos. Mejor manejarlo como entrenamiento por etapas:

```python
stage 1:
    train attention_type="mha"

stage 2:
    initialize new model attention_type="csa"
    load compatible weights where shapes match
    train CSA
```

Esto será un script aparte, no dentro de `train_step`.

---

# 9. EMA

Responsabilidad:

```python
class EMA:
    def __init__(model, decay, update_after_step, update_every)
    def update(model, step)
    def store(model)
    def copy_to(model)
    def restore(model)
    def state_dict()
    def load_state_dict()
```

EMA mantiene una sombra de pesos:

```python
shadow[name] = decay * shadow[name] + (1 - decay) * param
```

No aplicar EMA a buffers.

Usos:

```python
training:
    ema.update(model, step)

evaluation:
    ema.store(model)
    ema.copy_to(model)
    eval_step(...)
    ema.restore(model)
```

Para smoke training, EMA opcional.

---

# 10. Checkpoints

Responsabilidad:

```python
def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler_state,
    scaler,
    ema,
    step,
    epoch,
    config,
    metrics,
):
    ...
```

Debe guardar:

```python
model.state_dict()
optimizer.state_dict()
scaler.state_dict() if fp16
ema.state_dict() if use_ema
step
epoch
training_config
model_config
metrics
rng states
```

RNG states:

```python
torch.get_rng_state()
torch.cuda.get_rng_state_all()
numpy.random.get_state()
random.getstate()
```

Load:

```python
def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None, map_location="cpu"):
    ...
```

Debe devolver:

```python
step
epoch
metrics
configs
```

---

# 11. compute_grad_norm

Responsabilidad:

```python
def compute_grad_norm(model) -> torch.Tensor:
    ...
```

Debe ignorar parámetros sin gradiente.

Usar para logging antes/después de clipping.

También:

```python
def has_nonfinite_grads(model) -> bool:
    ...
```

---

# 12. compute_train_metrics

Responsabilidad:

```python
def compute_train_metrics(outputs, batch, model=None) -> dict:
    ...
```

Métricas principales:

```python
loss
lm_loss
mtp_loss
moe_aux_loss
perplexity = exp(lm_loss)
```

Cuidado: perplexity debe usar `lm_loss`, no total loss.

Métricas de tokens:

```python
num_tokens = input_ids.numel()
num_nonpad_tokens = attention_mask.sum() if exists
tokens_per_second
```

Métricas de MoE:

Desde aux por bloque si existe:

```python
router_entropy_mean
expert_fraction_min
expert_fraction_max
expert_fraction_std
balance_loss_mean
sequence_balance_loss_mean
```

Métricas de CSA/HCA:

Si `need_weights=True` ocasionalmente:

```python
local_attention_mass
global_attention_mass
topk_valid_fraction
compressed_valid_fraction
```

Para CSA:

```python
topk_valid_fraction = topk_mask.float().mean()
```

Para HCA:

```python
global_mass = global_attn_weights.sum(dim=-1).mean()
local_mass = local_attn_weights.sum(dim=-1).mean()
```

Métricas mHC:

Si `return_aux=True`:

```python
alpha_A
alpha_B
alpha_C
B_row_error = abs(B.sum(-1)-1).mean()
B_col_error = abs(B.sum(-2)-1).mean()
A_stream0_mean
C_stream0_mean
```

Métricas numéricas:

```python
logits_mean
logits_std
max_abs_logit
grad_norm
lr
```

---

# 13. train_step

Responsabilidad:

```python
def train_step(
    model,
    batch,
    optimizer,
    config,
    step,
    scaler=None,
    ema=None,
) -> dict:
    ...
```

Flujo:

```python
model.train()
batch = move_batch_to_device(batch, device)

apply_progressive_schedule(model, step, config)

with autocast if enabled:
    outputs = model(
        input_ids=batch["input_ids"],
        labels=batch.get("labels"),
        mtp_labels=batch.get("mtp_labels"),
        attention_mask=batch.get("attention_mask"),
        return_aux=should_collect_aux(step),
        need_weights=should_collect_weights(step),
    )

    loss = outputs["loss"] / grad_accum_steps

backward:
    if fp16 scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

if accumulation boundary:
    unscale if scaler
    grad_norm = clip_grad_norm
    optimizer.step
    scaler.update if fp16
    optimizer.zero_grad(set_to_none=True)
    ema.update

return metrics
```

Importante:

```python
if outputs["loss"] is None:
    raise RuntimeError("Model returned loss=None. Did you forget labels?")
```

---

# 14. eval_step

Responsabilidad:

```python
@torch.no_grad()
def eval_step(model, batch, config) -> dict:
    ...
```

Debe:

```python
model.eval()
outputs = model(..., return_aux=False, need_weights=False)
return metrics
```

Para debugging ocasional:

```python
need_weights=True
return_aux=True
```

pero no siempre.

---

# 15. train_one_epoch

Responsabilidad:

```python
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    config,
    epoch,
    global_step,
    scaler=None,
    ema=None,
):
    ...
```

Flujo:

```python
for batch_idx, batch in enumerate(train_loader):
    metrics = train_step(...)

    if global_step % log_every == 0:
        log metrics

    if global_step % eval_every == 0:
        evaluate(...)

    if global_step % save_every == 0:
        save_checkpoint(...)

    global_step += 1

    if max_steps and global_step >= max_steps:
        break
```

Debe devolver:

```python
global_step
last_metrics
```

---

# 16. evaluate

Responsabilidad:

```python
def evaluate(model, val_loader, config, max_batches=None, ema=None):
    ...
```

Si EMA:

```python
ema.store(model)
ema.copy_to(model)
run eval
ema.restore(model)
```

Promediar:

```python
loss
lm_loss
mtp_loss
moe_aux_loss
perplexity
```

No usar grad.

---

# 17. train_deepseekv4

Wrapper final:

```python
def train_deepseekv4(
    model,
    train_loader,
    val_loader,
    training_config,
    resume_from=None,
):
    set_seed(...)
    setup device
    model.to(device)

    optimizer = build_optimizer(...)
    scaler = build_scaler(...)
    ema = EMA(...) if use_ema
    maybe load checkpoint

    for epoch in range(start_epoch, max_epochs):
        global_step, metrics = train_one_epoch(...)

        if val_loader:
            val_metrics = evaluate(...)

        save checkpoint end epoch

        if max_steps reached:
            break

    return training_state
```

---

# 18. Losses que vale la pena supervisar

Siempre:

```python
loss
lm_loss
perplexity
```

Si MTP:

```python
mtp_loss
raw_mtp_loss
mtp_loss_per_depth
mtp_loss_weight
```

Si MoE:

```python
moe_aux_loss
balance_loss
sequence_balance_loss
router_entropy
expert_fraction_min
expert_fraction_max
expert_fraction_std
dead_experts = count(expert_fraction == 0)
```

Si CSA/HCA:

```python
global_attention_mass
local_attention_mass
topk_valid_fraction
compressed_valid_fraction
```

Si mHC:

```python
alpha_A
alpha_B
alpha_C
B_row_error
B_col_error
```

Numerics:

```python
grad_norm
param_norm optional
logits_std
max_abs_logit
nan_or_inf_flag
```

---

# 19. Tests mínimos del training stack

Antes de entrenar en serio:

```python
test_get_lr_warmup_cosine
test_move_batch_to_device
test_build_adamw_optimizer
test_parameter_groups_no_duplicates
test_train_step_returns_metrics
test_train_step_updates_parameters
test_gradient_clipping_runs
test_eval_step_no_grad
test_checkpoint_save_load_roundtrip
test_ema_update_and_restore
test_tiny_overfit_one_batch_reduces_loss
```

Configuraciones mínimas a probar:

```python
mha + dense
hca + dense
csa + dense
csa + moe
csa + moe + mtp
csa + moe + mhc + mtp
```

---

# 20. Orden recomendado de implementación

Implementar en este orden:

```python
1. TrainingConfig
2. set_seed
3. move_batch_to_device
4. get_lr / set_optimizer_lr
5. build_adamw_optimizer
6. compute_grad_norm
7. train_step sin AMP ni EMA
8. eval_step
9. train_one_epoch
10. checkpoint save/load
11. EMA
12. AMP/autocast
13. metrics avanzadas
14. Muon parameter grouping
15. Hybrid Muon+AdamW
16. progressive schedules
17. train_deepseekv4 wrapper
```

No empieces con Muon. Primero asegúrate de que AdamW reduce loss en un batch. Luego activamos Muon.