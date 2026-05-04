# DeepSeek-V4 Mini: Guía inicial del proyecto

> Objetivo: usar el paper de DeepSeek-V4 como mapa para construir una versión **mini, pedagógica y testeable** de sus componentes arquitectónicos centrales. No se busca replicar el modelo real ni entrenarlo a escala frontier.

---

## 1. ¿Qué es DeepSeek-V4?

DeepSeek-V4 es una familia de modelos **Transformer autoregresivos Mixture-of-Experts (MoE)** diseñados para soportar contextos extremadamente largos, hasta **1 millón de tokens**. El paper presenta dos versiones principales:

- **DeepSeek-V4-Pro**: 1.6T parámetros totales, 49B activados por token.
- **DeepSeek-V4-Flash**: 284B parámetros totales, 13B activados por token.

La idea central no es abandonar el Transformer, sino modificar sus cuellos de botella principales:

1. **Atención demasiado costosa en contexto largo**  
   Solución: atención híbrida con **Compressed Sparse Attention (CSA)** y **Heavily Compressed Attention (HCA)**.

2. **Necesidad de mucha capacidad sin activar todos los parámetros**  
   Solución: **DeepSeekMoE**, con expertos compartidos y expertos ruteados.

3. **Entrenamiento profundo y estable**  
   Solución: **Manifold-Constrained Hyper-Connections (mHC)**, normalizaciones adicionales, attention sink y **Muon optimizer**.

4. **Mejor señal de entrenamiento autoregresivo**  
   Solución: **Multi-Token Prediction (MTP)**, heredado de DeepSeek-V3.

En forma compacta:

```text
DeepSeek-V4 = Transformer causal
            + DeepSeekMoE
            + hybrid attention (CSA/HCA)
            + mHC residual stream
            + MTP
            + Muon
            + infraestructura industrial para escalar
```

Para nuestro proyecto, la parte importante es separar:

- **Arquitectura**: lo que define el forward.
- **Entrenamiento**: pérdidas, optimizador y protocolo de aprendizaje.
- **Ingeniería industrial**: kernels, paralelismo, cuantización, cache en disco y sistemas distribuidos.

---

## 2. Componentes que sí deberíamos implementar

Estos componentes son el núcleo del proyecto. Algunos pueden ir en etapas tardías, pero deberían estar en el scope final.

### 2.1 Transformer causal base

Implementar primero:

```text
TokenEmbedding
RMSNorm
Causal LM head
RoPE o partial RoPE
Causal masking
Generation loop mínimo
```

Esto no es lo más novedoso, pero es la base sobre la cual viven CSA, HCA, MoE y mHC.

---

### 2.2 DeepSeekMoE mini

DeepSeek-V4 usa MoE en las capas feed-forward. Para una versión mini:

```text
x -> router -> top-k expertos ruteados
x -> shared expert
output = suma ponderada de expertos ruteados + shared expert
```

Implementar:

```text
TopKRouter
RoutedExpertMLP
SharedExpertMLP
MoELayer
RoutingStats
OptionalBalanceLoss
```

Configuración mini sugerida:

```python
num_experts = 8
top_k = 2
num_shared_experts = 1
d_model = 256
expert_hidden_dim = 512
```

Etapas posteriores:

- Hash routing para las primeras capas.
- Métricas de carga por experto.
- Ablation: MLP denso vs MoE.

---

### 2.3 HCA: Heavily Compressed Attention

HCA comprime agresivamente los KV tokens. En vez de atender a todos los tokens pasados, cada query atiende a bloques comprimidos.

Idea:

```text
H: [B, T, D]

KV compression:
    [B, T, D] -> [B, ceil(T / m_prime), C]

Attention:
    query tokens atienden a compressed_KV + sliding_window_KV
```

Implementar antes que CSA porque es más simple.

Módulos:

```text
TokenCompressor
CompressedKVBuilder
SharedKVMQA
SlidingWindowBranch
HCAAttention
```

Configuración mini:

```python
m_prime = 16
window_size = 32
num_heads = 4
head_dim = 64
d_model = 256
```

Tests obligatorios:

```text
compressed_length == ceil(T / m_prime)
output shape == [B, T, D]
no future leakage
gradients finite
```

---

### 2.4 CSA: Compressed Sparse Attention

CSA es la parte más importante y más difícil. Combina compresión y selección sparse.

Idea:

```text
1. Comprimir KV por bloques de m tokens.
2. Construir indexer keys comprimidas.
3. Para cada query token, calcular scores contra bloques comprimidos.
4. Seleccionar top-k bloques relevantes.
5. Hacer atención sobre esos bloques + ventana local reciente.
```

Forward conceptual:

```text
H: [B, T, D]

C_comp = compress_kv(H)                     # [B, T/m, C]
K_index = compress_indexer_keys(H)          # [B, T/m, C_index]
Q_index = build_indexer_queries(H)          # [B, T, n_index_heads, C_index]

scores = indexer(Q_index, K_index)          # [B, T, T/m]
selected = topk(scores, k)                  # [B, T, k]
selected_kv = gather(C_comp, selected)      # [B, T, k, C]
local_kv = sliding_window_kv(H)             # [B, T, window, C]

out = MQA(query=Q, key/value=selected_kv + local_kv)
```

Implementar en versión pedagógica, no optimizada.

Configuración mini:

```python
m = 4
csa_top_k = 8
window_size = 32
num_heads = 4
head_dim = 64
indexer_heads = 4
indexer_dim = 64
```

Tests obligatorios:

```text
top-k no selecciona bloques futuros
output shape == [B, T, D]
causal test pasa
si se modifican tokens futuros, logits pasados no cambian
CSA reduce longitud efectiva de KV
```

---

### 2.5 Sliding window branch

Tanto CSA como HCA necesitan una rama local no comprimida. Esto evita que el modelo pierda información fina de los tokens recientes.

Idea:

```text
global compressed attention + exact local attention
```

Implementar dentro de HCA y CSA.

Tests:

```text
cada query solo ve tokens <= t
window respeta tamaño máximo
no hay acceso a tokens futuros dentro del bloque comprimido actual
```

---

### 2.6 mHC: Manifold-Constrained Hyper-Connections

mHC reemplaza el residual stream simple por un residual stream expandido.

Residual clásico:

```text
x_{l+1} = x_l + F_l(x_l)
```

mHC conceptual:

```text
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
```

Donde:

```text
X_l: residual stream expandido [B, T, n_hc, D]
A_l: mezcla previa para producir input del bloque
B_l: mezcla residual restringida
C_l: mezcla posterior para insertar output del bloque
```

La idea matemática clave: restringir `B_l` a ser aproximadamente doblemente estocástica mediante Sinkhorn-Knopp.

Implementar:

```text
ExpandedResidualStream
PreBlockMixing
ResidualMixing
PostBlockMixing
SinkhornProjection
ManifoldHyperConnection
```

Configuración mini:

```python
n_hc = 4
sinkhorn_iters = 10  # luego 20
```

Tests obligatorios:

```text
B >= 0
B.sum(dim=-1) ≈ 1
B.sum(dim=-2) ≈ 1
output shape correcto
gradients finite
no NaNs
```

---

### 2.7 MTP: Multi-Token Prediction

MTP añade objetivos auxiliares para predecir más de un token futuro.

LM estándar:

```text
h_t -> predice x_{t+1}
```

Con MTP:

```text
h_t -> predice x_{t+1}
h_t -> predice x_{t+2}
h_t -> predice x_{t+3}  # opcional
```

Implementar en etapa posterior:

```text
MTPHead
MTPLoss
loss = lm_loss + lambda_mtp * mtp_loss
```

Configuración mini:

```python
mtp_depth = 1  # primero solo x_{t+2}
lambda_mtp = 0.1
```

---

### 2.8 Muon optimizer

Muon no es arquitectura, pero sí es parte importante del entrenamiento de DeepSeek-V4. Para nuestro proyecto debe ser experimental, no requisito inicial.

Plan:

1. Primero entrenar todo con AdamW.
2. Luego implementar Muon para matrices 2D.
3. Excluir embeddings, RMSNorm, biases y parámetros escalares.
4. Comparar AdamW vs Muon en tiny training.

Tests:

```text
optimizer step no produce NaNs
loss baja en tiny overfit
matrices reciben updates
parámetros excluidos usan AdamW
```

---

## 3. Componentes de ingeniería que NO deberíamos implementar

Estas partes pertenecen a la escala industrial de DeepSeek-V4. Son muy importantes para el modelo real, pero no para una implementación académica desde cero.

### 3.1 No implementar en el proyecto inicial

```text
FP4 quantization-aware training real
TileLang kernels
fused MoE kernels
expert parallelism distribuido
all-to-all communication
communication-computation overlap
contextual parallelism
on-disk KV cache
heterogeneous KV cache industrial
extended autograd custom
ZeRO híbrido para Muon
batch-invariant deterministic kernel libraries
pre-training con trillones de tokens
post-training completo con SFT + GRPO + OPD
```

### 3.2 Qué hacer en su lugar

| Ingeniería real de DeepSeek-V4 | Versión razonable para nosotros |
|---|---|
| FP4 QAT | no implementar; quizá fake quantization después |
| TileLang kernels | PyTorch puro |
| Fused MoE kernels | MoE simple vectorizado |
| Expert parallelism | single-GPU / CPU implementation |
| On-disk KV cache | KV cache comprimida en memoria |
| Contextual parallelism | secuencias pequeñas |
| GRPO / RL | no implementar |
| On-policy distillation | no implementar inicialmente |
| 32T+ tokens | tiny dataset / smoke training |

La regla de oro:

```text
Implementamos arquitectura y validación.
No implementamos infraestructura frontier.
```

---

## 4. Forward del modelo: sección clave

Esta sección define cómo debería fluir el modelo completo.

### 4.1 Forward general

Input:

```text
input_ids: [B, T]
```

Flujo general:

```text
1. token embedding
2. inicializar residual stream expandido X
3. repetir por cada bloque Transformer:
   a. pre-block mixing para attention
   b. attention CSA/HCA/SWA
   c. post-block mixing
   d. residual mixing
   e. pre-block mixing para MoE
   f. DeepSeekMoE
   g. post-block mixing
   h. residual mixing
4. colapsar residual stream expandido a hidden states
5. LM head
6. MTP heads opcionales
7. calcular losses si hay labels
```

Pseudocódigo:

```python
def forward(input_ids, labels=None):
    h = token_embedding(input_ids)          # [B, T, D]
    X = expand_residual(h)                  # [B, T, n_hc, D]

    for layer_idx, block in enumerate(blocks):
        # ----- Attention sublayer -----
        h_attn_in = block.mhc_attn.pre_mix(X)       # [B, T, D]

        if block.attn_type == "hca":
            h_attn = block.hca(h_attn_in)           # [B, T, D]
        elif block.attn_type == "csa":
            h_attn = block.csa(h_attn_in)           # [B, T, D]
        elif block.attn_type == "swa":
            h_attn = block.swa(h_attn_in)           # [B, T, D]

        X = block.mhc_attn.post_and_residual_mix(
            X=X,
            F_out=h_attn
        )                                           # [B, T, n_hc, D]

        # ----- MoE sublayer -----
        h_moe_in = block.mhc_moe.pre_mix(X)         # [B, T, D]
        h_moe, router_stats = block.moe(h_moe_in)   # [B, T, D]

        X = block.mhc_moe.post_and_residual_mix(
            X=X,
            F_out=h_moe
        )                                           # [B, T, n_hc, D]

    h_final = collapse_residual(X)                  # [B, T, D]
    logits = lm_head(h_final)                       # [B, T, vocab_size]

    loss = None
    if labels is not None:
        lm_loss = causal_lm_loss(logits, labels)
        loss = lm_loss

        if use_mtp:
            mtp_logits = mtp_heads(h_final)
            mtp_loss = compute_mtp_loss(mtp_logits, labels)
            loss = loss + lambda_mtp * mtp_loss

        if use_moe_balance_loss:
            loss = loss + lambda_balance * router_balance_loss(router_stats)

    return {
        "loss": loss,
        "logits": logits,
        "hidden_states": h_final,
        "router_stats": router_stats,
    }
```

---

### 4.2 Forward de un bloque

Un bloque minimal tendría:

```text
MiniDeepSeekV4Block
    - attention_type: "hca" | "csa" | "swa"
    - mHC para attention
    - attention module
    - mHC para MoE
    - MoE module
```

Pseudocódigo:

```python
class MiniDeepSeekV4Block(nn.Module):
    def forward(self, X):
        # Attention
        h = self.attn_mhc.pre_mix(X)
        h = self.attn_norm(h)
        h = self.attention(h)
        X = self.attn_mhc.update(X, h)

        # MoE
        h = self.moe_mhc.pre_mix(X)
        h = self.moe_norm(h)
        h, stats = self.moe(h)
        X = self.moe_mhc.update(X, h)

        return X, stats
```

---

### 4.3 Forward de HCA

```python
def hca_forward(h):
    # h: [B, T, D]
    C = W_kv(h)                         # [B, T, C]
    Z = W_z(h)                          # [B, T, C]

    C_comp = compress_blocks(C, Z, m_prime)   # [B, T_comp, C]

    Q = build_queries(h)                # [B, T, n_heads, C]
    local_kv = build_sliding_window(h)  # [B, T, window, C]

    out = shared_kv_mqa(
        queries=Q,
        compressed_kv=C_comp,
        local_kv=local_kv,
        causal=True,
    )                                   # [B, T, D]

    return out
```

Key idea:

```text
HCA = dense attention sobre KV fuertemente comprimido + ventana local exacta
```

---

### 4.4 Forward de CSA

```python
def csa_forward(h):
    # h: [B, T, D]

    # 1. Compressed KV
    C_comp = compress_kv_overlapping(h, m)          # [B, T_comp, C]

    # 2. Compressed indexer keys
    K_index = compress_indexer_keys(h, m)           # [B, T_comp, C_index]

    # 3. Indexer queries
    Q_index = build_indexer_queries(h)              # [B, T, n_index_heads, C_index]

    # 4. Scores over compressed blocks
    scores = compute_index_scores(Q_index, K_index) # [B, T, T_comp]
    scores = apply_compressed_causal_mask(scores)

    # 5. Top-k compressed blocks per query
    topk_idx = scores.topk(k=csa_top_k, dim=-1).indices
    selected_kv = gather_compressed_kv(C_comp, topk_idx) # [B, T, k, C]

    # 6. Local exact window
    local_kv = build_sliding_window(h)              # [B, T, window, C]

    # 7. Core attention
    Q = build_queries(h)                            # [B, T, n_heads, C]
    out = shared_kv_mqa(
        queries=Q,
        selected_compressed_kv=selected_kv,
        local_kv=local_kv,
        causal=True,
    )                                               # [B, T, D]

    return out
```

Key idea:

```text
CSA = compressed KV + learned sparse retrieval + local exact attention
```

---

### 4.5 Forward de MoE

```python
def moe_forward(h):
    # h: [B, T, D]
    router_logits = router(h)                         # [B, T, E]
    router_scores = sqrt_softplus(router_logits)       # [B, T, E]

    topk_scores, topk_idx = topk(router_scores, k)     # [B, T, k]
    weights = normalize(topk_scores)                   # [B, T, k]

    routed_out = dispatch_to_experts(h, topk_idx)      # [B, T, D]
    shared_out = shared_expert(h)                      # [B, T, D]

    out = routed_out + shared_out

    stats = {
        "topk_idx": topk_idx,
        "router_scores": router_scores,
        "expert_load": compute_expert_load(topk_idx),
    }

    return out, stats
```

---

### 4.6 Forward de mHC

```python
def mhc_update(X, F_out):
    # X: [B, T, n_hc, D]
    # F_out: [B, T, D]

    A_raw, B_raw, C_raw = generate_dynamic_params(X)

    A = sigmoid(A_raw)                 # [B, T, 1, n_hc]
    C = 2 * sigmoid(C_raw)             # [B, T, n_hc, 1]
    B = sinkhorn(exp(B_raw))           # [B, T, n_hc, n_hc]

    residual_part = B @ X              # [B, T, n_hc, D]
    update_part = C * F_out.unsqueeze(-2)  # [B, T, n_hc, D]

    X_next = residual_part + update_part
    return X_next
```

---

## 5. Scope del proyecto

### 5.1 Nombre tentativo

Opciones:

```text
deepseek-v4-mini-from-scratch
compressed-moe-transformer
mini-compressed-moe-llm
long-context-moe-transformer
```

Nombre recomendado:

```text
compressed-moe-transformer
```

Subtítulo:

```text
A research-scale PyTorch implementation of a DeepSeek-V4-style compressed-attention MoE Transformer.
```

---

### 5.2 Objetivo del proyecto

Construir una implementación mini en PyTorch de los componentes centrales de DeepSeek-V4, con énfasis en:

```text
claridad arquitectónica
correctitud de shapes
tests de causalidad
smoke training
ablation studies
lectura fiel del paper
```

No buscamos:

```text
entrenar un LLM competitivo
replicar pesos de DeepSeek
hacer inference industrial
soportar realmente 1M tokens
hacer distributed training
```

---

### 5.3 Estructura sugerida del repositorio

```text
compressed-moe-transformer/
│
├── README.md
├── pyproject.toml
├── configs/
│   ├── tiny_hca.yaml
│   ├── tiny_csa.yaml
│   ├── tiny_full.yaml
│
├── src/
│   └── cmt/
│       ├── model.py
│       ├── config.py
│       ├── layers/
│       │   ├── rmsnorm.py
│       │   ├── rope.py
│       │   ├── attention_hca.py
│       │   ├── attention_csa.py
│       │   ├── moe.py
│       │   ├── mhc.py
│       │   └── mtp.py
│       ├── optim/
│       │   └── muon.py
│       ├── training/
│       │   ├── train.py
│       │   ├── losses.py
│       │   └── data.py
│       └── generation.py
│
├── tests/
│   ├── test_shapes.py
│   ├── test_causality.py
│   ├── test_hca.py
│   ├── test_csa.py
│   ├── test_moe.py
│   ├── test_mhc.py
│   ├── test_mtp.py
│   └── test_tiny_training.py
│
└── notebooks/
    ├── 01_hca_walkthrough.ipynb
    ├── 02_csa_walkthrough.ipynb
    └── 03_tiny_training.ipynb
```

---

## 6. Etapas de implementación

### Etapa 1: Base LLM

```text
RMSNorm
RoPE
causal attention normal
MLP normal
LM head
forward/backward
generation mínimo
```

Entrega: mini GPT funcional.

---

### Etapa 2: MoE

```text
Top-k router
shared expert
routed experts
expert load stats
MoE FFN replacement
```

Entrega: Transformer MoE tiny.

---

### Etapa 3: HCA

```text
KV compression
compressed dense attention
sliding window branch
causality tests
```

Entrega: long-context compressed attention simple.

---

### Etapa 4: CSA

```text
overlapping compression
indexer queries/keys
top-k selection
sparse compressed attention
sliding window branch
```

Entrega: núcleo diferenciador del proyecto.

---

### Etapa 5: mHC

```text
expanded residual stream
pre/post/residual mixing
Sinkhorn projection
mHC block integration
```

Entrega: residual stream estilo DeepSeek-V4.

---

### Etapa 6: MTP + Muon

```text
MTP head
MTP loss
Muon optimizer experimental
AdamW vs Muon comparison
```

Entrega: entrenamiento más cercano al paper.

---

## 7. Ablations y research directions

Después de tener el modelo funcionando, las mejores ablations serían:

### 7.1 Atención

```text
Vanilla causal attention vs HCA
Vanilla causal attention vs CSA
HCA vs CSA
CSA top-k pequeño vs grande
HCA compression rate m_prime bajo vs alto
Sliding window on/off
Partial RoPE vs full RoPE
Attention sink on/off
```

Preguntas:

```text
¿Cuánta pérdida de calidad introduce la compresión?
¿CSA recupera mejor información relevante que HCA?
¿Qué tanto ayuda la ventana local?
```

---

### 7.2 MoE

```text
Dense MLP vs MoE
Top-1 vs Top-2 routing
shared expert on/off
sqrt_softplus router vs softmax router
hash routing en primeras capas vs learned routing
```

Preguntas:

```text
¿MoE mejora capacidad bajo presupuesto pequeño?
¿El router colapsa en tiny training?
¿Los shared experts estabilizan el entrenamiento?
```

---

### 7.3 mHC

```text
Residual clásico vs mHC
n_hc = 2, 4, 8
Sinkhorn iterations = 1, 5, 10, 20
static mHC vs dynamic mHC
```

Preguntas:

```text
¿mHC mejora estabilidad?
¿Reduce exploding/vanishing activations?
¿Aumenta demasiado el costo de memoria?
```

---

### 7.4 MTP

```text
LM only vs LM + MTP
mtp_depth = 1 vs 2
lambda_mtp bajo vs alto
```

Preguntas:

```text
¿MTP acelera la caída de loss?
¿Mejora representaciones intermedias?
¿Desestabiliza modelos pequeños?
```

---

### 7.5 Muon

```text
AdamW vs Muon
Muon only matrices 2D
Muon + AdamW hybrid
Newton-Schulz iterations bajas vs altas
```

Preguntas:

```text
¿Muon converge más rápido en tiny training?
¿Es estable en modelos pequeños?
¿Vale la pena implementarlo para este repo?
```

---

## 8. Criterio de éxito del proyecto

El proyecto será exitoso si logra:

```text
1. Implementar forward completo sin NaNs.
2. Pasar tests de shapes, causalidad y gradientes.
3. Entrenar en un dataset pequeño y mostrar caída de loss.
4. Generar texto de smoke test.
5. Comparar HCA/CSA/MoE/mHC mediante ablations simples.
6. Tener documentación clara tipo paper-to-code.
```

No se medirá por obtener un LLM útil. Se medirá por:

```text
fidelidad arquitectónica
claridad matemática
calidad de ingeniería en PyTorch
calidad de tests
capacidad de explicar una arquitectura frontier desde código
```

---

## 9. Resumen final de scope

### Implementar sí o sí

```text
Transformer causal base
DeepSeekMoE mini
HCA
CSA
sliding window branch
mHC
MTP
smoke training
causality tests
shape tests
```

### Implementar después si hay tiempo

```text
Muon
attention sink
partial RoPE detallado
hash routing
KV cache para generación
ablation suite completa
```

### No implementar

```text
FP4 real
TileLang
kernels custom
expert parallelism
training distribuido
on-disk KV cache
GRPO
on-policy distillation
million-token inference real
pre-training masivo
```

---

## 10. Frase guía del proyecto

```text
We are not replicating DeepSeek-V4 at scale.
We are implementing a research-scale, from-scratch version of its core architectural ideas.
```

En español:

```text
No vamos a replicar DeepSeek-V4 a escala.
Vamos a implementar desde cero una versión mini, rigurosa y testeable de sus ideas arquitectónicas centrales.
```
