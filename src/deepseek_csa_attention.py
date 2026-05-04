# ============================================================
# CANONICAL CSA ATTENTION
# ============================================================


from src.csa_light_indexer import *
from src.transformer_modules.rope import * 

class CSAAttention(nn.Module):
    """
    Canonical CSA mini implementation.

    Core pieces:
        - Overlapped a/b compression
        - Low-rank shared query path
        - Lightning indexer
        - Top-k causal sparse global attention
        - Local sliding-window branch with separate local KV path
        - Shared KV MQA
        - Optional attention sink
        - Grouped output projection
        - RoPE / partial RoPE

    Input:
        x: [B,T,d_model]

    Output:
        out: [B,T,d_model]

    If need_weights=True:
        out, aux
    """

    def __init__(self, config: CSAConfig):
        super().__init__()

        config.validate()
        self.config = config

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.d_model // config.n_heads
        )

        self.inner_dim = self.n_heads * self.head_dim

        self.compression_factor = config.compression_factor
        self.top_k = config.top_k
        self.window_size = config.window_size

        self.indexer_dim = config.indexer_dim
        self.n_indexer_heads = config.n_indexer_heads
        self.query_compression_dim = (
            config.query_compression_dim
            if config.query_compression_dim is not None
            else config.indexer_dim
        )

        self.max_seq_len = config.max_seq_len
        self.use_rope = config.use_rope
        self.use_attention_sink = config.use_attention_sink
        self.use_grouped_output_projection = config.use_grouped_output_projection
        self.use_indexer_score_bias = config.use_indexer_score_bias
        self.use_separate_local_kv = config.use_separate_local_kv

        # ----------------------------------------------------
        # Shared low-rank query path
        # ----------------------------------------------------
        self.q_down_proj = nn.Linear(
            self.d_model,
            self.query_compression_dim,
            bias=config.use_bias,
        )

        self.q_up_proj = nn.Linear(
            self.query_compression_dim,
            self.inner_dim,
            bias=config.use_bias,
        )

        self.index_q_up_proj = nn.Linear(
            self.query_compression_dim,
            self.n_indexer_heads * self.indexer_dim,
            bias=config.use_bias,
        )

        self.index_weight_proj = nn.Linear(
            self.d_model,
            self.n_indexer_heads,
            bias=config.use_bias,
        )

        # ----------------------------------------------------
        # Compressed KV path: a/b branches
        # ----------------------------------------------------
        self.a_kv_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.b_kv_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.a_z_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        self.b_z_proj = nn.Linear(
            self.d_model,
            self.head_dim,
            bias=config.use_bias,
        )

        # ----------------------------------------------------
        # Separate local exact KV branch
        # ----------------------------------------------------
        if self.use_separate_local_kv:
            self.local_kv_proj = nn.Linear(
                self.d_model,
                self.head_dim,
                bias=config.use_bias,
            )
        else:
            self.local_kv_proj = None

        # ----------------------------------------------------
        # Index key path: a/b branches
        # ----------------------------------------------------
        self.a_index_kv_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        self.b_index_kv_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        self.a_index_z_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        self.b_index_z_proj = nn.Linear(
            self.d_model,
            self.indexer_dim,
            bias=config.use_bias,
        )

        # ----------------------------------------------------
        # Compressors + indexer
        # ----------------------------------------------------
        self.kv_compressor = CSAOverlappedCompressor(
            compression_factor=config.compression_factor,
            dim=self.head_dim,
            init_std=config.init_std,
        )

        self.index_compressor = CSAOverlappedCompressor(
            compression_factor=config.compression_factor,
            dim=self.indexer_dim,
            init_std=config.init_std,
        )

        self.indexer = CSALightningIndexer(
            compression_factor=config.compression_factor,
            top_k=config.top_k,
        )

        # ----------------------------------------------------
        # Optional attention sink
        # ----------------------------------------------------
        if self.use_attention_sink:
            self.sink_k = nn.Parameter(torch.empty(1, 1, self.head_dim))
            self.sink_v = nn.Parameter(torch.empty(1, 1, self.head_dim))
        else:
            self.sink_k = None
            self.sink_v = None

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
        if self.use_grouped_output_projection:
            num_groups = (
                config.output_projection_groups
                if config.output_projection_groups is not None
                else self.n_heads
            )

            self.out_proj = GroupedOutputProjection(
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                num_groups=num_groups,
                bias=config.use_bias,
                init_std=config.init_std,
            )
        else:
            self.out_proj = nn.Linear(
                self.inner_dim,
                self.d_model,
                bias=config.use_bias,
            )

        if self.use_rope:
            # Assumes RotaryEmbedding exists in your project and accepts:
            #   x: [B,T,H,Dh]
            #   position_ids: None, [T], or [B,T]
            #   start_pos: int
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                rotary_dim=config.rotary_dim,
                base=config.rope_theta,
            )
        else:
            self.rope = None

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        modules = [
            self.q_down_proj,
            self.q_up_proj,
            self.index_q_up_proj,
            self.index_weight_proj,
            self.a_kv_proj,
            self.b_kv_proj,
            self.a_z_proj,
            self.b_z_proj,
            self.a_index_kv_proj,
            self.b_index_kv_proj,
            self.a_index_z_proj,
            self.b_index_z_proj,
        ]

        if self.local_kv_proj is not None:
            modules.append(self.local_kv_proj)

        if not self.use_grouped_output_projection:
            modules.append(self.out_proj)

        for module in modules:
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if self.use_attention_sink:
            nn.init.normal_(self.sink_k, mean=0.0, std=self.config.init_std)
            nn.init.normal_(self.sink_v, mean=0.0, std=self.config.init_std)

    def _shape_q(self, q: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.shape
        return q.view(B, T, self.n_heads, self.head_dim)

    def _shape_index_q(self, index_q: torch.Tensor) -> torch.Tensor:
        B, T, _ = index_q.shape
        return index_q.view(B, T, self.n_indexer_heads, self.indexer_dim)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must have shape [B,T], "
                f"got {tuple(attention_mask.shape)}"
            )

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        return attention_mask

    def _build_local_allowed_mask(
        self,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_pos = torch.arange(T, device=device)[:, None]
        k_pos = torch.arange(T, device=device)[None, :]

        causal = k_pos <= q_pos
        in_window = (q_pos - k_pos) < self.window_size

        return causal & in_window

    def _gather_selected(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        values:  [B,S,D]
        indices: [B,T,K]
        return:  [B,T,K,D]
        """
        B, S, D = values.shape
        B_i, T, K = indices.shape

        if B_i != B:
            raise ValueError(f"Batch mismatch: values B={B}, indices B={B_i}")

        source = values[:, None, :, :].expand(B, T, S, D)
        idx = indices[..., None].expand(B, T, K, D)

        return torch.gather(source, dim=2, index=idx)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                f"CSAAttention expects x [B,T,d_model], got {tuple(x.shape)}"
            )

        B, T, C_model = x.shape

        if C_model != self.d_model:
            raise ValueError(
                f"Expected hidden size {self.d_model}, got {C_model}"
            )

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        if attention_mask is not None:
            attention_mask = self._validate_attention_mask(
                attention_mask=attention_mask,
                batch_size=B,
                seq_len=T,
            )

        # ----------------------------------------------------
        # Shared low-rank query path
        # ----------------------------------------------------
        q_latent = self.q_down_proj(x)          # [B,T,Qc]

        q = self.q_up_proj(q_latent)
        q = self._shape_q(q)                    # [B,T,H,Dh]

        index_q = self.index_q_up_proj(q_latent)
        index_q = self._shape_index_q(index_q)  # [B,T,H_i,I]

        index_weights = self.index_weight_proj(x)  # [B,T,H_i]

        # ----------------------------------------------------
        # KV a/b projections
        # ----------------------------------------------------
        C_a = self.a_kv_proj(x)
        C_b = self.b_kv_proj(x)
        Z_a = self.a_z_proj(x)
        Z_b = self.b_z_proj(x)
        # [B,T,Dh]

        if self.local_kv_proj is not None:
            C_local = self.local_kv_proj(x)      # [B,T,Dh]
        else:
            C_local = C_a                        # backward-compatible fallback

        I_a = self.a_index_kv_proj(x)
        I_b = self.b_index_kv_proj(x)
        IZ_a = self.a_index_z_proj(x)
        IZ_b = self.b_index_z_proj(x)
        # [B,T,I]

        # ----------------------------------------------------
        # RoPE on query
        # ----------------------------------------------------
        if self.rope is not None:
            q = self.rope(
                q,
                position_ids=position_ids,
                start_pos=start_pos,
            )

        # ----------------------------------------------------
        # Overlapped compression: KV and index keys
        # ----------------------------------------------------
        C_comp, comp_valid_mask, comp_position_ids = self.kv_compressor(
            C_a=C_a,
            C_b=C_b,
            Z_a=Z_a,
            Z_b=Z_b,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_pos=start_pos,
        )

        I_comp, index_valid_mask, _ = self.index_compressor(
            C_a=I_a,
            C_b=I_b,
            Z_a=IZ_a,
            Z_b=IZ_b,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_pos=start_pos,
        )

        if not torch.equal(comp_valid_mask, index_valid_mask):
            raise RuntimeError(
                "KV compressed valid mask differs from index compressed valid mask."
            )

        S = C_comp.shape[1]

        # ----------------------------------------------------
        # RoPE on compressed global keys and local keys
        # ----------------------------------------------------
        if self.rope is not None:
            K_global_all = C_comp[:, :, None, :]  # [B,S,1,Dh]
            K_global_all = self.rope(
                K_global_all,
                position_ids=comp_position_ids,
                start_pos=0,
            )
            K_global_all = K_global_all[:, :, 0, :]  # [B,S,Dh]

            K_local = C_local[:, :, None, :]  # [B,T,1,Dh]
            K_local = self.rope(
                K_local,
                position_ids=position_ids,
                start_pos=start_pos,
            )
            K_local = K_local[:, :, 0, :]  # [B,T,Dh]
        else:
            K_global_all = C_comp
            K_local = C_local

        V_global_all = C_comp
        V_local = C_local

        # ----------------------------------------------------
        # Lightning indexer top-k selection
        # ----------------------------------------------------
        if need_weights:
            topk_indices, topk_scores, topk_mask, index_scores = self.indexer(
                index_q=index_q,
                index_weights=index_weights,
                I_comp=I_comp,
                comp_valid_mask=comp_valid_mask,
                need_scores=True,
            )
        else:
            topk_indices, topk_scores, topk_mask = self.indexer(
                index_q=index_q,
                index_weights=index_weights,
                I_comp=I_comp,
                comp_valid_mask=comp_valid_mask,
                need_scores=False,
            )
            index_scores = None

        K_eff = topk_indices.shape[-1]

        # ----------------------------------------------------
        # Gather selected global K/V
        # ----------------------------------------------------
        K_selected = self._gather_selected(K_global_all, topk_indices)   # [B,T,K,Dh]
        V_selected = self._gather_selected(V_global_all, topk_indices)   # [B,T,K,Dh]

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------
        q = q  # [B,T,H,Dh]

        scores_parts = []
        allowed_parts = []

        # -------------------------
        # Optional attention sink
        # -------------------------
        if self.use_attention_sink:
            K_sink = self.sink_k.expand(B, -1, -1)  # [B,1,Dh]
            scores_sink = torch.einsum(
                "bthd,bsd->bhts",
                q,
                K_sink,
            ) / math.sqrt(self.head_dim)             # [B,H,T,1]

            sink_allowed = torch.ones(
                B,
                self.n_heads,
                T,
                1,
                device=x.device,
                dtype=torch.bool,
            )

            scores_parts.append(scores_sink)
            allowed_parts.append(sink_allowed)

        # -------------------------
        # Sparse selected global scores
        # -------------------------
        scores_global = torch.einsum(
            "bthd,btkd->bhtk",
            q,
            K_selected,
        ) / math.sqrt(self.head_dim)                 # [B,H,T,K]

        # Canonical default: indexer chooses selected blocks only. It does not
        # bias the core attention logits unless explicitly enabled.
        if self.use_indexer_score_bias:
            scores_global = scores_global + topk_scores[:, None, :, :].to(
                dtype=scores_global.dtype
            )

        global_allowed = topk_mask[:, None, :, :].expand(B, self.n_heads, T, K_eff)
        # [B,H,T,K]

        scores_parts.append(scores_global)
        allowed_parts.append(global_allowed)

        # -------------------------
        # Local exact scores
        # -------------------------
        scores_local = torch.einsum(
            "bthd,bsd->bhts",
            q,
            K_local,
        ) / math.sqrt(self.head_dim)                 # [B,H,T,T]

        local_allowed = self._build_local_allowed_mask(
            T=T,
            device=x.device,
        )
        local_allowed = local_allowed[None, None, :, :]  # [1,1,T,T]

        if attention_mask is not None:
            local_key_allowed = attention_mask[:, None, None, :].to(
                device=x.device,
                dtype=torch.bool,
            )
            local_allowed = local_allowed & local_key_allowed

        local_allowed = local_allowed.expand(B, self.n_heads, T, T)

        scores_parts.append(scores_local)
        allowed_parts.append(local_allowed)

        # ----------------------------------------------------
        # Combined sink + sparse global + local softmax
        # ----------------------------------------------------
        scores = torch.cat(scores_parts, dim=-1)          # [B,H,T,N]
        allowed = torch.cat(allowed_parts, dim=-1)        # [B,H,T,N]

        weights = safe_masked_softmax(
            scores=scores,
            allowed_mask=allowed,
            dim=-1,
        )

        weights = self.attention_dropout(weights)

        # ----------------------------------------------------
        # Split attention weights
        # ----------------------------------------------------
        offset = 0

        if self.use_attention_sink:
            weights_sink = weights[..., offset:offset + 1]  # [B,H,T,1]
            offset += 1
        else:
            weights_sink = None

        weights_global = weights[..., offset:offset + K_eff]  # [B,H,T,K]
        offset += K_eff

        weights_local = weights[..., offset:]                 # [B,H,T,T]

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------
        context = torch.zeros(
            B,
            self.n_heads,
            T,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        if self.use_attention_sink:
            V_sink = self.sink_v.expand(B, -1, -1)  # [B,1,Dh]
            context_sink = torch.einsum(
                "bhts,bsd->bhtd",
                weights_sink,
                V_sink,
            )
            context = context + context_sink

        context_global = torch.einsum(
            "bhtk,btkd->bhtd",
            weights_global,
            V_selected,
        )

        context_local = torch.einsum(
            "bhts,bsd->bhtd",
            weights_local,
            V_local,
        )

        context = context + context_global + context_local  # [B,H,T,Dh]

        # ----------------------------------------------------
        # Merge heads + output projection
        # ----------------------------------------------------
        context = context.transpose(1, 2).contiguous()  # [B,T,H,Dh]

        if self.use_grouped_output_projection:
            out = self.out_proj(context)               # [B,T,D]
        else:
            context = context.view(B, T, self.inner_dim)
            out = self.out_proj(context)               # [B,T,D]

        out = self.residual_dropout(out)

        if need_weights:
            aux = {
                "global_attn_weights": weights_global,
                "local_attn_weights": weights_local,
                "topk_indices": topk_indices,
                "topk_scores": topk_scores,
                "topk_mask": topk_mask,
                "compressed_valid_mask": comp_valid_mask,
                "compressed_position_ids": comp_position_ids,
                "index_scores": index_scores,
            }

            if self.use_attention_sink:
                aux["sink_attn_weights"] = weights_sink

            return out, aux

        return out