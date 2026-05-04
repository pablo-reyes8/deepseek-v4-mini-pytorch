# ============================================================
# MULTI-TOKEN PREDICTION HEAD
# ============================================================

from src.mtp_components import * 

class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction auxiliary heads.

    Main LM head predicts:
        x_{t+1}

    MTP heads predict:
        head 0 -> x_{t+2}
        head 1 -> x_{t+3}
        ...
        head k -> x_{t+k+2}

    Input:
        hidden_states: [B,T,d_model]

    Output dict:
        {
            "mtp_logits": [B,mtp_depth,T,vocab_size],
            "mtp_loss": weighted_mtp_loss or None,
            "aux": {...}
        }
    """

    def __init__(self, config: MTPConfig):
        super().__init__()

        config.validate()

        self.config = config

        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.mtp_depth = config.mtp_depth
        self.pad_token_id = config.pad_token_id
        self.ignore_index = config.ignore_index
        self.mtp_loss_weight = config.mtp_loss_weight
        self.tie_with_lm_head = config.tie_with_lm_head
        self.validate_label_range = config.validate_label_range

        if config.depth_loss_weights is None:
            depth_weights = torch.ones(config.mtp_depth, dtype=torch.float32)
        else:
            depth_weights = torch.tensor(config.depth_loss_weights, dtype=torch.float32)

        depth_weights = depth_weights / depth_weights.sum()
        self.register_buffer("depth_loss_weights", depth_weights, persistent=False)

        if config.use_mtp_transform:
            self.transforms = nn.ModuleList(
                [
                    MTPTransform(
                        d_model=config.d_model,
                        hidden_dim=config.hidden_dim,
                        activation=config.activation,
                        dropout=config.dropout,
                        use_bias=config.use_bias,
                        init_std=config.init_std,
                    )
                    for _ in range(config.mtp_depth)
                ]
            )
        else:
            self.transforms = nn.ModuleList(
                [nn.Identity() for _ in range(config.mtp_depth)]
            )

        self.heads = nn.ModuleList(
            [
                nn.Linear(
                    config.d_model,
                    config.vocab_size,
                    bias=False,
                )
                for _ in range(config.mtp_depth)
            ]
        )

        self.reset_head_parameters()

    def reset_head_parameters(self) -> None:
        for head in self.heads:
            nn.init.normal_(
                head.weight,
                mean=0.0,
                std=self.config.init_std,
            )

    def tie_weights(self, lm_head_weight: nn.Parameter) -> None:
        """
        Tie every MTP head weight to the main LM head weight.

        Args:
            lm_head_weight:
                nn.Parameter with shape [vocab_size, d_model]

        This shares the same Parameter object instead of copying values.
        """
        if lm_head_weight.shape != (self.vocab_size, self.d_model):
            raise ValueError(
                "lm_head_weight must have shape "
                f"{(self.vocab_size, self.d_model)}, got {tuple(lm_head_weight.shape)}"
            )

        for head in self.heads:
            head.weight = lm_head_weight

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> Tuple[int, int]:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must have shape [B,T,d_model], got {tuple(hidden_states.shape)}"
            )

        B, T, D = hidden_states.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected hidden_states.shape[-1] == d_model={self.d_model}, got {D}"
            )

        return B, T

    def _validate_mtp_labels(
        self,
        mtp_labels: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> None:
        expected_shape = (batch_size, self.mtp_depth, seq_len)

        if mtp_labels.shape != expected_shape:
            raise ValueError(
                "mtp_labels must have shape [B,mtp_depth,T]. "
                f"Expected {expected_shape}, got {tuple(mtp_labels.shape)}"
            )

        if torch.is_floating_point(mtp_labels):
            raise TypeError("mtp_labels must be integer token ids, not floating point.")

        if not self.validate_label_range:
            return

        labels = mtp_labels.long()

        valid_ignore = labels == int(self.ignore_index)
        valid_token = (labels >= 0) & (labels < self.vocab_size)

        valid = valid_ignore | valid_token

        if not valid.all():
            bad = labels[~valid]
            bad_min = bad.min().item()
            bad_max = bad.max().item()
            raise ValueError(
                "mtp_labels contain values outside [0, vocab_size) and not equal "
                f"to ignore_index={self.ignore_index}. "
                f"Bad value range: [{bad_min}, {bad_max}]"
            )

    def _compute_loss(
        self,
        mtp_logits: torch.Tensor,
        mtp_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mtp_logits:
                [B,K,T,V]

            mtp_labels:
                [B,K,T]

        Returns:
            weighted_mtp_loss:
                scalar

            raw_mtp_loss:
                scalar

            mtp_loss_per_depth:
                [K]
        """
        B, K, T, V = mtp_logits.shape

        labels = mtp_labels.long()

        losses = []

        for k in range(K):
            logits_k = mtp_logits[:, k, :, :]
            labels_k = labels[:, k, :]

            loss_k = F.cross_entropy(
                logits_k.reshape(B * T, V),
                labels_k.reshape(B * T),
                ignore_index=int(self.ignore_index),
            )

            losses.append(loss_k)

        mtp_loss_per_depth = torch.stack(losses, dim=0)  # [K]

        depth_weights = self.depth_loss_weights.to(
            device=mtp_loss_per_depth.device,
            dtype=mtp_loss_per_depth.dtype,
        )

        raw_mtp_loss = (depth_weights * mtp_loss_per_depth).sum()
        weighted_mtp_loss = self.mtp_loss_weight * raw_mtp_loss

        return weighted_mtp_loss, raw_mtp_loss, mtp_loss_per_depth

    def forward(
        self,
        hidden_states: torch.Tensor,
        mtp_labels: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:

        B, T = self._validate_hidden_states(hidden_states)

        if mtp_labels is not None:
            self._validate_mtp_labels(
                mtp_labels=mtp_labels,
                batch_size=B,
                seq_len=T,
            )

        logits_list = []

        for k in range(self.mtp_depth):
            h_k = self.transforms[k](hidden_states)
            logits_k = self.heads[k](h_k)
            logits_list.append(logits_k)

        mtp_logits = torch.stack(logits_list, dim=1)
        # [B,mtp_depth,T,V]

        mtp_loss = None
        aux: Dict[str, Any] = {}

        if mtp_labels is not None:
            weighted_loss, raw_loss, loss_per_depth = self._compute_loss(
                mtp_logits=mtp_logits,
                mtp_labels=mtp_labels,
            )

            mtp_loss = weighted_loss

            aux["raw_mtp_loss"] = raw_loss
            aux["weighted_mtp_loss"] = weighted_loss
            aux["mtp_loss_per_depth"] = loss_per_depth
            aux["depth_loss_weights"] = self.depth_loss_weights.to(
                device=loss_per_depth.device,
                dtype=loss_per_depth.dtype,
            )

        elif return_aux:
            aux["raw_mtp_loss"] = None
            aux["weighted_mtp_loss"] = None
            aux["mtp_loss_per_depth"] = None
            aux["depth_loss_weights"] = self.depth_loss_weights

        return {
            "mtp_logits": mtp_logits,
            "mtp_loss": mtp_loss,
            "aux": aux if return_aux or mtp_labels is not None else {},
        }