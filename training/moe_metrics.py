# ============================================================
#  MoE diagnostics
# ============================================================
from training.deepseek_modules_metrics_utils import * 


def _compute_entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = probs.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)


def compute_moe_diagnostics(
    outputs: Any,
    model: Optional[nn.Module] = None,
    prefix: str = "moe",
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute MoE diagnostics from returned aux dictionaries.

    Looks for keys:
        expert_fraction
        sequence_expert_fraction
        router_entropy
        router_probs / router_scores
        topk_weights
        topk_indices
        balance_loss
        sequence_balance_loss
        moe_aux_loss

    Returns:
        moe/router_entropy_mean
        moe/expert_fraction_min/max/std
        moe/sequence_expert_fraction_min/max/std
        moe/topk_weights_mean/std/min/max
        moe/n_experts_used_per_batch
        moe/dead_experts
        moe/balance_loss_mean
        moe/sequence_balance_loss_mean
    """
    metrics: Dict[str, float] = {}

    # -----------------------------
    # Expert fraction
    # -----------------------------
    expert_fraction_values = collect_values_by_key(outputs, "expert_fraction")
    expert_fraction = cat_flat_tensors(expert_fraction_values)

    if expert_fraction is not None:
        metrics.update(safe_stat_tensor(expert_fraction, f"{prefix}/expert_fraction"))

        # Counts across all collected MoE layers.
        # Example: n_layers=2 and num_experts=4 => max possible = 8.
        metrics[f"{prefix}/dead_experts_across_layers"] = float(
            (expert_fraction <= 0).sum().item()
        )
        metrics[f"{prefix}/active_experts_across_layers"] = float(
            (expert_fraction > 0).sum().item()
        )

    # -----------------------------
    # Sequence expert fraction
    # -----------------------------
    seq_expert_values = collect_values_by_key(outputs, "sequence_expert_fraction")

    seq_tensors = []
    for v in seq_expert_values:
        t = tensor_float(v)
        if t is not None:
            seq_tensors.append(t)

    if seq_tensors:
        seq_flat = torch.cat([t.reshape(-1) for t in seq_tensors], dim=0)
        metrics.update(safe_stat_tensor(seq_flat, f"{prefix}/sequence_expert_fraction"))

        # If shape is [..., E], count experts used per batch/sequence item.
        used_counts = []
        for t in seq_tensors:
            if t.dim() >= 1:
                used = (t > 0).sum(dim=-1).float()
                used_counts.append(used.reshape(-1))

        if used_counts:
            used_counts = torch.cat(used_counts, dim=0)
            metrics[f"{prefix}/n_experts_used_per_batch_mean"] = float(used_counts.mean().item())
            metrics[f"{prefix}/n_experts_used_per_batch_min"] = float(used_counts.min().item())
            metrics[f"{prefix}/n_experts_used_per_batch_max"] = float(used_counts.max().item())

    # Fallback: if no sequence fraction, use global expert fraction.
    elif expert_fraction is not None:
        metrics[f"{prefix}/n_experts_used_per_batch_mean"] = float((expert_fraction > 0).sum().item())

    # -----------------------------
    # Router entropy
    # -----------------------------
    router_entropy_values = collect_values_by_key(outputs, "router_entropy")
    router_entropy = cat_flat_tensors(router_entropy_values)

    if router_entropy is not None:
        metrics[f"{prefix}/router_entropy_mean"] = float(router_entropy.mean().item())
        metrics[f"{prefix}/router_entropy_std"] = (
            float(router_entropy.std(unbiased=False).item())
            if router_entropy.numel() > 1 else 0.0
        )
    else:
        # Fallback: compute from router_probs or router_scores if available.
        prob_values = collect_values_by_any_key(outputs, ["router_probs", "router_probabilities"])
        score_values = collect_values_by_key(outputs, "router_scores")

        entropy_chunks = []

        for v in prob_values:
            t = tensor_float(v)
            if t is not None and t.dim() >= 1:
                entropy_chunks.append(_compute_entropy_from_probs(t, eps=eps).reshape(-1))

        for v in score_values:
            t = tensor_float(v)
            if t is not None and t.dim() >= 1:
                probs = t / t.sum(dim=-1, keepdim=True).clamp_min(eps)
                entropy_chunks.append(_compute_entropy_from_probs(probs, eps=eps).reshape(-1))

        if entropy_chunks:
            ent = torch.cat(entropy_chunks, dim=0)
            metrics[f"{prefix}/router_entropy_mean"] = float(ent.mean().item())
            metrics[f"{prefix}/router_entropy_std"] = (
                float(ent.std(unbiased=False).item()) if ent.numel() > 1 else 0.0
            )

    # -----------------------------
    # Top-k weights
    # -----------------------------
    topk_weight_values = collect_values_by_key(outputs, "topk_weights")
    topk_weights = cat_flat_tensors(topk_weight_values)

    if topk_weights is not None:
        metrics.update(safe_stat_tensor(topk_weights, f"{prefix}/topk_weights"))

    # Optional top-k indices: useful for number of actually selected experts.
    topk_indices_values = collect_values_by_key(outputs, "topk_indices")
    topk_indices_flat = []

    for v in topk_indices_values:
        if torch.is_tensor(v) and v.numel() > 0:
            topk_indices_flat.append(v.detach().cpu().reshape(-1))

    if topk_indices_flat:
        idx = torch.cat(topk_indices_flat, dim=0)
        metrics[f"{prefix}/n_unique_selected_experts"] = float(torch.unique(idx).numel())

    # -----------------------------
    # Aux losses
    # -----------------------------
    for key in ["moe_aux_loss", "balance_loss", "sequence_balance_loss"]:
        values = collect_values_by_key(outputs, key)
        mean_value = mean_of_scalar_values(values)

        if mean_value is not None:
            metrics[f"{prefix}/{key}_mean"] = mean_value

    return metrics