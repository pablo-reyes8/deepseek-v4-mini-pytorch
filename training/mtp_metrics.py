# ============================================================
# MTP diagnostics
# ============================================================

from training.deepseek_modules_metrics_utils import * 

def compute_mtp_diagnostics(
    outputs: Any,
    model: Optional[nn.Module] = None,
    prefix: str = "mtp",) -> Dict[str, float]:
    """
    Compute MTP diagnostics.

    Looks for:
        raw_mtp_loss
        weighted_mtp_loss
        mtp_loss
        mtp_loss_per_depth
        depth_losses
        mtp_depth_losses

    If only mtp_loss exists, logs it as weighted_mtp_loss fallback.
    """
    metrics: Dict[str, float] = {}

    raw_keys = [
        "raw_mtp_loss",
        "mtp_raw_loss",
        "unweighted_mtp_loss",
    ]

    weighted_keys = [
        "weighted_mtp_loss",
        "mtp_weighted_loss",
        "mtp_loss",
    ]

    per_depth_keys = [
        "mtp_loss_per_depth",
        "mtp_losses_per_depth",
        "mtp_depth_losses",
        "depth_losses",
    ]

    raw_values = collect_values_by_any_key(outputs, raw_keys)
    weighted_values = collect_values_by_any_key(outputs, weighted_keys)
    per_depth_values =collect_values_by_any_key(outputs, per_depth_keys)

    raw_mean = mean_of_scalar_values(raw_values)
    weighted_mean = mean_of_scalar_values(weighted_values)

    if raw_mean is not None:
        metrics[f"{prefix}/raw_mtp_loss"] = raw_mean

    if weighted_mean is not None:
        metrics[f"{prefix}/weighted_mtp_loss"] = weighted_mean

    # Fallback derivation if raw missing but weighted exists and config has mtp_loss_weight.
    if raw_mean is None and weighted_mean is not None and model is not None:
        raw_model = unwrap_model(model)
        cfg = getattr(raw_model, "config", None)
        weight = getattr(cfg, "mtp_loss_weight", None) if cfg is not None else None

        if weight is not None and float(weight) > 0:
            metrics[f"{prefix}/raw_mtp_loss_derived"] = float(weighted_mean / float(weight))

    # Loss weight.
    if model is not None:
        raw_model = unwrap_model(model)
        cfg = getattr(raw_model, "config", None)

        if cfg is not None and hasattr(cfg, "mtp_loss_weight"):
            metrics[f"{prefix}/loss_weight"] = float(cfg.mtp_loss_weight)

        if cfg is not None and hasattr(cfg, "mtp_depth"):
            metrics[f"{prefix}/depth"] = float(cfg.mtp_depth)

    # Per-depth losses.
    per_depth_tensors = []

    for v in per_depth_values:
        t = tensor_float(v)
        if t is not None:
            per_depth_tensors.append(t.reshape(-1))

    if per_depth_tensors:
        # Stack with possible different sources/layers. We average by depth index when possible.
        max_depth = max(t.numel() for t in per_depth_tensors)

        padded = []
        masks = []

        for t in per_depth_tensors:
            pad_len = max_depth - t.numel()
            if pad_len > 0:
                padded.append(torch.cat([t, torch.zeros(pad_len)], dim=0))
                masks.append(torch.cat([torch.ones_like(t), torch.zeros(pad_len)], dim=0))
            else:
                padded.append(t)
                masks.append(torch.ones_like(t))

        values = torch.stack(padded, dim=0)
        mask = torch.stack(masks, dim=0)

        depth_mean = (values * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)

        for i, value in enumerate(depth_mean):
            metrics[f"{prefix}/loss_depth_{i + 1}"] = float(value.item())

        metrics[f"{prefix}/loss_per_depth_mean"] = float(depth_mean.mean().item())
        metrics[f"{prefix}/loss_per_depth_std"] = (
            float(depth_mean.std(unbiased=False).item())
            if depth_mean.numel() > 1 else 0.0
        )

    return metrics