# ============================================================
# Loss component metrics
# ============================================================

from training.deepseek_modules_metrics_utils import * 

def compute_deepseek_loss_metrics(
    outputs: Any,
    model: Optional[nn.Module] = None,
    prefix: str = "train",
) -> Dict[str, float]:
    """
    Extract main loss components from DeepSeekV4LM outputs.

    Expected / supported keys:
        loss
        lm_loss
        mtp_loss
        moe_aux_loss
        raw_mtp_loss
        weighted_mtp_loss
        balance_loss
        sequence_balance_loss

    Notes:
        - perplexity should use lm_loss, not total loss.
        - if lm_loss is missing, we do not compute perplexity from total loss.
    """
    metrics: Dict[str, float] = {}

    # Main scalar losses.
    for key in [
        "loss",
        "lm_loss",
        "mtp_loss",
        "moe_aux_loss",
        "raw_mtp_loss",
        "weighted_mtp_loss",
        "balance_loss",
        "sequence_balance_loss",
    ]:
        value = get_from_output(outputs, key, None)

        # If not top-level, search nested aux dicts.
        if value is None:
            nested = collect_values_by_key(outputs, key)
            value = mean_of_scalar_values(nested)

        maybe_add_metric(metrics, f"{prefix}/{key}", value)

    # Perplexity from lm_loss only.
    lm_loss = metrics.get(f"{prefix}/lm_loss", None)

    if lm_loss is not None and math.isfinite(lm_loss):
        metrics[f"{prefix}/perplexity_from_lm_loss"] = float(math.exp(min(lm_loss, 50.0)))

    # Useful sanity: total - components.
    total = metrics.get(f"{prefix}/loss", None)
    lm = metrics.get(f"{prefix}/lm_loss", 0.0)
    mtp = metrics.get(f"{prefix}/mtp_loss", 0.0)
    moe = metrics.get(f"{prefix}/moe_aux_loss", 0.0)

    if total is not None:
        metrics[f"{prefix}/loss_minus_components"] = float(total - lm - mtp - moe)

    return metrics