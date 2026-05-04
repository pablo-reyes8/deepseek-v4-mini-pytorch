# ============================================================
# Combined DeepSeek module metrics
# ============================================================
import math

from training.loss_metrics import * 
from training.mhc_metrics import * 
from training.mtp_metrics import * 
from training.moe_metrics import * 


def compute_deepseek_module_metrics(
    outputs: Any,
    model: Optional[nn.Module] = None,
    include_loss: bool = True,
    include_moe: bool = True,
    include_mtp: bool = True,
    include_mhc: bool = True,
    prefix: str = "train",
) -> Dict[str, float]:
    """
    Combined module-level diagnostics for DeepSeekV4LM training.

    This is intended to be called inside train_step after forward.

    Example:
        outputs = model(..., return_aux=True, need_weights=False)
        metrics = compute_deepseek_module_metrics(outputs, model=model)
    """
    metrics: Dict[str, float] = {}

    if include_loss:
        metrics.update(
            compute_deepseek_loss_metrics(
                outputs=outputs,
                model=model,
                prefix=prefix,
            )
        )

    if include_moe:
        moe_metrics = compute_moe_diagnostics(
            outputs=outputs,
            model=model,
            prefix=f"{prefix}/moe",
        )
        metrics.update(moe_metrics)

    if include_mtp:
        mtp_metrics = compute_mtp_diagnostics(
            outputs=outputs,
            model=model,
            prefix=f"{prefix}/mtp",
        )
        metrics.update(mtp_metrics)

    if include_mhc:
        mhc_metrics = compute_mhc_diagnostics(
            outputs=outputs,
            model=model,
            prefix=f"{prefix}/mhc",
        )
        metrics.update(mhc_metrics)

    return metrics

# ============================================================
# Pretty formatter for DeepSeek module metrics
# ============================================================

def _fmt_metric_value(value, precision: int = 4) -> str:
    if value is None:
        return "n/a"

    if isinstance(value, bool):
        return str(value)

    try:
        value = float(value)
    except Exception:
        return str(value)

    if not math.isfinite(value):
        return str(value)

    abs_v = abs(value)

    if abs_v == 0:
        return "0"

    if abs_v >= 1e4 or abs_v < 1e-3:
        return f"{value:.{precision}e}"

    return f"{value:.{precision}f}"


def _print_metric_block(
    title: str,
    metrics: dict,
    rows: list[tuple[str, str]],
    precision: int = 4,
    indent: int = 2,
) -> None:
    """
    rows:
        [
            (metric_key, reference_note),
            ...
        ]
    """
    available = [(k, note) for k, note in rows if k in metrics]

    if not available:
        return

    pad = " " * indent
    print(f"\n{title}")
    print("-" * len(title))

    names = [k.split("/")[-1] for k, _ in available]
    max_name_len = max(len(name) for name in names)
    max_value_len = max(len(_fmt_metric_value(metrics[k], precision)) for k, _ in available)

    for key, note in available:
        short_name = key.split("/")[-1]
        value_str = _fmt_metric_value(metrics[key], precision=precision)

        if note:
            print(f"{pad}{short_name:<{max_name_len}} : {value_str:<{max_value_len}}   # {note}")
        else:
            print(f"{pad}{short_name:<{max_name_len}} : {value_str}")


def _router_entropy_reference(num_experts: int | None) -> str:
    if num_experts is None or num_experts <= 1:
        return "higher = more uniform routing"
    return f"max ≈ log(E)={math.log(num_experts):.3f}; very low => routing collapse"


def _expert_fraction_reference(num_experts: int | None) -> str:
    if num_experts is None or num_experts <= 0:
        return "watch min/max; extreme imbalance is bad"
    return f"ideal mean ≈ 1/E={1.0 / num_experts:.3f}; min near 0 => dead expert"


def _active_experts_reference(num_experts: int | None, n_layers: int | None) -> str:
    if num_experts is None or n_layers is None:
        return "should stay high; falling means dead experts"
    return f"max = n_layers*E = {n_layers * num_experts}; lower => unused experts"


def print_deepseek_module_metrics(
    metrics: dict,
    prefix: str = "train",
    precision: int = 4,
    title: str | None = None,
    num_experts: int | None = None,
    top_k_experts: int | None = None,
    n_layers: int | None = None,
) -> None:
    """
    Pretty block-style printer for DeepSeek-V4 module diagnostics.

    Ordered by importance for training monitoring.

    Example:
        print_deepseek_module_metrics(
            metrics,
            prefix="train",
            title="DeepSeek-V4 module diagnostics",
            num_experts=model.config.num_experts,
            top_k_experts=model.config.top_k_experts,
            n_layers=model.config.n_layers,
        )
    """

    if title is not None:
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)

    # ========================================================
    # 1. Core convergence metrics
    # ========================================================

    loss_rows = [
        (
            f"{prefix}/loss",
            "lower is better; total optimized objective",
        ),
        (
            f"{prefix}/lm_loss",
            "lower is better; main next-token CE",
        ),
        (
            f"{prefix}/perplexity_from_lm_loss",
            "lower is better; exp(lm_loss)",
        ),
        (
            f"{prefix}/mtp_loss",
            "auxiliary; should not dominate total loss",
        ),
        (
            f"{prefix}/moe_aux_loss",
            "should be small; high values imply routing imbalance penalty",
        ),
        (
            f"{prefix}/loss_minus_components",
            "should be ≈ 0; nonzero means loss accounting bug",
        ),
        (
            f"{prefix}/raw_mtp_loss",
            "unweighted MTP CE; compare with lm_loss",
        ),
        (
            f"{prefix}/weighted_mtp_loss",
            "raw_mtp_loss * mtp_loss_weight",
        ),
    ]

    # ========================================================
    # 2. MoE health: most important after loss
    # ========================================================

    moe_rows_critical = [
        (
            f"{prefix}/moe/router_entropy_mean",
            _router_entropy_reference(num_experts),
        ),
        (
            f"{prefix}/moe/expert_fraction_min",
            "too close to 0 => dead/underused expert",
        ),
        (
            f"{prefix}/moe/expert_fraction_max",
            "too high => expert collapse / overload",
        ),
        (
            f"{prefix}/moe/expert_fraction_mean",
            _expert_fraction_reference(num_experts),
        ),
        (
            f"{prefix}/moe/dead_experts_across_layers",
            "should be 0 after warmup",
        ),
        (
            f"{prefix}/moe/active_experts_across_layers",
            _active_experts_reference(num_experts, n_layers),
        ),
        (
            f"{prefix}/moe/n_unique_selected_experts",
            "should approach E within a batch",
        ),
        (
            f"{prefix}/moe/n_experts_used_per_batch_mean",
            "higher is better; ideally all experts used per layer/batch",
        ),
        (
            f"{prefix}/moe/topk_weights_mean",
            "with normalized top-k, ideal ≈ 1/top_k",
        ),
        (
            f"{prefix}/moe/topk_weights_std",
            "very high => routing confidence collapse to few experts",
        ),
    ]

    moe_rows_secondary = [
        (
            f"{prefix}/moe/router_entropy_std",
            "large std => unstable routing across tokens/layers",
        ),
        (
            f"{prefix}/moe/expert_fraction_std",
            "lower is more balanced; rising can signal specialization or collapse",
        ),
        (
            f"{prefix}/moe/sequence_expert_fraction_min",
            "per-sequence min; near 0 means sequence-level imbalance",
        ),
        (
            f"{prefix}/moe/sequence_expert_fraction_max",
            "per-sequence max; high means sequence overuses one expert",
        ),
        (
            f"{prefix}/moe/sequence_expert_fraction_mean",
            "should be near 1/E",
        ),
        (
            f"{prefix}/moe/sequence_expert_fraction_std",
            "sequence-level load dispersion",
        ),
        (
            f"{prefix}/moe/topk_weights_min",
            "",
        ),
        (
            f"{prefix}/moe/topk_weights_max",
            "",
        ),
        (
            f"{prefix}/moe/n_experts_used_per_batch_min",
            "",
        ),
        (
            f"{prefix}/moe/n_experts_used_per_batch_max",
            "",
        ),
        (
            f"{prefix}/moe/moe_aux_loss_mean",
            "small is expected",
        ),
        (
            f"{prefix}/moe/balance_loss_mean",
            "small is expected",
        ),
        (
            f"{prefix}/moe/sequence_balance_loss_mean",
            "small is expected",
        ),
    ]

    # ========================================================
    # 3. MTP health
    # ========================================================

    mtp_rows = [
        (
            f"{prefix}/mtp/weighted_mtp_loss",
            "contributes to total loss; should not dominate lm_loss",
        ),
        (
            f"{prefix}/mtp/raw_mtp_loss",
            "compare to lm_loss; should fall with training",
        ),
        (
            f"{prefix}/mtp/loss_weight",
            "weighted_mtp_loss = raw * this weight",
        ),
        (
            f"{prefix}/mtp/depth",
            "number of future-token objectives",
        ),
        (
            f"{prefix}/mtp/loss_depth_1",
            "near-future target; should improve fastest",
        ),
        (
            f"{prefix}/mtp/loss_depth_2",
            "farther target; usually harder",
        ),
        (
            f"{prefix}/mtp/loss_depth_3",
            "",
        ),
        (
            f"{prefix}/mtp/loss_depth_4",
            "",
        ),
        (
            f"{prefix}/mtp/loss_per_depth_mean",
            "",
        ),
        (
            f"{prefix}/mtp/loss_per_depth_std",
            "large std => uneven MTP depth learning",
        ),
        (
            f"{prefix}/mtp/raw_mtp_loss_derived",
            "",
        ),
    ]

    # ========================================================
    # 4. mHC numerical stability
    # ========================================================

    mhc_rows_critical = [
        (
            f"{prefix}/mhc/active",
            "1 if mHC active",
        ),
        (
            f"{prefix}/mhc/B_row_sum_error_mean",
            "should be near 0; high => Sinkhorn/manifold failure",
        ),
        (
            f"{prefix}/mhc/B_column_sum_error_mean",
            "should be near 0; high => not doubly stochastic",
        ),
        (
            f"{prefix}/mhc/B_row_sum_error_max",
            "large spikes can signal numerical instability",
        ),
        (
            f"{prefix}/mhc/B_column_sum_error_max",
            "large spikes can signal numerical instability",
        ),
        (
            f"{prefix}/mhc/alpha_A_mean",
            "small initially; rapid growth may destabilize pre-mixing",
        ),
        (
            f"{prefix}/mhc/alpha_B_mean",
            "small initially; rapid growth may destabilize residual mixing",
        ),
        (
            f"{prefix}/mhc/alpha_C_mean",
            "small initially; rapid growth may destabilize post-mixing",
        ),
    ]

    mhc_rows_secondary = [
        (
            f"{prefix}/mhc/num_abc_aux_dicts",
            "",
        ),
        (
            f"{prefix}/mhc/A_mean",
            "A in [0,1] after sigmoid; monitors pre-mixing scale",
        ),
        (
            f"{prefix}/mhc/A_std",
            "",
        ),
        (
            f"{prefix}/mhc/A_min",
            "",
        ),
        (
            f"{prefix}/mhc/A_max",
            "",
        ),
        (
            f"{prefix}/mhc/B_mean",
            "for n_hc=4, expected mean ≈ 0.25",
        ),
        (
            f"{prefix}/mhc/B_std",
            "",
        ),
        (
            f"{prefix}/mhc/B_min",
            "",
        ),
        (
            f"{prefix}/mhc/B_max",
            "",
        ),
        (
            f"{prefix}/mhc/C_mean",
            "C constrained; monitors post-block injection scale",
        ),
        (
            f"{prefix}/mhc/C_std",
            "",
        ),
        (
            f"{prefix}/mhc/C_min",
            "",
        ),
        (
            f"{prefix}/mhc/C_max",
            "",
        ),
        (
            f"{prefix}/mhc/alpha_A_std",
            "",
        ),
        (
            f"{prefix}/mhc/alpha_B_std",
            "",
        ),
        (
            f"{prefix}/mhc/alpha_C_std",
            "",
        ),
        (
            f"{prefix}/mhc/alpha_all_mean",
            "",
        ),
        (
            f"{prefix}/mhc/alpha_all_std",
            "",
        ),
    ]

    _print_metric_block(
        title="1. Loss / convergence metrics",
        metrics=metrics,
        rows=loss_rows,
        precision=precision,
    )

    _print_metric_block(
        title="2. MoE routing health — critical",
        metrics=metrics,
        rows=moe_rows_critical,
        precision=precision,
    )

    _print_metric_block(
        title="3. MTP auxiliary objective",
        metrics=metrics,
        rows=mtp_rows,
        precision=precision,
    )

    _print_metric_block(
        title="4. mHC numerical stability — critical",
        metrics=metrics,
        rows=mhc_rows_critical,
        precision=precision,
    )

    _print_metric_block(
        title="5. MoE detailed diagnostics",
        metrics=metrics,
        rows=moe_rows_secondary,
        precision=precision,
    )

    _print_metric_block(
        title="6. mHC detailed diagnostics",
        metrics=metrics,
        rows=mhc_rows_secondary,
        precision=precision,
    )

    if title is not None:
        print("\n" + "=" * 96)