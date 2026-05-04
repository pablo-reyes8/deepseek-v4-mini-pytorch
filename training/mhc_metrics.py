# ============================================================
# 4. mHC diagnostics
# ============================================================

from training.deepseek_modules_metrics_utils import * 

def _class_name_contains(module: nn.Module, text: str) -> bool:
    return text.lower() in module.__class__.__name__.lower()


def _collect_mhc_abc_dicts(outputs: Any) -> List[Dict[str, Any]]:
    """
    Collect nested aux dictionaries that contain A, B, C.
    """
    abc_dicts = []

    for d in _iter_nested_dicts(outputs):
        if all(k in d for k in ["A", "B", "C"]):
            abc_dicts.append(d)

    return abc_dicts


def _collect_mhc_alpha_metrics_from_model(
    model: Optional[nn.Module],
    prefix: str = "mhc",
) -> Dict[str, float]:
    """
    Collect alpha parameters from ManifoldHyperConnection modules.

    Supports possible names:
        alpha_A / alpha_pre
        alpha_B / alpha_res
        alpha_C / alpha_post

    Also logs any generic parameter containing 'alpha'.
    """
    metrics: Dict[str, float] = {}

    if model is None:
        return metrics

    raw_model = _unwrap_model(model)

    alpha_A_values = []
    alpha_B_values = []
    alpha_C_values = []
    alpha_all_values = []

    for module_name, module in raw_model.named_modules():
        is_mhc_like = (
            _class_name_contains(module, "ManifoldHyperConnection")
            or (
                hasattr(module, "compute_ABC")
                and hasattr(module, "pre_mix")
                and hasattr(module, "update")
            )
        )

        if not is_mhc_like:
            continue

        for pname, p in module.named_parameters(recurse=False):
            lower = pname.lower()

            if "alpha" not in lower:
                continue

            t = p.detach().float().cpu().reshape(-1)
            alpha_all_values.append(t)

            # Flexible mapping.
            if "alpha_a" in lower or "pre" in lower:
                alpha_A_values.append(t)
            elif "alpha_b" in lower or "res" in lower:
                alpha_B_values.append(t)
            elif "alpha_c" in lower or "post" in lower:
                alpha_C_values.append(t)

    def _add_alpha_stats(name: str, tensors: List[torch.Tensor]):
        if tensors:
            x = torch.cat(tensors, dim=0)
            metrics[f"{prefix}/{name}_mean"] = float(x.mean().item())
            metrics[f"{prefix}/{name}_std"] = (
                float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0
            )
            metrics[f"{prefix}/{name}_min"] = float(x.min().item())
            metrics[f"{prefix}/{name}_max"] = float(x.max().item())

    _add_alpha_stats("alpha_A", alpha_A_values)
    _add_alpha_stats("alpha_B", alpha_B_values)
    _add_alpha_stats("alpha_C", alpha_C_values)
    _add_alpha_stats("alpha_all", alpha_all_values)

    return metrics


def compute_mhc_diagnostics(
    outputs: Any,
    model: Optional[nn.Module] = None,
    prefix: str = "mhc",
) -> Dict[str, float]:
    """
    Compute mHC diagnostics.

    From aux A/B/C:
        B_row_sum_error
        B_column_sum_error
        B_min/max
        A mean/std
        C mean/std

    From model params:
        alpha_A / alpha_B / alpha_C
    """
    metrics: Dict[str, float] = {}

    abc_dicts = _collect_mhc_abc_dicts(outputs)

    A_values = []
    B_values = []
    C_values = []

    row_errors = []
    col_errors = []
    row_errors_max = []
    col_errors_max = []

    for d in abc_dicts:
        A = tensor_float(d.get("A", None))
        B = tensor_float(d.get("B", None))
        C = tensor_float(d.get("C", None))

        if A is not None:
            A_values.append(A.reshape(-1))

        if C is not None:
            C_values.append(C.reshape(-1))

        if B is not None:
            B_values.append(B.reshape(-1))

            # B shape can be [n_hc, n_hc] or [B,T,n_hc,n_hc].
            if B.dim() >= 2:
                row_sum = B.sum(dim=-1)
                col_sum = B.sum(dim=-2)

                row_err = (row_sum - 1.0).abs()
                col_err = (col_sum - 1.0).abs()

                row_errors.append(row_err.reshape(-1))
                col_errors.append(col_err.reshape(-1))
                row_errors_max.append(row_err.max().reshape(1))
                col_errors_max.append(col_err.max().reshape(1))

    if A_values:
        A_cat = torch.cat(A_values, dim=0)
        metrics.update(safe_stat_tensor(A_cat, f"{prefix}/A"))

    if C_values:
        C_cat = torch.cat(C_values, dim=0)
        metrics.update(safe_stat_tensor(C_cat, f"{prefix}/C"))

    if B_values:
        B_cat = torch.cat(B_values, dim=0)
        metrics.update(safe_stat_tensor(B_cat, f"{prefix}/B"))

    if row_errors:
        row_cat = torch.cat(row_errors, dim=0)
        metrics[f"{prefix}/B_row_sum_error_mean"] = float(row_cat.mean().item())
        metrics[f"{prefix}/B_row_sum_error_max"] = float(torch.cat(row_errors_max).max().item())

    if col_errors:
        col_cat = torch.cat(col_errors, dim=0)
        metrics[f"{prefix}/B_column_sum_error_mean"] = float(col_cat.mean().item())
        metrics[f"{prefix}/B_column_sum_error_max"] = float(torch.cat(col_errors_max).max().item())

    # Add alpha diagnostics from model.
    metrics.update(_collect_mhc_alpha_metrics_from_model(model, prefix=prefix))

    # If no mHC active, make it explicit but non-invasive.
    if not abc_dicts:
        if model is not None:
            raw_model = unwrap_model(model)
            use_mhc = bool(getattr(getattr(raw_model, "config", None), "use_mhc", False))
            metrics[f"{prefix}/active"] = float(use_mhc)
        else:
            metrics[f"{prefix}/active"] = 0.0
    else:
        metrics[f"{prefix}/active"] = 1.0
        metrics[f"{prefix}/num_abc_aux_dicts"] = float(len(abc_dicts))

    return metrics