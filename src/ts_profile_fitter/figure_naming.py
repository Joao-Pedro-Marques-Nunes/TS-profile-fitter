"""Consistent names for TS profile-fitting figures."""


def _format_time(value):
    """Format a time for a filename without using punctuation-heavy decimals."""
    return f"{float(value):.3f}".replace(".", "p")


def _format_spline_strategy(strategy):
    """Normalize a spline knot strategy for compact, stable filenames."""
    strategy = str(strategy).strip().lower().replace("-", "_").replace(" ", "_")
    return {"global_quantile": "global_qntl"}.get(strategy, strategy)


def make_fig_name(
    shot_id,
    equilibrium,
    n_knots,
    interval,
    plot_kind,
    spline_strategy=None,
):
    """Build a stable, human-readable figure filename stem.

    The interval bounds are included because interval indices are only unique
    within a shot and can change when an interval table is regenerated.
    """
    if interval is None:
        raise ValueError("interval is required to build a figure name.")
    if hasattr(interval, "get"):
        t_begin = interval.get("t_begin")
        t_end = interval.get("t_end")
    else:
        t_begin, t_end = interval
    if t_begin is None or t_end is None:
        raise ValueError("interval must define t_begin and t_end.")

    equilibrium_name = str(equilibrium or "unknown").replace(" ", "-")
    strategy_suffix = ''
    if spline_strategy is not None:
        strategy_suffix = f'_{_format_spline_strategy(spline_strategy)}'
    return (
        f"shot{int(shot_id)}_{equilibrium_name}_k{int(n_knots):02d}_"
        f"t{_format_time(t_begin)}-{_format_time(t_end)}_{plot_kind}{strategy_suffix}"
    )
