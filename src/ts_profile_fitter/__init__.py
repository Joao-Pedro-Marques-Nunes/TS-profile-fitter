"""TS profile fitter package."""

from .loading import (
    load_shot_list,
    load_intervals,
    load_ts_data,
    load_physics_inputs,
    load_elm_times,
    load_power_signals,
)
from .preprocess import preprocess_ts_data, map_to_r_midplane, filter_elm_cycles
from .fitting import (
    fit_mtanh_profile,
    fit_modified_tanh_profile,
    fit_spline_profile,
    fit_ipfn_spline_ne_Te_TS,
    fit_mbsplt_ne_Te_TS,
    compute_channel_knots,
)
from .derived_quantities import (
    compute_separatrix,
    compute_profile_quantities,
    compute_equilibrium_separatrix,
    compute_physics_separatrix,
)
from .connection_length import get_connection_length
from .figure_naming import make_fig_name
from .fit_metrics import compute_fit_metrics
from .separatrix import make_profile, estimate_separatrix
from .pipeline import run_interval, run_shot, run_database
from .plotting import (
    plot_profile_fit,
    plot_filtered_profile_with_outliers,
    plot_spline_profile_fit,
    plot_profile_comparison,
    plot_profile_comparison_full,
    plot_profile_comparison_zoomed,
    plot_profile_summary,
    plot_profile_failure_diagnostic,
    plot_power_signals,
)

__all__ = [
    "load_shot_list",
    "load_intervals",
    "load_ts_data",
    "load_physics_inputs",
    "load_elm_times",
    "load_power_signals",
    "preprocess_ts_data",
    "map_to_r_midplane",
    "filter_elm_cycles",
    "fit_mtanh_profile",
    "fit_modified_tanh_profile",
    "fit_spline_profile",
    "fit_ipfn_spline_ne_Te_TS",
    "fit_mbsplt_ne_Te_TS",
    "compute_channel_knots",
    "compute_separatrix",
    "compute_profile_quantities",
    "compute_equilibrium_separatrix",
    "compute_physics_separatrix",
    "get_connection_length",
    "make_fig_name",
    "make_profile",
    "estimate_separatrix",
    "run_shot",
    "run_database",
    "run_interval",
    "plot_profile_fit",
    "plot_filtered_profile_with_outliers",
    "plot_spline_profile_fit",
    "plot_profile_comparison",
    "plot_profile_comparison_full",
    "plot_profile_comparison_zoomed",
    "plot_profile_summary",
    "plot_profile_failure_diagnostic",
    "plot_power_signals",
]
