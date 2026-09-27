import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from ipfnpytools.plot import plots as ipfn_plots
    from ipfnpytools.save_figure import save_figure as save_figure_ipfn
except Exception:  # pragma: no cover
    ipfn_plots = None
    save_figure_ipfn = None

from .fitting import modified_tanh


def _make_profile_title(title, nshot, interval):
    if title is not None:
        return title

    title_parts = []
    if nshot is not None:
        title_parts.append(f'Shot {nshot}')
    if interval is not None:
        if hasattr(interval, 'get'):
            t_begin = interval.get('t_begin')
            t_end = interval.get('t_end')
        else:
            t_begin, t_end = interval
        title_parts.append(f'[{float(t_begin):.3f}, {float(t_end):.3f}] s')
    return ', '.join(title_parts) if title_parts else None


def _as_1d_float_array(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    return arr.ravel()


def _build_time_color_values(time_vals, n_points):
    if time_vals is None:
        return None
    time_vals = _as_1d_float_array(time_vals)
    if time_vals.size == 0:
        return None
    if time_vals.size == n_points:
        return time_vals
    if time_vals.size == 1:
        return np.full(n_points, time_vals[0], dtype=float)
    if time_vals.size < n_points:
        return np.repeat(time_vals, int(np.ceil(n_points / time_vals.size)))[:n_points]
    return time_vals[:n_points]


def _add_interval_label(fig, interval):
    if hasattr(interval, 'get') and interval.get('label'):
        fig.text(0.98, 0.02, f"{interval['label']}", ha='right', va='bottom', fontsize=8)


def _save_final_figure(fig, fig_name, directory, save_fig):
    if not save_fig:
        return
    if save_figure_ipfn is None:
        raise ImportError('ipfnpytools.save_figure is not available; install ipfnpytools first.')
    save_figure_ipfn(fig, name=fig_name, directory=directory)


def _restore_grids(axes):
    for axis in np.asarray(axes, dtype=object).ravel():
        axis.grid(True)


def _finite_values(*arrays):
    values = []
    for array in arrays:
        if array is None:
            continue
        if isinstance(array, (list, tuple)):
            nested = _finite_values(*array)
            if nested.size:
                values.append(nested)
            continue
        finite = np.asarray(array, dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)
    return np.concatenate(values) if values else np.array([], dtype=float)


def _group_positions(values, tolerance=0.02):
    finite = np.sort(_finite_values(values))
    if finite.size == 0:
        return finite
    groups = [[finite[0]]]
    for value in finite[1:]:
        if value - groups[-1][0] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return np.array([np.median(group) for group in groups], dtype=float)


def _robust_limits(*arrays, include_zero=False, padding=0.08):
    finite = _finite_values(*arrays)
    if finite.size == 0:
        raise ValueError('No finite values available for automatic plot limits.')
    lower, upper = np.nanpercentile(finite, [1.0, 99.0])
    scale = upper - lower
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(lower), abs(upper), 1.0)
    lower -= padding * scale
    upper += padding * scale
    if include_zero:
        lower = min(lower, 0.0)
    return float(lower), float(upper)


def _automatic_zoom_start(x_values, separatrices):
    finite_x = _finite_values(x_values)
    finite_sep = _finite_values(separatrices)
    if finite_x.size == 0:
        raise ValueError('No finite radial coordinates available for automatic zoom.')
    if finite_sep.size:
        boundary = float(np.nanmedian(finite_sep))
    else:
        boundary = float(np.nanpercentile(finite_x, 75.0))
    spacings = np.diff(_group_positions(finite_x))
    spacings = spacings[np.isfinite(spacings) & (spacings > 0)]
    step = float(np.nanmedian(spacings)) if spacings.size else 0.01
    return max(float(np.nanmin(finite_x)), boundary - 2.0 * step)


def plot_profile_fit(
    x_core,
    x_edge,
    y_core,
    y_edge,
    y_err_core,
    y_err_edge,
    x_labels,
    y_labels,
    time_c=None,
    time_e=None,
    popt=None,
    x_core_out=None,
    x_edge_out=None,
    y_core_out=None,
    y_edge_out=None,
    y_err_core_out=None,
    y_err_edge_out=None,
    directory='/shares/users/work/joanune/Desktop/ipfn_examples/plots',
    close=False,
    save_fig=False,
    show=True,
    nshot=None,
    interval=None,
    fig_name=None,
    **plots_kw,
):
    """Plot core/edge profile data and an optional modified-tanh fit.

    This wrapper is designed for the TS-profile fitting workflow and uses the
    ipfnpytools plotting API when available. It supports both data-only and
    fit overlays, and can color-code points by time using the same style as the
    earlier mtanh plotting workflow.
    """

    if ipfn_plots is None:
        raise ImportError('ipfnpytools.plot.plots is not available; install ipfnpytools first.')

    x_core = np.asarray(x_core, dtype=float)
    x_edge = np.asarray(x_edge, dtype=float)
    y_core = np.asarray(y_core, dtype=float)
    y_edge = np.asarray(y_edge, dtype=float)
    y_err_core = np.asarray(y_err_core, dtype=float)
    y_err_edge = np.asarray(y_err_edge, dtype=float)

    if x_core.size == 0 and x_edge.size == 0:
        raise ValueError('No valid core/edge data available for plotting.')

    x_min = np.min(x_core) if x_core.size > 0 else np.min(x_edge)
    x_max = np.max(x_edge) if x_edge.size > 0 else np.max(x_core)

    if popt is None or (np.asarray(popt).size == 0) or np.isnan(np.asarray(popt)).all():
        y_fit = None
    else:
        x_fit = np.linspace(x_min, x_max, 500)
        y_fit = modified_tanh(x_fit, *np.asarray(popt, dtype=float))

    if time_c is None:
        time_c = np.array([], dtype=float)
    if time_e is None:
        time_e = np.array([], dtype=float)

    x_data = [x_core, x_core, x_edge, x_edge]
    y_data = [y_core, y_core, y_edge, y_edge]
    ey_data = [y_err_core, None, y_err_edge, None]
    z_data = [None, _build_time_color_values(time_c, x_core.size), None, _build_time_color_values(time_e, x_edge.size)]
    styles = ['', '', '', '']
    markers = [None, 'x', None, 'v']
    marker_sizes = [0, 14, 0, 14]
    time_values = np.concatenate([time_c, time_e])
    marker_plot_kw = {}
    if time_values.size:
        marker_plot_kw = {
            'vmin': np.nanmin(time_values),
            'vmax': np.nanmax(time_values),
            'cmap': 'plasma',
        }
    else:
        marker_sizes = [0, 6, 0, 6]
    plot_kw_list = [
        dict(ecolor='blue', alpha=0.2),
        marker_plot_kw.copy(),
        dict(ecolor='blue', alpha=0.2),
        marker_plot_kw.copy(),
    ]

    if y_fit is not None:
        x_data.append(x_fit)
        y_data.append(y_fit)
        ey_data.append(None)
        z_data.append(None)
        styles.append('-')
        markers.append(None)
        marker_sizes.append(None)
        plot_kw_list.append(dict(color='red'))

    if y_core_out is not None:
        x_data.append(x_core if x_core_out is None else _as_1d_float_array(x_core_out))
        y_data.append(np.asarray(y_core_out, dtype=float))
        ey_data.append(_as_1d_float_array(y_err_core_out))
        z_data.append(None)
        styles.append('')
        markers.append('o')
        marker_sizes.append(6)
        plot_kw_list.append(dict(color='orange', alpha=0.2))

    if y_edge_out is not None:
        x_data.append(x_edge if x_edge_out is None else _as_1d_float_array(x_edge_out))
        y_data.append(np.asarray(y_edge_out, dtype=float))
        ey_data.append(_as_1d_float_array(y_err_edge_out))
        z_data.append(None)
        styles.append('')
        markers.append('s')
        marker_sizes.append(6)
        plot_kw_list.append(dict(color='red', alpha=0.2))

    save_figure_kw = {'directory': directory} if directory is not None else None
    if fig_name is None:
        if nshot is None:
            fig_name = 'profile_fit'
        else:
            fig_name = f'{nshot}_profile_fit'

    fig, ax = ipfn_plots(
        x=x_data,
        y=y_data,
        z=z_data,
        ey=ey_data,
        x_labels=x_labels,
        y_labels=y_labels,
        boxes=0,
        write_legends=True,
        linestyles=styles,
        markers=markers,
        markersizes=marker_sizes,
        close=False,
        save_fig=False,
        show=False,
        title=_make_profile_title(plots_kw.pop('title', None), nshot, interval),
        fig_name=fig_name,
        save_figure_kw=save_figure_kw,
        plot_kw=plot_kw_list,
        **plots_kw,
    )
    _restore_grids(ax)
    _add_interval_label(fig, interval)
    _save_final_figure(fig, fig_name, directory, save_fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_spline_profile_fit(
    x_core,
    x_edge,
    y_core,
    y_edge,
    y_err_core,
    y_err_edge,
    ds_out,
    quantity,
    x_labels,
    y_labels,
    knot_pos=None,
    time_c=None,
    time_e=None,
    y_core_out=None,
    y_edge_out=None,
    y_err_core_out=None,
    y_err_edge_out=None,
    directory='/shares/users/work/joanune/Desktop/ipfn_examples/plots',
    close=False,
    save_fig=False,
    show=True,
    nshot=None,
    interval=None,
    fig_name=None,
    **plots_kw,
):
    """Plot core/edge profile data with a spline from ``ds_out``.

    ``ds_out`` is the joint output of ``fit_mbsplt_ne_Te_TS`` and must contain
    ``R_mid_spl`` plus either ``ne_spl`` or ``Te_spl``.
    """
    if ipfn_plots is None:
        raise ImportError('ipfnpytools.plot.plots is not available; install ipfnpytools first.')

    quantity_key = quantity.lower()
    spline_column = {'ne': 'ne_spl', 'te': 'Te_spl'}.get(quantity_key)
    if spline_column is None:
        raise ValueError("quantity must be 'Ne' or 'Te'.")

    x_fit = np.asarray(ds_out['R_mid_spl'], dtype=float)
    y_fit = np.asarray(ds_out[spline_column], dtype=float)

    x_core = np.asarray(x_core, dtype=float)
    x_edge = np.asarray(x_edge, dtype=float)
    y_core = np.asarray(y_core, dtype=float)
    y_edge = np.asarray(y_edge, dtype=float)
    y_err_core = np.asarray(y_err_core, dtype=float)
    y_err_edge = np.asarray(y_err_edge, dtype=float)

    if time_c is None:
        time_c = np.array([], dtype=float)
    if time_e is None:
        time_e = np.array([], dtype=float)

    x_data = [x_core, x_core, x_edge, x_edge, x_fit]
    y_data = [y_core, y_core, y_edge, y_edge, y_fit]
    ey_data = [y_err_core, None, y_err_edge, None, None]
    z_data = [
        None,
        _build_time_color_values(time_c, x_core.size),
        None,
        _build_time_color_values(time_e, x_edge.size),
        None,
    ]
    styles = ['', '', '', '', '-']
    markers = [None, 'x', None, 'v', None]
    marker_sizes = [0, 14, 0, 14, None]
    time_values = np.concatenate([time_c, time_e])
    plot_kw_list = [
        dict(ecolor='blue', alpha=0.2),
        dict(cmap='plasma'),
        dict(ecolor='blue', alpha=0.2),
        dict(cmap='plasma'),
        dict(color='green'),
    ]
    labels = [None, None, 'Core TS', 'Edge TS', 'Spline Fit']
    if time_values.size:
        time_min = np.nanmin(time_values)
        time_max = np.nanmax(time_values)
        plot_kw_list[1].update(vmin=time_min, vmax=time_max)
        plot_kw_list[3].update(vmin=time_min, vmax=time_max)

    if knot_pos is not None:
        knot_pos = np.asarray(knot_pos, dtype=float).ravel()
        y_values = np.concatenate([y_core, y_edge])
        y_values = y_values[np.isfinite(y_values)]
        y_range = np.ptp(y_values) if y_values.size else 1.0
        knot_y = np.min(y_values) - 0.05 * (y_range or 1.0)
        x_data.append(knot_pos)
        y_data.append(np.full(knot_pos.size, knot_y))
        ey_data.append(None)
        z_data.append(None)
        styles.append('')
        markers.append('o')
        marker_sizes.append(4)
        plot_kw_list.append(dict(color='blue', zorder=5))

    if y_core_out is not None:
        x_data.append(x_core)
        y_data.append(np.asarray(y_core_out, dtype=float))
        ey_data.append(_as_1d_float_array(y_err_core_out))
        z_data.append(None)
        styles.append('')
        markers.append('o')
        marker_sizes.append(9)
        plot_kw_list.append(dict(color='orange', alpha=0.8))

    if y_edge_out is not None:
        x_data.append(x_edge)
        y_data.append(np.asarray(y_edge_out, dtype=float))
        ey_data.append(_as_1d_float_array(y_err_edge_out))
        z_data.append(None)
        styles.append('')
        markers.append('s')
        marker_sizes.append(9)
        plot_kw_list.append(dict(color='green', alpha=0.8))

    save_figure_kw = {'directory': directory} if directory is not None else None
    if fig_name is None:
        fig_name = f'{nshot}_{quantity_key}_spline_fit' if nshot is not None else f'{quantity_key}_spline_fit'

    fig, ax = ipfn_plots(
        x=x_data,
        y=y_data,
        z=z_data,
        ey=ey_data,
        x_labels=x_labels,
        y_labels=y_labels,
        boxes=0,
        write_legends=True,
        linestyles=styles,
        markers=markers,
        markersizes=marker_sizes,
        close=False,
        save_fig=False,
        show=False,
        title=_make_profile_title(plots_kw.pop('title', None), nshot, interval),
        fig_name=fig_name,
        save_figure_kw=save_figure_kw,
        plot_kw=plot_kw_list,
        **plots_kw,
    )
    _restore_grids(ax)
    _add_interval_label(fig, interval)
    _save_final_figure(fig, fig_name, directory, save_fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_profile_comparison(
    x_core,
    x_edge,
    y_core,
    y_edge,
    y_err_core,
    y_err_edge,
    ds_out,
    quantity,
    mtanh_popt=None,
    knot_pos=None,
    time_c=None,
    time_e=None,
    Rsep_eqh=None,
    Rsep_spline=None,
    Rsep_mtanh=None,
    x_labels='R_mid [m]',
    y_labels=None,
    x_mins=None,
    x_maxs=None,
    y_mins=None,
    y_maxs=None,
    directory='/shares/users/work/joanune/Desktop/ipfn_examples/plots',
    close=False,
    save_fig=False,
    show=True,
    nshot=None,
    interval=None,
    fig_name=None,
    **plots_kw,
):
    """Plot full core/edge data with mtanh and spline fits overlaid.

    The optional separatrix positions are drawn as black (EQH), green
    (spline), and red (mtanh) dashed lines. Set ``x_mins``/``x_maxs`` for a
    zoomed view; omit them for the full profile range.
    """
    if ipfn_plots is None:
        raise ImportError('ipfnpytools.plot.plots is not available; install ipfnpytools first.')

    quantity_key = quantity.lower()
    spline_column = {'ne': 'ne_spl', 'te': 'Te_spl'}.get(quantity_key)
    if spline_column is None:
        raise ValueError("quantity must be 'Ne' or 'Te'.")

    x_core = np.asarray(x_core, dtype=float).ravel()
    x_edge = np.asarray(x_edge, dtype=float).ravel()
    y_core = np.asarray(y_core, dtype=float).ravel()
    y_edge = np.asarray(y_edge, dtype=float).ravel()
    y_err_core = np.asarray(y_err_core, dtype=float).ravel()
    y_err_edge = np.asarray(y_err_edge, dtype=float).ravel()
    x_spline = np.asarray(ds_out['R_mid_spl'], dtype=float).ravel()
    y_spline = np.asarray(ds_out[spline_column], dtype=float).ravel()
    labels= []

    if time_c is None:
        time_c = np.array([], dtype=float)
    if time_e is None:
        time_e = np.array([], dtype=float)
    time_values = np.concatenate([time_c, time_e])

    x_all = np.concatenate([x_core, x_edge, x_spline])
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size == 0:
        raise ValueError('No finite profile coordinates available for plotting.')

    x_fit = np.linspace(np.min(x_all), np.max(x_all), 500)
    if mtanh_popt is not None and np.isfinite(np.asarray(mtanh_popt, dtype=float)).all():
        y_mtanh = modified_tanh(x_fit, *np.asarray(mtanh_popt, dtype=float))
    else:
        y_mtanh = None

    y_data = [y_core, y_core, y_edge, y_edge, y_spline]
    x_data = [x_core, x_core, x_edge, x_edge, x_spline]
    ey_data = [y_err_core, None, y_err_edge, None, None]
    z_data = [
        None,
        _build_time_color_values(time_c, x_core.size),
        None,
        _build_time_color_values(time_e, x_edge.size),
        None,
    ]
    styles = ['', '', '', '', '-']
    markers = [None, 'x', None, 'v', None]
    marker_sizes = [0, 14, 0, 14, None]
    plot_kw_list = [
        dict(ecolor='blue', alpha=0.2),
        {},
        dict(ecolor='blue', alpha=0.2),
        {},
        dict(color='green'),
    ]

    if time_values.size:
        time_min = np.nanmin(time_values)
        time_max = np.nanmax(time_values)
        plot_kw_list[1].update(cmap='plasma', vmin=time_min, vmax=time_max)
        plot_kw_list[3].update(cmap='plasma', vmin=time_min, vmax=time_max)
    else:
        marker_sizes = [0, 6, 0, 6, None]

    if y_mtanh is not None:
        x_data.append(x_fit)
        y_data.append(y_mtanh)
        ey_data.append(None)
        z_data.append(None)
        styles.append('-')
        markers.append(None)
        marker_sizes.append(None)
        plot_kw_list.append(dict(color='red'))

    if knot_pos is not None:
        knot_pos = np.asarray(knot_pos, dtype=float).ravel()
        measured_y = np.concatenate([y_core, y_edge])
        measured_y = measured_y[np.isfinite(measured_y)]
        knot_y = np.min(measured_y) - 0.05 * (np.ptp(measured_y) or 1.0)
        x_data.append(knot_pos)
        y_data.append(np.full(knot_pos.size, knot_y))
        ey_data.append(None)
        z_data.append(None)
        styles.append('')
        markers.append('o')
        marker_sizes.append(4)
        plot_kw_list.append(dict(color='blue', zorder=5))

    title = _make_profile_title(plots_kw.pop('title', None), nshot, interval)
    if y_labels is None:
        y_labels = 'Ne [m$^{-3}$]' if quantity_key == 'ne' else 'Te [eV]'

    save_figure_kw = {'directory': directory} if directory is not None else None
    if fig_name is None:
        suffix = 'profile_comparison' if x_mins is None else 'profile_comparison_zoomed'
        fig_name = f'{nshot}_{quantity_key}_{suffix}' if nshot is not None else f'{quantity_key}_{suffix}'

    fig, ax = ipfn_plots(
        x=x_data,
        y=y_data,
        z=z_data,
        ey=ey_data,
        x_labels=x_labels,
        y_labels=y_labels,
        labels= [None, 'Core TS', None, 'Edge TS','Spline Fit','Mtanh Fit','Knots'],
        x_mins=x_mins,
        x_maxs=x_maxs,
        y_mins=y_mins,
        y_maxs=y_maxs,
        boxes=0,
        write_legends=True,
        linestyles=styles,
        markers=markers,
        markersizes=marker_sizes,
        close=False,
        save_fig=False,
        show=False,
        title=title,
        fig_name=fig_name,
        save_figure_kw=save_figure_kw,
        plot_kw=plot_kw_list,
        **plots_kw,
    )
    _restore_grids(ax)

    if Rsep_eqh is not None and np.isfinite(Rsep_eqh):
        ax.axvline(
            x=Rsep_eqh,
            color='blue',
            linestyle='--',
            linewidth=0.5,
        )
    if Rsep_spline is not None and np.isfinite(Rsep_spline):
        ax.axvline(
            x=Rsep_spline,
            color='green',
            linestyle='--',
            linewidth=0.5,
        )
    if Rsep_mtanh is not None and np.isfinite(Rsep_mtanh):
        ax.axvline(
            x=Rsep_mtanh,
            color='red',
            linestyle='--',
            linewidth=0.5,
        )
    if ax.get_legend() is not None:
        ax.legend(loc='upper right')
    _add_interval_label(fig, interval)
    _save_final_figure(fig, fig_name, directory, save_fig)

    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_profile_comparison_full(*args, **kwargs):
    """Plot the complete profile with mtanh, spline, and separatrix lines."""
    kwargs.pop('x_mins', None)
    kwargs.pop('x_maxs', None)
    return plot_profile_comparison(*args, **kwargs)


def plot_profile_comparison_zoomed(x_mins, x_maxs, *args, **kwargs):
    """Plot a zoomed profile with mtanh, spline, and separatrix lines."""
    kwargs['x_mins'] = x_mins
    kwargs['x_maxs'] = x_maxs
    return plot_profile_comparison(*args, **kwargs)


def plot_profile_summary(
    x_core,
    x_edge,
    ne_core,
    ne_edge,
    te_core,
    te_edge,
    ds_out,
    te_popt,
    ne_popt,
    lambda_spline,
    lambda_mtanh,
    q_sol_spline,
    q_cond_spline,
    q_sol_mtanh,
    q_cond_mtanh,
    R_spline=None,
    R_mtanh=None,
    ne_err_core=None,
    ne_err_edge=None,
    te_err_core=None,
    te_err_edge=None,
    Rsep_eqh=None,
    Rsep_spline=None,
    Rsep_mtanh=None,
    fit_metrics=None,
    q_reference_spline=None,
    q_reference_mtanh=None,
    q_intersection_values=None,
    knot_pos=None,
    nshot=None,
    interval=None,
    equilibrium=None,
    r_zoom=None,
    directory=None,
    fig_name=None,
    save_fig=False,
    close=False,
    show=True,
    **plots_kw,
):
    """Plot lambda, heat-flux, and Ne/Te/pressure profiles side by side.

    The five rows show the full profile in the first column and an edge zoom
    in the second. Profile coordinates and spline outputs come from ``ds_out``;
    mtanh curves are evaluated from ``te_popt`` and ``ne_popt``. Lambda and
    heat-flux inputs are SI values and are displayed in mm and GW/m^2.
    """
    def values(value, name):
        if value is None:
            raise ValueError(f'{name} is required for the summary plot.')
        return np.asarray(value, dtype=float).ravel()

    def optional_values(value):
        return None if value is None else np.asarray(value, dtype=float).ravel()

    x_core = values(x_core, 'x_core')
    x_edge = values(x_edge, 'x_edge')
    ne_core = values(ne_core, 'ne_core')
    ne_edge = values(ne_edge, 'ne_edge')
    te_core = values(te_core, 'te_core')
    te_edge = values(te_edge, 'te_edge')
    ne_core_err = optional_values(ne_err_core)
    ne_edge_err = optional_values(ne_err_edge)
    te_core_err = optional_values(te_err_core)
    te_edge_err = optional_values(te_err_edge)
    x_spline = values(ds_out['R_mid_spl'], 'ds_out[R_mid_spl]')
    ne_spline = values(ds_out['ne_spl'], 'ds_out[ne_spl]')
    te_spline = values(ds_out['Te_spl'], 'ds_out[Te_spl]')
    x_mtanh = np.linspace(
        np.nanmin(np.concatenate([x_core, x_edge])),
        np.nanmax(np.concatenate([x_core, x_edge])),
        500,
    )
    ne_mtanh = modified_tanh(x_mtanh, *values(ne_popt, 'ne_popt'))
    te_mtanh = modified_tanh(x_mtanh, *values(te_popt, 'te_popt'))
    pressure_core = ne_core * te_core
    pressure_edge = ne_edge * te_edge
    pressure_spline = ne_spline * te_spline
    pressure_mtanh = ne_mtanh * te_mtanh
    with np.errstate(divide='ignore', invalid='ignore'):
        pressure_core_err = pressure_core * np.sqrt(
            (ne_core_err / ne_core) ** 2 + (te_core_err / te_core) ** 2
        ) if ne_core_err is not None and te_core_err is not None else None
        pressure_edge_err = pressure_edge * np.sqrt(
            (ne_edge_err / ne_edge) ** 2 + (te_edge_err / te_edge) ** 2
        ) if ne_edge_err is not None and te_edge_err is not None else None

    density_scale = 1e-19
    temperature_scale = 1e-3
    pressure_scale = 1.602176634e-22
    ne_core_plot = ne_core * density_scale
    ne_edge_plot = ne_edge * density_scale
    ne_spline_plot = ne_spline * density_scale
    ne_mtanh_plot = ne_mtanh * density_scale
    ne_core_err_plot = None if ne_core_err is None else ne_core_err * density_scale
    ne_edge_err_plot = None if ne_edge_err is None else ne_edge_err * density_scale
    te_core_plot = te_core * temperature_scale
    te_edge_plot = te_edge * temperature_scale
    te_spline_plot = te_spline * temperature_scale
    te_mtanh_plot = te_mtanh * temperature_scale
    te_core_err_plot = None if te_core_err is None else te_core_err * temperature_scale
    te_edge_err_plot = None if te_edge_err is None else te_edge_err * temperature_scale
    pressure_core_plot = pressure_core * pressure_scale
    pressure_edge_plot = pressure_edge * pressure_scale
    pressure_spline_plot = pressure_spline * pressure_scale
    pressure_mtanh_plot = pressure_mtanh * pressure_scale
    pressure_core_err_plot = None if pressure_core_err is None else pressure_core_err * pressure_scale
    pressure_edge_err_plot = None if pressure_edge_err is None else pressure_edge_err * pressure_scale

    x_limits = _robust_limits(x_core, x_edge, x_spline, x_mtanh, padding=0.02)

    shot_text = f'#{nshot}' if nshot is not None else 'Shot'
    interval_text = interval
    if isinstance(interval_text, dict):
        interval_text = (
            interval_text.get('t_begin'),
            interval_text.get('t_end'),
        )
    title_parts = [shot_text]
    if interval_text is not None:
        title_parts.append(
            f'Interval: ({float(interval_text[0]):.3f}, {float(interval_text[1]):.3f}) s'
        )
    if equilibrium is not None:
        title_parts.append(str(equilibrium))
    title = ' | '.join(title_parts)
    lambda_spline = values(lambda_spline, 'lambda_spline')
    lambda_mtanh = values(lambda_mtanh, 'lambda_mtanh')
    q_spline = values(q_sol_spline, 'q_sol_spline')
    q_cond = values(q_cond_spline, 'q_cond_spline')
    q_mtanh = values(q_sol_mtanh, 'q_sol_mtanh')
    q_cond_mtanh = values(q_cond_mtanh, 'q_cond_mtanh')
    knot_pos = None if knot_pos is None else values(knot_pos, 'knot_pos')
    q_reference = q_reference_spline
    q_reference_mtanh = q_reference_mtanh
    x_lambda_spline = x_spline if R_spline is None else values(R_spline, 'R_spline')
    x_lambda_mtanh = x_mtanh if R_mtanh is None else values(R_mtanh, 'R_mtanh')
    if r_zoom is None:
        r_zoom = _automatic_zoom_start(
            [x_core, x_edge, x_spline, x_mtanh],
            [Rsep_eqh, Rsep_spline, Rsep_mtanh],
        )
    r_zoom = float(np.clip(r_zoom, x_limits[0], x_limits[1]))
    measured_x = _finite_values(x_core, x_edge)
    if measured_x.size == 0:
        raise ValueError('No finite measured radial coordinates available.')
    core_channels = _group_positions(x_core)
    if core_channels.size >= 2:
        zoom_x_start = float(core_channels[-2])
    else:
        zoom_x_start = float(np.min(measured_x))
    zoom_x_start = float(max(zoom_x_start, np.min(measured_x)))

    def match_grid(values_array, grid, name):
        if values_array.size == 1:
            return np.full(grid.size, values_array[0])
        if values_array.size != grid.size:
            raise ValueError(
                f'{name} has {values_array.size} values, but its coordinate grid '
                f'has {grid.size} values.'
            )
        return values_array

    lambda_spline = match_grid(lambda_spline, x_lambda_spline, 'lambda_spline')
    lambda_mtanh = match_grid(lambda_mtanh, x_lambda_mtanh, 'lambda_mtanh')
    q_spline = match_grid(q_spline, x_lambda_spline, 'q_sol_spline')
    q_cond = match_grid(q_cond, x_lambda_spline, 'q_cond_spline')
    q_mtanh = match_grid(q_mtanh, x_lambda_mtanh, 'q_sol_mtanh')
    q_cond_mtanh = match_grid(q_cond_mtanh, x_lambda_mtanh, 'q_cond_mtanh')

    lambda_spline = lambda_spline * 1e3
    lambda_mtanh = lambda_mtanh * 1e3
    q_spline = q_spline * 1e-9
    q_cond = q_cond * 1e-9
    q_mtanh = q_mtanh * 1e-9
    q_cond_mtanh = q_cond_mtanh * 1e-9

    if knot_pos is None:
        knot_pos = np.array([], dtype=float)

    def values_in_x_range(x_values, y_values, y_errors, x_min, x_max):
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        mask &= (x_values >= x_min) & (x_values <= x_max)
        selected_errors = None if y_errors is None else np.asarray(y_errors, dtype=float)[mask]
        return y_values[mask], selected_errors

    def row_limits(row_data, x_min, x_max, include_zero):
        row_values = []
        for x_values, y_values, y_errors in row_data:
            selected, selected_errors = values_in_x_range(
                x_values, y_values, y_errors, x_min, x_max
            )
            row_values.append(selected)
            if selected_errors is not None:
                row_values.extend((selected - selected_errors, selected + selected_errors))
        return _robust_limits(*row_values, include_zero=include_zero)

    def upper_tail_median(row_data, x_min, x_max):
        row_values = []
        for x_values, y_values, y_errors in row_data:
            selected, selected_errors = values_in_x_range(
                x_values, y_values, y_errors, x_min, x_max
            )
            row_values.append(selected)
            if selected_errors is not None:
                row_values.append(selected + selected_errors)
        finite = _finite_values(*row_values)
        if finite.size == 0:
            raise ValueError('No finite values available for plot limits.')
        threshold = np.nanpercentile(finite, 95.0)
        upper_tail = finite[finite >= threshold]
        return float(np.nanmedian(upper_tail))

    def interpolated_value(x_values, y_values, x_target):
        if x_target is None:
            return None
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(valid):
            return None
        x_values = x_values[valid]
        y_values = y_values[valid]
        target = float(np.nanmedian(np.asarray(x_target, dtype=float)))
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        if target < x_values[0] or target > x_values[-1]:
            return None
        return float(np.interp(target, x_values, y_values))

    x_zoom_max = float(np.max(measured_x))
    if x_zoom_max <= zoom_x_start:
        x_zoom_max = min(x_limits[1], zoom_x_start + 0.01)
    row_data = [
        [(x_lambda_spline, lambda_spline, None), (x_lambda_mtanh, lambda_mtanh, None)],
        [(x_lambda_spline, q_spline, None), (x_lambda_spline, q_cond, None),
         (x_lambda_mtanh, q_mtanh, None), (x_lambda_mtanh, q_cond_mtanh, None)],
        [(x_core, ne_core_plot, ne_core_err_plot), (x_edge, ne_edge_plot, ne_edge_err_plot),
         (x_spline, ne_spline_plot, None), (x_mtanh, ne_mtanh_plot, None)],
        [(x_core, te_core_plot, te_core_err_plot), (x_edge, te_edge_plot, te_edge_err_plot),
         (x_spline, te_spline_plot, None), (x_mtanh, te_mtanh_plot, None)],
        [(x_core, pressure_core_plot, pressure_core_err_plot),
         (x_edge, pressure_edge_plot, pressure_edge_err_plot),
         (x_spline, pressure_spline_plot, None), (x_mtanh, pressure_mtanh_plot, None)],
    ]
    full_y_limits = [row_limits(row, x_limits[0], x_limits[1], True) for row in row_data]
    zoom_y_limits = [row_limits(row, zoom_x_start, x_zoom_max, False) for row in row_data]

    def knot_level(row, x_min, x_max):
        values = []
        for x_values, y_values, y_errors in row_data[row]:
            selected, selected_errors = values_in_x_range(
                x_values, y_values, y_errors, x_min, x_max
            )
            values.append(selected)
            if selected_errors is not None:
                values.extend((selected - selected_errors, selected + selected_errors))
        finite = _finite_values(*values)
        lower = float(np.nanmin(finite)) if finite.size else 0.0
        upper = float(np.nanmax(finite)) if finite.size else 1.0
        scale = max(upper - lower, abs(upper), 1.0e-3)
        return min(lower - 0.06 * scale, -0.02 * scale)

    knot_levels = [knot_level(row, x_limits[0], x_zoom_max) for row in range(2, 5)]
    zoom_knot_levels = [knot_level(row, zoom_x_start, x_zoom_max) for row in range(2, 5)]
    if q_intersection_values is None:
        q_intersection_values = []
        for x_values, first_q, second_q in (
            (x_lambda_spline, q_spline, q_cond),
            (x_lambda_mtanh, q_mtanh, q_cond_mtanh),
        ):
            x_values = np.asarray(x_values, dtype=float)
            first_q = np.asarray(first_q, dtype=float)
            second_q = np.asarray(second_q, dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(first_q) & np.isfinite(second_q)
            x_values = x_values[valid]
            difference = first_q[valid] - second_q[valid]
            first_q = first_q[valid]
            if x_values.size < 2:
                continue
            order = np.argsort(x_values)
            x_values = x_values[order]
            difference = difference[order]
            first_q = first_q[order]
            crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
            for crossing in crossings:
                x_left, x_right = x_values[crossing:crossing + 2]
                if not (zoom_x_start <= x_left <= x_zoom_max or zoom_x_start <= x_right <= x_zoom_max):
                    continue
                if difference[crossing + 1] == difference[crossing]:
                    intersection_q = np.nanmean(first_q[crossing:crossing + 2])
                else:
                    fraction = -difference[crossing] / (
                        difference[crossing + 1] - difference[crossing]
                    )
                    intersection_q = first_q[crossing] + fraction * (
                        first_q[crossing + 1] - first_q[crossing]
                    )
                if np.isfinite(intersection_q):
                    q_intersection_values.append(intersection_q)
    else:
        q_intersection_values = (1e-9 * _finite_values(q_intersection_values)).tolist()
    if q_intersection_values:
        q_intersection_upper = max(q_intersection_values)
        q_padding = 0.08 * max(abs(q_intersection_upper), 1.0e-3)
        q_upper = q_intersection_upper + q_padding
        zoom_y_limits[1] = (0.0, q_upper)

    lambda_separatrix_values = _finite_values(
        interpolated_value(x_lambda_spline, lambda_spline, Rsep_eqh),
        interpolated_value(x_lambda_mtanh, lambda_mtanh, Rsep_eqh),
    )
    if lambda_separatrix_values.size:
        lambda_upper = np.nanmax(lambda_separatrix_values)
        lambda_padding = 0.05 * max(abs(lambda_upper), 1.0e-3)
        zoom_y_limits[0] = (0.0, lambda_upper + lambda_padding)
    else:
        zoom_y_limits[0] = (0.0, max(zoom_y_limits[0][1], 1.0e-3))
    for row in range(2, 5):
        full_lower, _ = full_y_limits[row]
        zoom_lower, _ = zoom_y_limits[row]
        zoom_upper = upper_tail_median(row_data[row], zoom_x_start, x_zoom_max)
        zoom_y_limits[row] = (zoom_lower, zoom_upper * 1.05)
    for row, knot_y in zip(range(2, 5), knot_levels):
        full_y_limits[row] = (min(full_y_limits[row][0], knot_y - 1 * abs(knot_y)), full_y_limits[row][1])
    for row, knot_y in zip(range(2, 5), zoom_knot_levels):
        zoom_y_limits[row] = (min(zoom_y_limits[row][0], knot_y - 1 * abs(knot_y)), zoom_y_limits[row][1])
    y_limits = full_y_limits + zoom_y_limits

    x_data = [
        x_lambda_spline, x_lambda_mtanh,
        x_lambda_spline, x_lambda_spline, x_lambda_mtanh, x_lambda_mtanh,
        x_core, x_edge, x_core, x_edge, x_spline, x_mtanh, knot_pos,
        x_core, x_edge, x_core, x_edge, x_spline, x_mtanh, knot_pos,
        x_core, x_edge, x_core, x_edge, x_spline, x_mtanh, knot_pos,
    ]
    y_data = [
        lambda_spline, lambda_mtanh,
        q_spline, q_cond, q_mtanh, q_cond_mtanh,
        ne_core_plot, ne_edge_plot, ne_core_plot, ne_edge_plot, ne_spline_plot, ne_mtanh_plot,
        knot_levels[0] * np.ones(knot_pos.size),
        te_core_plot, te_edge_plot, te_core_plot, te_edge_plot, te_spline_plot, te_mtanh_plot,
        knot_levels[1] * np.ones(knot_pos.size),
        pressure_core_plot, pressure_edge_plot, pressure_core_plot, pressure_edge_plot,
        pressure_spline_plot, pressure_mtanh_plot, knot_levels[2] * np.ones(knot_pos.size),
    ]
    ey_data = [
        None, None, None, None, None, None,
        ne_core_err_plot, ne_edge_err_plot, None, None, None, None, None,
        te_core_err_plot, te_edge_err_plot, None, None, None, None, None,
        pressure_core_err_plot, pressure_edge_err_plot, None, None, None, None, None,
    ]
    labels = [
        '$\\lambda_{Te,spline}$', '$\\lambda_{Te,mtanh}$',
        '$q_{SOL,spline}$', '$q_{cond,spline}$',
        '$q_{SOL,mtanh}$', '$q_{cond,mtanh}$',
        None, None, 'Core TS', 'Edge TS', 'Spline', 'Mtanh', 'Knots',
        None, None, None, None, None, None, None,
        None, None, None, None, None, None, None,
    ]
    linestyles = [
        '-', '-', '-', '-', '--', '--',
        '', '', '', '', '-', '-', '',
        '', '', '', '', '-', '-', '',
        '', '', '', '', '-', '-', '',
    ]
    markers = [
        None, None, None, None, None, None,
        None, None, 'x', 'v', None, None, 'o',
        None, None, 'x', 'v', None, None, 'o',
        None, None, 'x', 'v', None, None, 'o',
    ]
    markersizes = [
        None, None, None, None, None, None,
        0, 0, 6, 6, None, None, 4,
        0, 0, 6, 6, None, None, 4,
        0, 0, 6, 6, None, None, 4,
    ]
    plot_kw = [
        {'color': 'green'}, {'color': 'red'},
        {'color': 'green'}, {'color': 'blue'},
        {'color': 'red'}, {'color': 'orange'},
        {'ecolor': 'blue', 'alpha': 0.2}, {'ecolor': 'blue', 'alpha': 0.2},
        {'color': 'tab:blue'}, {'color': 'tab:orange'},
        {'color': 'green'}, {'color': 'red'}, {'color': 'blue'},
        {'ecolor': 'blue', 'alpha': 0.2}, {'ecolor': 'blue', 'alpha': 0.2},
        {'color': 'tab:blue'}, {'color': 'tab:orange'},
        {'color': 'green'}, {'color': 'red'}, {'color': 'blue'},
        {'ecolor': 'blue', 'alpha': 0.2}, {'ecolor': 'blue', 'alpha': 0.2},
        {'color': 'tab:blue'}, {'color': 'tab:orange'},
        {'color': 'green'}, {'color': 'red'}, {'color': 'blue'},
    ]
    x_data = x_data + x_data
    y_data = y_data + y_data
    y_data[12 + 27] = zoom_knot_levels[0] * np.ones(knot_pos.size)
    y_data[19 + 27] = zoom_knot_levels[1] * np.ones(knot_pos.size)
    y_data[26 + 27] = zoom_knot_levels[2] * np.ones(knot_pos.size)
    ey_data = ey_data + ey_data
    labels = labels + [None] * len(labels)
    linestyles = linestyles + linestyles
    markers = markers + markers
    markersizes = markersizes + markersizes
    plot_kw = plot_kw + plot_kw
    boxes = np.repeat(np.arange(5), [2, 4, 7, 7, 7])
    boxes = np.concatenate([boxes, boxes + 5])

    x_min, x_max = x_limits
    box_x_mins = [x_min] * 5 + [zoom_x_start] * 5
    box_x_maxs = [x_max] * 5 + [x_zoom_max] * 5
    x_labels = [None, None, None, None, '$R_{midplane}$'] + [None] * 4 + ['$R_{midplane}$']
    x_units = [None, None, None, None, 'm'] + [None] * 4 + ['m']
    y_units = [
        'mm', '$GW/m^2$', '$10^{19}m^{-3}$', 'keV', 'kPa',
        None, None, None, None, None,
    ]
    y_multipliers = [1.0] * len(y_data)
    fig, axes = ipfn_plots(
        x=x_data,
        y=y_data,
        ey=ey_data,
        boxes=boxes,
        nrows=5,
        ncols=2,
        order='column',
        labels=labels,
        linestyles=linestyles,
        markers=markers,
        markersizes=markersizes,
        plot_kw=plot_kw,
        x_labels=x_labels,
        y_labels=[
            '$\\lambda_{Te}$', '$q_{||}$', '$n_e$', '$T_e$', '$p_e$',
            None, None, None, None, None,
        ],
        x_units_boxes=x_units,
        y_units_boxes=y_units,
        y_multipliers=y_multipliers,
        x_mins_boxes=box_x_mins,
        x_maxs_boxes=box_x_maxs,
        y_mins_boxes=[limit[0] for limit in y_limits],
        y_maxs_boxes=[limit[1] for limit in y_limits],
        v_factor=3,
        h_factor=1.5,
        hspace=0.07,
        wspace=0.2,
        write_legends=True,
        title=title,
        show=False,
        close=False,
        save_fig=False,
        fig_name=fig_name or (f'{nshot}_shot_interval_summary' if nshot is not None else 'shot_interval_summary'),
        save_figure_kw={'directory': directory} if directory is not None else None,
    )
    _restore_grids(axes)

    equilibrium_name = str(equilibrium or 'equilibrium')
    sep_lines = (
        (rf'$R_{{sep,{equilibrium_name}}}$', Rsep_eqh, 'black'),
        (r'$R_{sep,spline}$', Rsep_spline, 'green'),
        (r'$R_{sep,mtanh}$', Rsep_mtanh, 'red'),
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(5, 2)
    for axis in axes[:4, :].flat:
        axis.tick_params(axis='x', labelbottom=False)
    for axis in axes[:, 1]:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    sep_handles = []
    for axis in axes.flat:
        for label, position, color in sep_lines:
            if position is not None and np.isfinite(position):
                line = axis.axvline(position, color=color, linestyle='--', linewidth=0.8, label=label)
                if axis is axes[0, 1]:
                    sep_handles.append(line)
    sep_labels = {label for label, _, _ in sep_lines}
    data_handles, data_labels = axes[2, 0].get_legend_handles_labels()
    data_entries = [
        (handle, label)
        for handle, label in zip(data_handles, data_labels)
        if label not in sep_labels
    ]
    if data_entries:
        axes[2, 0].legend(
            handles=[handle for handle, _ in data_entries],
            labels=[label for _, label in data_entries],
            loc='center left',
        )
    if sep_handles:
        axes[1, 1].legend(
            handles=sep_handles,
            labels=[line.get_label() for line in sep_handles],
            loc='upper left',
            #bbox_to_anchor=(0.98, 0.08),
            fontsize=12,
        )
    for axis in axes[1, :]:
        if q_reference is not None and np.isfinite(q_reference):
            axis.axhline(q_reference / 1e3, color='green', linestyle='--', linewidth=0.8)
        if q_reference_mtanh is not None and np.isfinite(q_reference_mtanh):
            axis.axhline(q_reference_mtanh / 1e3, color='red', linestyle='--', linewidth=0.8)
    _add_interval_label(fig, interval)
    if fit_metrics:
        metric_labels = []
        for method in ('spline', 'mtanh'):
            method_metrics = fit_metrics.get(method, {})
            values_text = []
            for quantity in ('Te', 'Ne'):
                weighted_sse = method_metrics.get(quantity, {}).get('weighted_sse', np.nan)
                if np.isfinite(weighted_sse):
                    values_text.append(f'{quantity} wSSE={weighted_sse:.3g}')
            if values_text:
                metric_labels.append(f'{method}: ' + ', '.join(values_text))
        if metric_labels:
            fig.text(
                0.02,
                0.02,
                'Fit quality | ' + ' | '.join(metric_labels),
                ha='left',
                va='bottom',
                fontsize=8,
            )
    summary_fig_name = fig_name or (
        f'{nshot}_shot_interval_summary' if nshot is not None else 'shot_interval_summary'
    )
    _save_final_figure(fig, summary_fig_name, directory, save_fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axes


def plot_power_signals(
    power_data,
    shot_id=None,
    shot_description=None,
    interval=None,
    intervals=None,
    directory=None,
    fig_name='power_signals',
    save_fig=False,
    close=False,
    show=True,
    **plots_kw,
):
    """Plot shot-level power, stored energy, and ELM timing signals.

    ``power_data`` is the result of ``loading.load_power_signals`` and keeps
    signal values in SI units. Display units are MW, MW/s, and MJ.
    """
    if ipfn_plots is None:
        raise ImportError('ipfnpytools.plot is not available; install ipfnpytools first.')

    signal_specs = (
        ('P_TOT', '$P_{TOT}$ (TOT)', 'MW', 1e-6, 0, 'tab:blue'),
        ('P_Net', '$P_{NET}$ (TTH)', 'MW', 1e-6, 0, 'tab:orange'),
        ('Pr_sep', '$P_{r,sep}$ (BPT)', 'MW', 1e-6, 1, 'tab:purple'),
        ('dWdt_TOT', '$dW_{MHD}/dt$ (TOT)', 'MW/s', 1e-6, 2, 'tab:green'),
        ('W_mhd', '$W_{MHD}$', 'MJ', 1e-6, 3, 'tab:red'),
    )
    x_data = []
    y_data = []
    labels = []
    plot_kw = []
    available = []
    boxes = []
    for key, label, unit, scale, box, color in signal_specs:
        signal = power_data.get('signals', {}).get(key, {})
        time = signal.get('time')
        values = signal.get('values')
        if time is None or values is None:
            continue
        time = np.asarray(time, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        finite = np.isfinite(time) & np.isfinite(values)
        if not np.any(finite):
            continue
        x_data.append(time[finite])
        y_data.append(values[finite] * scale)
        labels.append(label)
        plot_kw.append({'color': color})
        available.append((label, unit))
        boxes.append(box)
    if not x_data:
        raise ValueError('No power or stored-energy signals are available for plotting.')

    elm_frequency = power_data.get('elm_frequency', np.nan)
    if interval is not None:
        if hasattr(interval, 'get'):
            interval_begin = interval.get('t_begin')
            interval_end = interval.get('t_end')
        else:
            interval_begin, interval_end = interval
        f_elm_signal = power_data.get('signals', {}).get('f_ELM', {})
        f_elm_time = f_elm_signal.get('time')
        f_elm_values = f_elm_signal.get('values')
        if f_elm_time is not None and f_elm_values is not None:
            f_elm_time = np.asarray(f_elm_time, dtype=float).ravel()
            f_elm_values = np.asarray(f_elm_values, dtype=float).ravel()
            finite = (
                np.isfinite(f_elm_time)
                & np.isfinite(f_elm_values)
                & (f_elm_time >= float(interval_begin))
                & (f_elm_time <= float(interval_end))
            )
            if np.any(finite):
                elm_frequency = float(np.nanmean(f_elm_values[finite]))
    wmhd_frequency = power_data.get('wmhd_frequency', np.nan)
    title_parts = [f'Shot {shot_id}' if shot_id is not None else 'Power signals']
    title_parts.append(
        f'ELM: {elm_frequency:.1f} Hz' if np.isfinite(elm_frequency)
        else 'ELM: unavailable'
    )
    title_parts.append(
        f'Wmhd: {wmhd_frequency:.0f} Hz' if np.isfinite(wmhd_frequency)
        else 'Wmhd resolution: unavailable'
    )

    fig, axes = ipfn_plots(
        x=x_data,
        y=y_data,
        boxes=boxes,
        nrows=4,
        ncols=1,
        order='column',
        labels=labels,
        linestyles=['-'] * len(x_data),
        markers=[None] * len(x_data),
        markersizes=[None] * len(x_data),
        plot_kw=plot_kw,
        x_labels=['', '', '', 'time (s)'],
        y_labels=['MW', 'MW', 'MW/s', 'MJ'],
        write_legends=True,
        title=' | '.join(title_parts),
        show=False,
        close=False,
        save_fig=False,
        fig_name=fig_name,
        save_figure_kw={'directory': directory} if directory is not None else None,
        **plots_kw,
    )
    axes = np.asarray(axes, dtype=object).ravel()
    _restore_grids(axes)
    elm_times = _finite_values(power_data.get('t_beg_elm'))
    selected_interval = interval
    if selected_interval is None and intervals:
        selected_interval = intervals[0] if isinstance(intervals, (list, tuple)) else intervals
    if selected_interval is not None:
        if hasattr(selected_interval, 'get'):
            begin = selected_interval.get('t_begin')
            end = selected_interval.get('t_end')
            label = selected_interval.get('label', '')
        else:
            begin, end = selected_interval[0], selected_interval[1]
            label = ''
        begin = float(begin)
        end = float(end)

    def interval_mean(key):
        if selected_interval is None:
            return np.nan
        signal = power_data.get('signals', {}).get(key, {})
        time = signal.get('time')
        values = signal.get('values')
        if time is None or values is None:
            return np.nan
        time = np.asarray(time, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        finite = (
            np.isfinite(time)
            & np.isfinite(values)
            & (time >= begin)
            & (time <= end)
        )
        return float(np.nanmean(values[finite])) if np.any(finite) else np.nan

    for axis in axes:
        for elm_time in elm_times:
            axis.axvline(elm_time, color='black', linestyle='--', linewidth=0.8, alpha=0.65)
        if selected_interval is not None:
            axis.axvspan(begin, end, color='grey', alpha=0.12)
            axis.set_xlim(begin, end)
    if selected_interval is not None:
        average_labels = (
            (0, 'avg $P_{abs}$ (TTH):', 'P_Net', 1e-6, 'MW'),
            (1, 'avg $P_{r,sep}$ (BPT):', 'Pr_sep', 1e-6, 'MW'),
            (2, 'avg $dW_{MHD}/dt$ (TOT):', 'dWdt_TOT', 1e-6, 'MW/s'),
        )
        for axis_index, annotation_label, key, scale, unit in average_labels:
            value = interval_mean(key)
            if np.isfinite(value):
                axes[axis_index].text(
                    0.55,
                    0.95,
                    f'{annotation_label} {value * scale:.2f} {unit}',
                    transform=axes[axis_index].transAxes,
                    ha='left',
                    va='top',
                    fontsize=8,
                    color='black',
                    bbox={
                        'facecolor': 'white',
                        'alpha': 0.9,
                        'edgecolor': '0.5',
                        'pad': 3,
                    },
                )
    if selected_interval is not None:
        text = f'{begin:.3f}-{end:.3f} s'
        interval_text = f'{text} ({label})' if label else text
        fig.text(
            0.99,
            0.01,
            interval_text,
            ha='right',
            va='bottom',
            fontsize=8,
        )
    elif shot_description:
        fig.text(
            0.99,
            0.01,
            str(shot_description),
            ha='right',
            va='bottom',
            fontsize=8,
        )
    _save_final_figure(fig, fig_name, directory, save_fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axes


def plot_profile_failure_diagnostic(
    data,
    nshot=None,
    interval=None,
    failure_stage=None,
    error=None,
    directory=None,
    fig_name=None,
    save_fig=False,
    close=False,
    show=False,
):
    """Plot available TS data for an interval whose fit failed.

    This intentionally uses the same ``ipfnpytools.plot.plots`` wrapper as the
    normal profile figures, so failure diagnostics retain the established
    markers, error bars, legends, and layout.
    """
    if ipfn_plots is None:
        raise ImportError('ipfnpytools.plot.plots is not available; install ipfnpytools first.')
    if data is None or len(data) == 0:
        raise ValueError('No TS data available for failure diagnostic plot.')
    coordinate = 'r_midplane' if 'r_midplane' in data else 'r'
    required = {'system', coordinate, 'Te', 'Te_err', 'Ne', 'Ne_err'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'Failure diagnostic data is missing columns: {sorted(missing)}')

    quantities = (('Ne', 'Density', 1e-19), ('Te', 'Temperature', 1e-3))
    systems = (('core', 'Core TS'), ('edge', 'Edge TS'))
    x_data = []
    y_data = []
    ey_data = []
    labels = []
    plot_kw = []
    for row, (quantity, label, scale) in enumerate(quantities):
        for column, (system, system_label) in enumerate(systems):
            subset = data[data['system'] == system]
            x = np.asarray(subset[coordinate], dtype=float)
            y = np.asarray(subset[quantity], dtype=float) * scale
            yerr = np.asarray(subset[f'{quantity}_err'], dtype=float) * scale
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            x_data.append(x[valid])
            y_data.append(y[valid])
            ey_data.append(yerr[valid])
            labels.append(system_label)
            plot_kw.append({'ecolor': 'blue', 'alpha': 0.25})

    title_parts = []
    if nshot is not None:
        title_parts.append(f'Shot {nshot}')
    if interval is not None:
        bounds = interval if not hasattr(interval, 'get') else (
            interval.get('t_begin'), interval.get('t_end'))
        title_parts.append(f'[{float(bounds[0]):.3f}, {float(bounds[1]):.3f}] s')
    title_parts.append(f'Fit failure: {failure_stage or "unknown stage"}')
    title = ' | '.join(title_parts)
    y_labels = [
        'Density ($10^{19}$ m$^{-3}$)', 'Density ($10^{19}$ m$^{-3}$)',
        'Temperature (keV)', 'Temperature (keV)',
    ]
    x_labels = ['R midplane (m)'] * 4
    fig, axes = ipfn_plots(
        x=x_data,
        y=y_data,
        ey=ey_data,
        boxes=[0, 1, 2, 3],
        nrows=2,
        ncols=2,
        order='column',
        labels=labels,
        linestyles=[''] * 4,
        markers=['o'] * 4,
        markersizes=[6] * 4,
        plot_kw=plot_kw,
        x_labels=x_labels,
        y_labels=y_labels,
        title=title,
        write_legends=True,
        show=False,
        close=False,
        save_fig=False,
        fig_name=fig_name or 'profile_failure_diagnostic',
        save_figure_kw={'directory': directory} if directory is not None else None,
    )
    _restore_grids(axes)
    _add_interval_label(fig, interval)
    if error:
        fig.text(0.5, 0.01, str(error), ha='center', va='bottom', fontsize=8, wrap=True)
    _save_final_figure(fig, fig_name, directory, save_fig)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axes
