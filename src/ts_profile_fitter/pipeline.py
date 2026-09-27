"""Batch execution for the TS profile fitting workflow."""

from __future__ import annotations

import json
import os
import time
import traceback
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .connection_length import get_connection_length
from .cache import PhysicsCache
from .derived_quantities import compute_equilibrium_separatrix, compute_physics_separatrix
from .figure_naming import make_fig_name
from .fitting import (
    count_modified_tanh_parameters,
    compute_channel_knots,
    fit_ipfn_spline_ne_Te_TS,
    fit_mbsplt_ne_Te_TS,
    fit_modified_tanh_profile,
    modified_tanh,
)
from .fit_metrics import compute_fit_metrics
from .loading import (
    EquilibriumUnavailableError,
    load_physics_context,
    load_power_signals,
    load_physics_inputs,
    load_ts_data,
)
from .plotting import (
    plot_power_signals,
    plot_profile_comparison_full,
    plot_profile_comparison_zoomed,
    plot_profile_failure_diagnostic,
    plot_profile_summary,
)
from .preprocess import (
    apply_cold_dense_edge_filters,
    apply_distance_outlier_filters,
    apply_preprocessing_filters,
    map_to_r_midplane,
    preprocess_ts_data,
)


def _as_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _interval_dict(interval):
    if hasattr(interval, 'to_dict'):
        interval = interval.to_dict()
    elif not hasattr(interval, 'get'):
        interval = {'t_begin': interval[0], 't_end': interval[1]}
    return {
        't_begin': float(interval.get('t_begin')),
        't_end': float(interval.get('t_end')),
        'label': interval.get('label', ''),
    }


def _fit_config(config):
    config = config or {}
    return config.get('preprocess', {}), config.get('fits', config)


def _run_metadata(fit_config):
    fit_config = fit_config or {}
    spline_config = fit_config.get('fits', fit_config).get('spline', {})
    return {
        'equilibrium': fit_config.get('equilibrium', 'EQH'),
        'diagnostic': fit_config.get('diagnostic', 'VTA'),
        'fit_config_id': fit_config.get('fit_config_id', 'default'),
        'spline_method': spline_config.get('method', 'ondrej_spline'),
        'knot_strategy': spline_config.get('knot_strategy', 'global_quantile'),
    }


def _write_cache_reference_row(output_dir, kind, key_data, value):
    """Append a newly computed physics cache value to its legacy-style CSV."""
    if kind == 'physics':
        metadata = value.get('metadata', {})
        power = value.get('power_balance', {})
        geometry = value.get('geometry', {})
        row = {
            'shot': key_data['shot_id'],
            't1 [s]': key_data['t_begin'],
            't2 [s]': key_data['t_end'],
            'time [s]': metadata.get('time'),
            'equilibrium': key_data.get('equilibrium'),
            'P_abs [W]': power.get('P_abs'),
            'P_rsep [W]': power.get('P_rsep'),
            'dWdt [W]': power.get('dWdt'),
            'dWdt_err [W]': power.get('dWdt_err'),
            'R_out [m]': geometry.get('R_out'),
            'B_pol [T]': geometry.get('B_pol'),
            'B_tor [T]': geometry.get('B_tor'),
            'Zeff': geometry.get('Z_eff'),
            'f_ELM': metadata.get('f_ELM'),
            'P_abs source': metadata.get('sources', {}).get('P_abs'),
            'P_rsep source': metadata.get('sources', {}).get('P_rsep'),
            'dWdt source': metadata.get('sources', {}).get('dWdt'),
            'geometry source': metadata.get('sources', {}).get('geometry'),
        }
        filename = 'db_Lgil2024_physics input.csv'
    elif kind == 'connection_length':
        row = {
            'shot': key_data['shot_id'],
            't1 [s]': key_data.get('t_begin'),
            't2 [s]': key_data.get('t_end'),
            'time [s]': key_data.get('time'),
            'dR [m]': key_data.get('dR'),
            'L_par [m]': value,
            'equilibrium': key_data.get('equilibrium'),
            'provider': key_data.get('provider'),
        }
        filename = 'db_Lgil2024_connection length.csv'
    else:
        return

    path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = list(row)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _profile_arrays(mapped, quantity):
    error_column = f'{quantity}_err'
    required = {'r_midplane', quantity, error_column}
    missing = required.difference(mapped.columns)
    if missing:
        raise ValueError(f'Mapped TS data is missing columns: {sorted(missing)}')
    subset = mapped[['r_midplane', quantity, error_column]].apply(
        pd.to_numeric, errors='coerce')
    subset = subset.dropna()
    finite = np.isfinite(subset.to_numpy(dtype=float)).all(axis=1)
    subset = subset.loc[finite]
    if subset.empty:
        raise ValueError(f'No valid {quantity} points remain after preprocessing.')
    return tuple(subset[column].to_numpy(float) for column in ('r_midplane', quantity, error_column))


def _physics_result(R, Te, Ne, physics_inputs, model_config, grid_size, fit_method):
    x_grid = np.linspace(float(np.min(R)), float(np.max(R)), int(grid_size))
    Te = np.asarray(Te, dtype=float).ravel()
    if fit_method == 'mtanh' and (
        not np.isfinite(Te).all() or np.any(Te < 0)
    ):
        nan_grid = np.full(x_grid.shape, np.nan, dtype=float)
        return {
            'Rsep': np.nan,
            'Te_sep': np.nan,
            'Ne_sep': np.nan,
            'P_SOL': np.nan,
            'P_e_SOL': np.nan,
            'fit_method': fit_method,
            'valid': False,
            'invalid_reason': 'modified-tanh Te profile contains values below 0 eV or non-finite values',
            'R_grid': x_grid,
            'mismatch_grid': nan_grid.copy(),
            'Te_grid': nan_grid.copy(),
            'lam_Te_grid': nan_grid.copy(),
            'lam_q_grid': nan_grid.copy(),
            'q_par_P_SOL_grid': nan_grid.copy(),
            'q_par_SH_grid': nan_grid.copy(),
            'optimizer_result': None,
        }
    return compute_physics_separatrix(
        R,
        Te,
        Ne=Ne,
        power_balance=physics_inputs['power_balance'],
        geometry=physics_inputs['geometry'],
        model_config=model_config,
        x_grid=x_grid,
        fit_method=fit_method,
    )


def _fit_quality_metrics(
    x_te,
    y_te,
    yerr_te,
    x_ne,
    y_ne,
    yerr_ne,
    ds_out,
    mtanh_results,
    mtanh_config,
    spline_parameters,
):
    """Compute fit metrics against the original measured TS points."""
    metrics = {'mtanh': {}, 'spline': {}}
    for quantity, x, y, yerr in (
        ('Te', x_te, y_te, yerr_te),
        ('Ne', x_ne, y_ne, yerr_ne),
    ):
        quantity_config = mtanh_config.get(quantity, mtanh_config)
        n_parameters = count_modified_tanh_parameters(
            deg_tail=quantity_config.get('deg_tail', 3),
            deg_top=quantity_config.get('deg_top', 3),
            fix_x0=quantity_config.get('fix_x0'),
            fix_w=quantity_config.get('fix_w'),
            fix_a_tail=quantity_config.get('fix_a_tail'),
        )
        mtanh_fit = modified_tanh(
            x, *np.asarray(mtanh_results[quantity][0], dtype=float)
        )
        metrics['mtanh'][quantity] = compute_fit_metrics(
            y, mtanh_fit, yerr, n_parameters=n_parameters
        )

    spline_x = ds_out['R_mid_spl'].to_numpy(float)
    for quantity, observed, column, parameter_count in (
        ('Te', (x_te, y_te, yerr_te), 'Te_spl', len(spline_parameters['Te'])),
        ('Ne', (x_ne, y_ne, yerr_ne), 'ne_spl', len(spline_parameters['Ne'])),
    ):
        x, y, yerr = observed
        spline_fit = np.interp(x, spline_x, ds_out[column].to_numpy(float))
        metrics['spline'][quantity] = compute_fit_metrics(
            y, spline_fit, yerr, n_parameters=parameter_count
        )
    return metrics


def run_interval(
    shot_id,
    interval,
    ts_data,
    output_dir=None,
    fit_config=None,
    connection_length_provider=None,
    make_plots=False,
    physics_cache=None,
    diagnostic_context=None,
    mapped_shot_data=None,
    physics_context=None,
):
    """Run preprocessing and all profile fits for one shot interval."""
    diagnostic_context = diagnostic_context if diagnostic_context is not None else {}
    diagnostic_context['stage'] = 'preprocess'
    interval = _interval_dict(interval)
    preprocess_config, fits_config = _fit_config(fit_config)
    if mapped_shot_data is None:
        mapped, preprocess_metadata = preprocess_ts_data(
            ts_data,
            interval,
            r_edge_offset=preprocess_config.get('r_shift', preprocess_config.get('r_edge_offset', 0.008)),
            mapping_method=preprocess_config.get('mapping_method', 'ipfnpytools'),
            outlier_config=preprocess_config.get('outlier_config'),
            rhop_filter=preprocess_config.get('rhop_filter'),
            channel_trim=preprocess_config.get('channel_trim'),
            report_outliers=preprocess_config.get('report_outliers', False),
            return_metadata=True,
        )
    else:
        begin, end = interval['t_begin'], interval['t_end']
        selected = mapped_shot_data[
            mapped_shot_data['time'].between(begin, end)
        ].copy().reset_index(drop=True)
        mapped, preprocess_metadata = apply_preprocessing_filters(
            selected,
            outlier_config=preprocess_config.get('outlier_config'),
            rhop_filter=preprocess_config.get('rhop_filter'),
            channel_trim=preprocess_config.get('channel_trim'),
            report_outliers=preprocess_config.get('report_outliers', False),
        )
    cold_dense_config = preprocess_config.get('cold_dense_edge_config')
    if cold_dense_config:
        mapped, cold_dense_metadata = apply_cold_dense_edge_filters(
            mapped,
            cold_dense_config=cold_dense_config,
            report=preprocess_config.get('report_outliers', False),
            return_metadata=True,
        )
        diagnostic_context['cold_dense_edge_metadata'] = cold_dense_metadata
    distance_outlier_config = preprocess_config.get('distance_outlier_config')
    if distance_outlier_config:
        mapped, distance_outlier_metadata = apply_distance_outlier_filters(
            mapped,
            distance_outlier_config=distance_outlier_config,
            report=preprocess_config.get('report_outliers', False),
            return_metadata=True,
        )
        diagnostic_context['distance_outlier_metadata'] = distance_outlier_metadata
    mapped = map_to_r_midplane(mapped)
    diagnostic_context['mapped'] = mapped
    x_te, y_te, yerr_te = _profile_arrays(mapped, 'Te')
    x_ne, y_ne, yerr_ne = _profile_arrays(mapped, 'Ne')

    diagnostic_context['stage'] = 'mtanh'
    mtanh_config = fits_config.get('mtanh', {})
    mtanh_results = {}
    for quantity, x, y, yerr in (('Te', x_te, y_te, yerr_te), ('Ne', x_ne, y_ne, yerr_ne)):
        quantity_config = mtanh_config.get(quantity, mtanh_config)
        mtanh_results[quantity] = fit_modified_tanh_profile(
            x,
            y,
            yerr,
            p0=quantity_config.get('p0'),
            deg_tail=quantity_config.get('deg_tail', 3),
            deg_top=quantity_config.get('deg_top', 3),
            fix_x0=quantity_config.get('fix_x0'),
            fix_w=quantity_config.get('fix_w'),
            fix_a_tail=quantity_config.get('fix_a_tail'),
            y_scale=quantity_config.get('y_scale', 1.0),
        )
    diagnostic_context['mtanh_results'] = mtanh_results

    diagnostic_context['stage'] = 'spline'
    spline_config = fits_config.get('spline', {})
    spline_input = mapped.rename(columns={'r_midplane': 'R_mid'}).dropna(
        subset=['R_mid', 'Ne', 'Ne_err', 'Te', 'Te_err']
    ).rename(columns={'Ne': 'ne', 'Ne_err': 'ne_err'})
    R_knots = spline_config.get('R_knots')
    if R_knots is None:
        R_knots = compute_channel_knots(
            spline_input,
            preprocess_metadata['channels_before'],
            preprocess_metadata['channels_removed'],
            knot_count_shift=spline_config.get('knot_count_shift', 0),
            strategy=spline_config.get('knot_strategy', 'global_quantile'),
        )
    spline_method = spline_config.get('method', 'ondrej_spline')
    if spline_method == 'ondrej_spline':
        spline_result = fit_mbsplt_ne_Te_TS(
            spline_input,
            dR_fine=spline_config.get('dR_fine', 5e-4),
            sigma_conv=spline_config.get('sigma_conv', 5e-3),
            R_knots=R_knots,
            dR_channel_max=spline_config.get('dR_channel_max', 3e-3),
            max_step_scale=spline_config.get('max_step_scale', 3),
            use_errors=spline_config.get('use_errors', True),
            loss=spline_config.get('loss', 'linear'),
            f_scale=spline_config.get('f_scale', 1.0),
            ne_scale=spline_config.get('ne_scale', 1e19),
            Te_scale=spline_config.get('Te_scale', 1e3),
        )
    elif spline_method == 'ipfn_spline':
        spline_result = fit_ipfn_spline_ne_Te_TS(
            spline_input,
            dR_fine=spline_config.get('dR_fine', 5e-4),
            R_knots=R_knots,
            dR_channel_max=spline_config.get('dR_channel_max', 3e-3),
            ne_scale=spline_config.get('ne_scale', 1e19),
            Te_scale=spline_config.get('Te_scale', 1e3),
            s_factor=spline_config.get('s_factor', 1.0),
            s=spline_config.get('s'),
            mono=spline_config.get('mono', 'decreasing'),
            bc_type=spline_config.get('bc_type'),
            maxiter=spline_config.get('maxiter', 200),
            minimize_kw=spline_config.get('minimize_kw'),
        )
    else:
        raise ValueError(
            f"Unknown spline method {spline_method!r}; expected "
            "'ondrej_spline' or 'ipfn_spline'."
        )
    ds_out, knots, p_ne, pcov_ne, p_te, pcov_te = spline_result
    diagnostic_context['ds_out'] = ds_out
    diagnostic_context['knots'] = knots
    fit_metrics = _fit_quality_metrics(
        x_te,
        y_te,
        yerr_te,
        x_ne,
        y_ne,
        yerr_ne,
        ds_out,
        mtanh_results,
        mtanh_config,
        {'Te': p_te, 'Ne': p_ne},
    )
    result = {
        'shot_id': int(shot_id),
        **_run_metadata(fit_config),
        'interval': interval,
        'n_points': int(len(spline_input)),
        'fit_metrics': fit_metrics,
        'mtanh': {
            'Te': {'params': mtanh_results['Te'][0], 'covariance': mtanh_results['Te'][1]},
            'Ne': {'params': mtanh_results['Ne'][0], 'covariance': mtanh_results['Ne'][1]},
        },
        'spline': {
            'method': spline_method,
            'knot_strategy': spline_config.get('knot_strategy', 'global_quantile'),
            'R_mid_spl': ds_out['R_mid_spl'].to_numpy(),
            'knots': knots,
            'ne_params': p_ne,
            'ne_covariance': pcov_ne,
            'Te_params': p_te,
            'Te_covariance': pcov_te,
        },
        'physics': None,
        '_mapped': mapped,
        '_ds_out': ds_out,
        '_knots': knots,
    }

    physics_config = (fit_config or {}).get('physics', {})
    if connection_length_provider is not None and physics_config.get('enabled', True):
        diagnostic_context['stage'] = 'physics'
        equilibrium = (fit_config or {}).get('equilibrium', 'EQH')
        def load_inputs():
            context = physics_context() if callable(physics_context) else physics_context
            return load_physics_inputs(
                shot_id,
                interval,
                equilibrium=equilibrium,
                dWdt_method=physics_config.get('dWdt_method', 'paper'),
                elm_fraction=physics_config.get('elm_fraction', (0.75, 0.95)),
                physics_context=context,
            )
        if physics_cache is None:
            physics_inputs = load_inputs()
            physics_cache_hit = False
        else:
            physics_inputs, physics_cache_hit = physics_cache.physics_inputs(
                shot_id, interval, equilibrium, load_inputs)
        dR = physics_config.get('connection_length', {}).get('dR', 1e-3)
        get_length = lambda: get_connection_length(
            shot_id, interval, provider=connection_length_provider, dR=dR,
            equilibrium=equilibrium)
        if physics_cache is None:
            physics_inputs['geometry']['L_par'] = get_length()
            connection_cache_hit = False
        else:
            physics_inputs['geometry']['L_par'], connection_cache_hit = physics_cache.connection_length(
                shot_id, interval, dR, get_length, equilibrium=equilibrium)
        result_cache = result.setdefault('metadata', {})
        result_cache.update({
            'physics_cache_hit': physics_cache_hit,
            'connection_length_cache_hit': connection_cache_hit,
        })
        R_spl = ds_out['R_mid_spl'].to_numpy(float)
        result['physics'] = {
            'inputs': physics_inputs,
            'eqh': compute_equilibrium_separatrix(
                ts_data['equ'], interval, rho_sep=physics_config.get('rho_sep', 1.0)
            ),
            'spline': _physics_result(
                R_spl,
                ds_out['Te_spl'].to_numpy(float),
                ds_out['ne_spl'].to_numpy(float),
                physics_inputs,
                physics_config.get('model', {}),

                physics_config.get('grid_size', 2000),
                'spline',
            ),
        }
        x_mtanh = np.linspace(min(x_te.min(), x_ne.min()), max(x_te.max(), x_ne.max()), 500)
        result['physics']['mtanh'] = _physics_result(
            x_mtanh,
            modified_tanh(x_mtanh, *mtanh_results['Te'][0]),
            modified_tanh(x_mtanh, *mtanh_results['Ne'][0]),
            physics_inputs,
            physics_config.get('model', {}),
            physics_config.get('grid_size', 2000),
            'mtanh',
        )

    if make_plots:
        diagnostic_context['stage'] = 'plots'
        if output_dir is None:
            raise ValueError('output_dir is required when make_plots=True.')
        plot_config = (fit_config or {}).get('plots', {})
        equilibrium = (fit_config or {}).get('equilibrium', 'EQH')
        plot_kind = plot_config.get('kind', 'comparison')
        if plot_kind == 'summary':
            physics = result.get('physics') or {}
            spline_physics = physics.get('spline')
            mtanh_physics = physics.get('mtanh')
            if spline_physics is None or mtanh_physics is None:
                raise ValueError('Summary plot requires successful spline and mtanh physics results.')
            summary_directory = plot_config.get('summary_directory', output_dir)
            summary_directory = os.path.join(summary_directory, f'shot_{int(shot_id)}')
            os.makedirs(summary_directory, exist_ok=True)
            stem = make_fig_name(
                shot_id, equilibrium, len(knots), interval, 'summary',
                spline_strategy=spline_config.get('knot_strategy', 'global_quantile'),
            )
            plot_profile_summary(
                x_core=mapped.loc[mapped['system'] == 'core', 'r_midplane'],
                x_edge=mapped.loc[mapped['system'] == 'edge', 'r_midplane'],
                ne_core=mapped.loc[mapped['system'] == 'core', 'Ne'],
                ne_edge=mapped.loc[mapped['system'] == 'edge', 'Ne'],
                te_core=mapped.loc[mapped['system'] == 'core', 'Te'],
                te_edge=mapped.loc[mapped['system'] == 'edge', 'Te'],
                ds_out=ds_out,
                te_popt=mtanh_results['Te'][0],
                ne_popt=mtanh_results['Ne'][0],
                lambda_spline=spline_physics['lam_Te_grid'],
                lambda_mtanh=mtanh_physics['lam_Te_grid'],
                q_sol_spline=spline_physics['q_par_P_SOL_grid'],
                q_cond_spline=spline_physics['q_par_SH_grid'],
                q_sol_mtanh=mtanh_physics['q_par_P_SOL_grid'],
                q_cond_mtanh=mtanh_physics['q_par_SH_grid'],
                R_spline=spline_physics['R_grid'],
                R_mtanh=mtanh_physics['R_grid'],
                ne_err_core=mapped.loc[mapped['system'] == 'core', 'Ne_err'],
                ne_err_edge=mapped.loc[mapped['system'] == 'edge', 'Ne_err'],
                te_err_core=mapped.loc[mapped['system'] == 'core', 'Te_err'],
                te_err_edge=mapped.loc[mapped['system'] == 'edge', 'Te_err'],
                Rsep_eqh=physics.get('eqh', {}).get('Rsep_eqh'),
                Rsep_spline=spline_physics.get('Rsep'),
                Rsep_mtanh=mtanh_physics.get('Rsep'),
                fit_metrics=result.get('fit_metrics'),
                knot_pos=knots,
                nshot=shot_id,
                interval=interval,
                equilibrium=equilibrium,
                r_zoom=plot_config.get('r_zoom'),
                directory=summary_directory,
                fig_name=stem,
                save_fig=True,
                close=True,
                show=False,
            )
            result['figure_directory'] = summary_directory
            return result

        os.makedirs(output_dir, exist_ok=True)
        common = {
            'x_core': mapped.loc[mapped['system'] == 'core', 'r_midplane'],
            'x_edge': mapped.loc[mapped['system'] == 'edge', 'r_midplane'],
            'ds_out': ds_out,
            'knot_pos': knots,
            'time_c': mapped.loc[mapped['system'] == 'core', 'time'],
            'time_e': mapped.loc[mapped['system'] == 'edge', 'time'],
            'directory': output_dir,
            'save_fig': True,
            'show': False,
            'close': True,
            'nshot': shot_id,
            'interval': interval,
        }
        physics = result['physics'] or {}
        for quantity, params in (('Ne', mtanh_results['Ne'][0]), ('Te', mtanh_results['Te'][0])):
            data_common = dict(common)
            data_common.update({
                'y_core': mapped.loc[mapped['system'] == 'core', quantity],
                'y_edge': mapped.loc[mapped['system'] == 'edge', quantity],
                'y_err_core': mapped.loc[mapped['system'] == 'core', f'{quantity}_err'],
                'y_err_edge': mapped.loc[mapped['system'] == 'edge', f'{quantity}_err'],
                'quantity': quantity,
                'mtanh_popt': params,
                'Rsep_eqh': physics.get('eqh', {}).get('Rsep_eqh'),
                'Rsep_spline': physics.get('spline', {}).get('Rsep'),
                'Rsep_mtanh': physics.get('mtanh', {}).get('Rsep'),
            })
            stem = make_fig_name(
                shot_id, equilibrium, len(knots), interval,
                f'{quantity.lower()}_comparison_full',
                spline_strategy=spline_config.get('knot_strategy', 'global_quantile'),
            )
            plot_profile_comparison_full(fig_name=stem, **data_common)
            zoom = plot_config.get('r_zoom')
            zoom_max = plot_config.get('r_zoom_max')
            if zoom is not None and zoom_max is not None:
                stem = make_fig_name(
                    shot_id, equilibrium, len(knots), interval,
                    f'{quantity.lower()}_comparison_zoom',
                    spline_strategy=spline_config.get('knot_strategy', 'global_quantile'),
                )
                plot_profile_comparison_zoomed(zoom, zoom_max, fig_name=stem, **data_common)
        result['figure_directory'] = output_dir

    return result


def _plot_failure_diagnostic(shot_id, interval, ts_data, output_dir, fit_config,
                             diagnostic_context, error):
    plot_config = (fit_config or {}).get('plots', {})
    if not plot_config.get('failure_diagnostics', False):
        return None
    directory = plot_config.get('diagnostics_directory', output_dir)
    shot_directory = os.path.join(directory, f'shot_{int(shot_id)}')
    os.makedirs(shot_directory, exist_ok=True)
    mapped = diagnostic_context.get('mapped')
    data = mapped if mapped is not None else ts_data.get('data')
    if data is None:
        return None
    if mapped is None:
        begin, end = interval['t_begin'], interval['t_end']
        data = data[(data['time'] >= begin) & (data['time'] <= end)]
    stage = diagnostic_context.get('stage', 'fit')
    equilibrium = (fit_config or {}).get('equilibrium', 'EQH')
    mtanh_results = diagnostic_context.get('mtanh_results')
    ds_out = diagnostic_context.get('ds_out')
    knots = diagnostic_context.get('knots', [])
    if (stage == 'physics' and mtanh_results is not None and ds_out is not None
            and all(quantity in mtanh_results for quantity in ('Ne', 'Te'))):
        for quantity in ('Ne', 'Te'):
            name = make_fig_name(
                shot_id, equilibrium, len(knots), interval,
                f'physics_failed_{quantity.lower()}_comparison',
            )
            common = {
                'x_core': mapped.loc[mapped['system'] == 'core', 'r_midplane'],
                'x_edge': mapped.loc[mapped['system'] == 'edge', 'r_midplane'],
                'y_core': mapped.loc[mapped['system'] == 'core', quantity],
                'y_edge': mapped.loc[mapped['system'] == 'edge', quantity],
                'y_err_core': mapped.loc[mapped['system'] == 'core', f'{quantity}_err'],
                'y_err_edge': mapped.loc[mapped['system'] == 'edge', f'{quantity}_err'],
                'quantity': quantity,
                'mtanh_popt': mtanh_results[quantity][0],
                'ds_out': ds_out,
                'knot_pos': knots,
                'time_c': mapped.loc[mapped['system'] == 'core', 'time'],
                'time_e': mapped.loc[mapped['system'] == 'edge', 'time'],
                'directory': shot_directory,
                'save_fig': True,
                'show': False,
                'close': True,
                'nshot': shot_id,
                'interval': interval,
                'Rsep_eqh': None,
                'Rsep_spline': None,
                'Rsep_mtanh': None,
            }
            try:
                plot_profile_comparison_full(fig_name=name, **common)
            except Exception as plot_error:
                print(f'  warning | failure comparison skipped | {plot_error}', flush=True)
        return shot_directory

    name = make_fig_name(
        shot_id, equilibrium, len(knots), interval,
        f'{stage}_failed_diagnostic',
    )
    try:
        plot_profile_failure_diagnostic(
            data, nshot=shot_id, interval=interval, failure_stage=stage,
            error=error, directory=shot_directory, fig_name=name,
            save_fig=True, close=True,
        )
        return os.path.join(shot_directory, f'{name}.png')
    except Exception as plot_error:
        print(f'  warning | failure diagnostic plot skipped | {plot_error}', flush=True)
        return None


def _public_result(result):
    return {key: value for key, value in result.items() if not key.startswith('_')}


def _result_warning(result):
    mtanh_physics = (result.get('physics') or {}).get('mtanh', {})
    if mtanh_physics.get('valid', True) is False:
        return mtanh_physics.get('invalid_reason', 'mtanh physics result is unavailable')
    return ''


def _scalar_at_sep(state, grid_name, value_name):
    """Interpolate a grid quantity at the state's separatrix radius."""
    if not state or not np.isfinite(state.get('Rsep', np.nan)):
        return np.nan
    grid = np.asarray(state.get(grid_name, []), dtype=float)
    values = np.asarray(state.get(value_name, []), dtype=float)
    if grid.size == 0 or grid.size != values.size:
        return np.nan
    return float(np.interp(state['Rsep'], grid, values))


def _result_summary(result, status='ok', error=''):
    """Return scalar fields suitable for one row of the enriched database."""
    summary = {
        'fit_status': status,
        'fit_error': error,
        'n_points': result.get('n_points', np.nan) if result else np.nan,
        'n_knots': len(result.get('spline', {}).get('knots', [])) if result else np.nan,
    }
    fit_metrics = result.get('fit_metrics', {}) if result else {}
    for method in ('spline', 'mtanh'):
        for quantity in ('Te', 'Ne'):
            metric_values = fit_metrics.get(method, {}).get(quantity, {})
            prefix = f'{method}_{quantity}_'
            for name in ('sse', 'weighted_sse', 'reduced_chi2'):
                summary[prefix + name] = metric_values.get(name, np.nan)
    physics = result.get('physics') if result else None
    for method in ('spline', 'mtanh'):
        state = (physics or {}).get(method, {})
        prefix = f'{method}_'
        for name in ('Rsep', 'Te_sep', 'Ne_sep', 'P_SOL', 'P_e_SOL'):
            summary[prefix + name] = state.get(name, np.nan)
        summary[prefix + 'lambda_Te_sep'] = _scalar_at_sep(
            state, 'R_grid', 'lam_Te_grid')
        summary[prefix + 'lambda_q_sep'] = _scalar_at_sep(
            state, 'R_grid', 'lam_q_grid')
        summary[prefix + 'q_par_P_SOL_sep'] = _scalar_at_sep(
            state, 'R_grid', 'q_par_P_SOL_grid')
        summary[prefix + 'q_par_SH_sep'] = _scalar_at_sep(
            state, 'R_grid', 'q_par_SH_grid')
    eqh = (physics or {}).get('eqh', {})
    summary['Rsep_eqh'] = eqh.get('Rsep_eqh', np.nan)
    return summary


def _write_run_summary(output_dir, records, selected_shots, run_started, run_finished):
    """Write compact aggregate timing and success metrics for a database run."""
    interval_times = [
        record['_interval_runtime_seconds']
        for record in records
        if '_interval_runtime_seconds' in record
    ]
    successful = sum(record.get('status') in ('ok', 'warning') for record in records)
    warnings = sum(record.get('status') == 'warning' for record in records)
    failed = sum(record.get('status') == 'failed' for record in records)
    skipped = sum(record.get('status') == 'skipped' for record in records)
    total_intervals = successful + failed
    shot_summaries = []
    for shot_id in selected_shots:
        shot_records = [record for record in records if int(record['shot_id']) == int(shot_id)]
        shot_successful = sum(record.get('status') in ('ok', 'warning') for record in shot_records)
        shot_warnings = sum(record.get('status') == 'warning' for record in shot_records)
        shot_total = len(shot_records)
        shot_times = [
            record['_interval_runtime_seconds']
            for record in shot_records
            if '_interval_runtime_seconds' in record
        ]
        shot_summaries.append({
            'shot_id': int(shot_id),
            'intervals': shot_total,
            'successful': shot_successful,
            'warnings': shot_warnings,
            'failed': sum(record.get('status') == 'failed' for record in shot_records),
            'skipped': sum(record.get('status') == 'skipped' for record in shot_records),
            'success_rate': shot_successful / shot_total if shot_total else None,
            'interval_time_seconds': {
                'total': sum(shot_times),
                'mean': sum(shot_times) / len(shot_times) if shot_times else None,
            },
        })
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_time_utc': {
            'started': datetime.fromtimestamp(run_started, timezone.utc).isoformat(),
            'finished': datetime.fromtimestamp(run_finished, timezone.utc).isoformat(),
            'wall_seconds': run_finished - run_started,
        },
        'shots': len(selected_shots),
        'intervals': {
            'total': total_intervals,
            'successful': successful,
            'warnings': warnings,
            'failed': failed,
            'skipped': skipped,
            'success_rate': successful / total_intervals if total_intervals else None,
        },
        'interval_time_seconds': {
            'total': sum(interval_times),
            'mean': sum(interval_times) / len(interval_times) if interval_times else None,
            'min': min(interval_times) if interval_times else None,
            'max': max(interval_times) if interval_times else None,
        },
        'per_shot': shot_summaries,
    }
    summary_path = os.path.join(output_dir, 'run_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    return summary_path


def enrich_database(intervals_df, records):
    """Merge scalar interval results into a canonical interval database."""
    enriched = intervals_df.copy()
    key_columns = ['shot_id', 't_begin', 't_end']
    summaries = pd.DataFrame(records)
    if summaries.empty:
        for column in ('fit_status', 'fit_error'):
            enriched[column] = pd.Series(dtype=object)
        return enriched
    summaries = summaries.drop_duplicates(key_columns, keep='last')
    return enriched.merge(summaries, on=key_columns, how='left', suffixes=('', '_result'))


def run_shot(shot_id, data_dir, intervals_df, output_dir, fit_config=None,
             connection_length_provider=None, return_records=False,
             physics_cache=None, shot_number=None, shot_total=None,
             shot_description=None):
    """Run all configured intervals for one shot and write JSON results."""
    if physics_cache is None:
        physics_cache = PhysicsCache(
            os.path.join(output_dir, 'cache'), enabled=False)
    shot_intervals = intervals_df[intervals_df['shot_id'] == int(shot_id)]
    run_metadata = _run_metadata(fit_config)
    ts_data = load_ts_data(
        int(shot_id), data_dir,
        diagnostic=(fit_config or {}).get('diagnostic', 'VTA'),
        equilibrium=(fit_config or {}).get('equilibrium', 'EQH'),
    )
    physics_config = (fit_config or {}).get('physics', {})
    physics_context_state = {}

    def get_physics_context():
        if 'value' not in physics_context_state:
            physics_context_state['value'] = load_physics_context(
                int(shot_id),
                equilibrium=(fit_config or {}).get('equilibrium', 'EQH'),
                equilibrium_object=ts_data['equ'],
            )
        return physics_context_state['value']

    physics_context = (
        get_physics_context
        if connection_length_provider is not None
        and physics_config.get('enabled', True)
        else None
    )
    preprocess_config = (fit_config or {}).get('preprocess', {})
    mapping_method = preprocess_config.get('mapping_method', 'ipfnpytools')
    r_edge_offset = preprocess_config.get(
        'r_shift', preprocess_config.get('r_edge_offset', 0.008))
    mapping_started = time.perf_counter()
    mapping_compute = lambda: preprocess_ts_data(
        ts_data,
        interval=None,
        r_edge_offset=r_edge_offset,
        mapping_method=mapping_method,
        apply_filters=False,
    )
    mapped_cache_enabled = physics_cache.is_enabled('mapped_data')
    if mapped_cache_enabled:
        mapped_shot_data, mapped_cache_hit = physics_cache.mapped_data(
            shot_id,
            (fit_config or {}).get('diagnostic', 'VTA'),
            (fit_config or {}).get('equilibrium', 'EQH'),
            mapping_method,
            r_edge_offset,
            mapping_compute,
        )
    else:
        mapped_shot_data = mapping_compute()
        mapped_cache_hit = False
    print(
        f'[shot {int(shot_id)}] mapped preprocessing '
        f'| {"cache" if mapped_cache_hit else "computed"} '
        f'| {time.perf_counter() - mapping_started:.1f} s',
        flush=True,
    )
    shot_output = os.path.join(output_dir, f'shot_{int(shot_id)}')
    os.makedirs(shot_output, exist_ok=True)
    records = []
    total_intervals = len(shot_intervals)
    shot_started = time.perf_counter()
    for interval_number, (_, interval) in enumerate(shot_intervals.iterrows(), start=1):
        interval_started = time.perf_counter()
        print(
            f'[shot {int(shot_id)} {shot_number or "?"}/{shot_total or "?"}] '
            f'interval {interval_number}/{total_intervals} '
            f'{float(interval["t_begin"]):.3f}-{float(interval["t_end"]):.3f} s | running',
            flush=True,
        )
        try:
            diagnostic_context = {}
            result = run_interval(
                int(shot_id), interval, ts_data, shot_output, fit_config,
                connection_length_provider,
                make_plots=(fit_config or {}).get('plots', {}).get('enabled', False),
                physics_cache=physics_cache,
                diagnostic_context=diagnostic_context,
                mapped_shot_data=mapped_shot_data,
                physics_context=physics_context,
            )
            result_path = os.path.join(shot_output, f"interval_{interval['t_begin']:.3f}_{interval['t_end']:.3f}.json")
            with open(result_path, 'w', encoding='utf-8') as stream:
                json.dump(_as_jsonable(_public_result(result)), stream, indent=2)
            warning = _result_warning(result)
            status = 'warning' if warning else 'ok'
            records.append({
                'shot_id': shot_id,
                **_interval_dict(interval),
                **run_metadata,
                'status': status,
                'warning': warning,
                'result': result_path,
                '_interval_runtime_seconds': time.perf_counter() - interval_started,
                '_result_data': result,
            })
            physics = result.get('physics') or {}
            spline_sep = physics.get('spline', {}).get('Rsep', np.nan)
            print(
                f'  done | {status} | {time.perf_counter() - interval_started:.1f} s '
                f'| Rsep_spline={spline_sep:.4f}', flush=True)
        except Exception as exc:
            diagnostic_path = _plot_failure_diagnostic(
                int(shot_id), interval, ts_data, shot_output, fit_config,
                diagnostic_context, str(exc),
            )
            records.append({
                'shot_id': shot_id, **_interval_dict(interval), **run_metadata,
                'status': 'failed',
                'warning': '',
                'error': str(exc), 'traceback': traceback.format_exc(),
                'diagnostic_plot': diagnostic_path or '',
                '_interval_runtime_seconds': time.perf_counter() - interval_started,
            })
            print(
                f'  done | FAILED | {time.perf_counter() - interval_started:.1f} s '
                f'| {exc}', flush=True)
    status_path = os.path.join(shot_output, 'interval_status.csv')
    status_records = [
        {key: value for key, value in record.items() if not key.startswith('_')}
        for record in records
    ]
    pd.DataFrame(status_records).to_csv(status_path, index=False)
    plot_config = (fit_config or {}).get('plots', {})
    if plot_config.get('power_signals', False):
        try:
            power_data = load_power_signals(int(shot_id))
            power_directory = plot_config.get('power_directory', shot_output)
            os.makedirs(power_directory, exist_ok=True)
            power_name = make_fig_name(
                shot_id, run_metadata['equilibrium'], 0,
                {'t_begin': float(shot_intervals['t_begin'].min()),
                 't_end': float(shot_intervals['t_end'].max())},
                'power_signals',
            )
            plot_power_signals(
                power_data,
                shot_id=int(shot_id),
                shot_description=shot_description,
                directory=power_directory,
                fig_name=power_name,
                save_fig=True,
                close=True,
                show=False,
            )
            print(f'  power plot | {os.path.join(power_directory, power_name)}', flush=True)
        except Exception as exc:
            print(f'  warning | power plot skipped | {exc}', flush=True)
    if return_records:
        return records
    return status_path


def run_database(shot_list_df, intervals_df, data_dir, output_dir, fit_config=None, connection_length_provider=None):
    """Run the TS profile fitter and return an enriched selected-shot database."""
    os.makedirs(output_dir, exist_ok=True)
    run_started = time.time()
    selected_shots = shot_list_df['shot_id'].dropna().astype(int).unique()
    selected_intervals = intervals_df[intervals_df['shot_id'].isin(selected_shots)].copy()
    cache_config = (fit_config or {}).get('cache', {})
    mapped_cache_config = cache_config.get('mapped_data', {})
    physics_cache_config = cache_config.get('physics_inputs', {})
    connection_cache_config = cache_config.get('connection_length', {})
    physics_cache = PhysicsCache(
        cache_config.get('directory', os.path.join(output_dir, 'cache')),
        enabled=cache_config.get('enabled'),
        reuse_failed=cache_config.get('reuse_failed', False),
        provider_id=cache_config.get('provider_id', 'get_LcSOL_v1'),
        mapped_data_enabled=mapped_cache_config.get(
            'enabled', cache_config.get('mapped_data_enabled')),
        physics_inputs_enabled=physics_cache_config.get('enabled'),
        connection_length_enabled=connection_cache_config.get('enabled'),
        on_new_cache=lambda kind, key_data, value: _write_cache_reference_row(
            output_dir, kind, key_data, value),
    )
    records = []
    total_shots = len(selected_shots)
    for shot_number, shot_id in enumerate(selected_shots, start=1):
        shot_intervals = selected_intervals[selected_intervals['shot_id'] == shot_id]
        shot_rows = shot_list_df[shot_list_df['shot_id'].astype(int) == int(shot_id)]
        shot_description = (
            shot_rows.iloc[0].get('description')
            if not shot_rows.empty else None
        )
        print(
            f'[shot {int(shot_id)} {shot_number}/{total_shots}] '
            f'starting {len(shot_intervals)} intervals', flush=True)
        try:
            shot_records = run_shot(
                shot_id, data_dir, selected_intervals, output_dir,
                fit_config=fit_config,
                connection_length_provider=connection_length_provider,
                return_records=True,
                physics_cache=physics_cache,
                shot_number=shot_number,
                shot_total=total_shots,
                shot_description=shot_description,
            )
        except EquilibriumUnavailableError as exc:
            shot_records = [{
                'shot_id': int(shot_id),
                **_interval_dict(interval),
                **_run_metadata(fit_config),
                'status': 'skipped',
                'error': str(exc),
                '_interval_runtime_seconds': 0.0,
            } for _, interval in shot_intervals.iterrows()]
            print(
                f'[shot {int(shot_id)} {shot_number}/{total_shots}] '
                f'skipped | {exc}', flush=True)
        records.extend(shot_records)
        shot_records = records[-len(shot_intervals):] if len(shot_intervals) else []
        ok_count = sum(record['status'] == 'ok' for record in shot_records)
        print(
            f'[shot {int(shot_id)} {shot_number}/{total_shots}] '
            f'finished: {ok_count}/{len(shot_intervals)} ok', flush=True)

    summary_records = []
    for record in records:
        summary = {
            'shot_id': int(record['shot_id']),
            **_interval_dict(record),
            **{key: record.get(key, '') for key in (
                'equilibrium', 'diagnostic', 'fit_config_id')},
        }
        if record['status'] in ('ok', 'warning'):
            summary.update(_result_summary(record.get('_result_data'), status='ok'))
        else:
            summary.update(_result_summary(
                None,
                status=record['status'],
                error=record.get('error', ''),
            ))
        summary_records.append(summary)
    run_finished = time.time()
    _write_run_summary(
        output_dir, records, selected_shots, run_started, run_finished)
    return enrich_database(selected_intervals, summary_records)
