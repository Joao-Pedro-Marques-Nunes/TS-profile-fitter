#!/usr/bin/env python
"""Scan spline knot strategies/counts using the pipeline's shared caches."""

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT.parent / 'aug_separatrix'))

from get_connection_length_selectable import get_LcSOL
from ts_profile_fitter.cache import PhysicsCache
from ts_profile_fitter.loading import (
    EquilibriumUnavailableError,
    load_intervals,
    load_physics_context,
    load_shot_list,
    load_ts_data,
)
from ts_profile_fitter.pipeline import (
    _failure_stage,
    _partial_result,
    _scalar_at_sep,
    _stage_status_fields,
    run_interval,
)
from ts_profile_fitter.preprocess import (
    apply_shot_wide_statistical_outlier_filter,
    preprocess_ts_data,
)

KNOT_STRATEGIES = (
    'global_quantile',
    'balanced_quantile',
    'edge_biased_quantile',
    'hybrid',
)
CSV_FIELDS = (
    'shot_id',
    't_begin',
    't_end',
    'interval_label',
    'fit_success',
    'fit_error',
    'failed_stage',
    'spline_status',
    'mtanh_status',
    'physics_status',
    'knot_strategy',
    'knot_count_shift',
    'n_knots',
    'spline_method',
    'spline_knots',
    'Te_spline_parameters',
    'Ne_spline_parameters',
    'Rsep_equilibrium_m',
    'Rsep_spline_m',
    'Te_at_Rsep_eV',
    'Ne_at_Rsep_m3',
    'lambda_Te_sep_m',
    'lambda_Ne_sep_m',
    'lambda_q_sep_m',
    'lambda_p_sep_m',
    'Te_chi_squared',
    'Te_reduced_chi_squared',
    'Ne_chi_squared',
    'Ne_reduced_chi_squared',
    'mapped_cache_hit',
    'physics_cache_hit',
    'connection_length_cache_hit',
)


def _resolve_path(config_dir, value, default):
    path = Path(value if value is not None else default)
    return path if path.is_absolute() else config_dir / path


def _load_fit_config(config):
    fit_config = copy.deepcopy(config.get('fit_config') or {})
    for section in ('preprocess', 'fits', 'physics', 'plots', 'cache'):
        if section in config:
            fit_config[section] = copy.deepcopy(config[section])
    for key in ('diagnostic', 'equilibrium', 'fit_config_id'):
        if key in config:
            fit_config[key] = config[key]
    fit_config.setdefault('diagnostic', 'VTA')
    fit_config.setdefault('equilibrium', 'EQH')
    return fit_config


def _make_cache(config, output_dir):
    cache_config = config.get('cache', {})
    mapped_config = cache_config.get('mapped_data', {})
    physics_config = cache_config.get('physics_inputs', {})
    connection_config = cache_config.get('connection_length', {})
    return PhysicsCache(
        cache_config.get('directory', output_dir / 'cache'),
        enabled=cache_config.get('enabled'),
        reuse_failed=cache_config.get('reuse_failed', False),
        provider_id=cache_config.get('provider_id', 'get_LcSOL_v1'),
        mapped_data_enabled=mapped_config.get(
            'enabled', cache_config.get('mapped_data_enabled')),
        physics_inputs_enabled=physics_config.get('enabled'),
        connection_length_enabled=connection_config.get('enabled'),
    )


def _as_json_array(values):
    if values is None:
        return ''
    array = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(array).all():
        return ''
    return json.dumps(array.tolist(), separators=(',', ':'))


def _finite_or_blank(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ''
    return value if np.isfinite(value) else ''


def _successful_row(result, base_row):
    spline = result['spline']
    spline_metrics = result.get('fit_metrics', {}).get('spline', {})
    te_metrics = spline_metrics.get('Te', {})
    ne_metrics = spline_metrics.get('Ne', {})
    te_params = np.asarray(spline['Te_params'], dtype=float)
    ne_params = np.asarray(spline['ne_params'], dtype=float)
    fit_success = (
        np.isfinite(te_params).all()
        and np.isfinite(ne_params).all()
        and np.isfinite(te_metrics.get('weighted_sse', np.nan))
        and np.isfinite(te_metrics.get('reduced_chi2', np.nan))
        and np.isfinite(ne_metrics.get('weighted_sse', np.nan))
        and np.isfinite(ne_metrics.get('reduced_chi2', np.nan))
    )
    physics = result.get('physics') or {}
    spline_physics = physics.get('spline') or {}
    metadata = result.get('metadata') or {}
    row = {
        **base_row,
        'fit_success': bool(fit_success),
        'fit_error': '' if fit_success else 'Non-finite spline fit output or metrics.',
        'n_knots': int(len(spline.get('knots', ()))),
        'spline_method': spline.get('method', ''),
        'spline_knots': _as_json_array(spline.get('knots')),
        'Te_spline_parameters': _as_json_array(te_params),
        'Ne_spline_parameters': _as_json_array(ne_params),
        'Rsep_equilibrium_m': _finite_or_blank(
            physics.get('eqh', {}).get('Rsep_eqh')),
        'Rsep_spline_m': _finite_or_blank(spline_physics.get('Rsep')),
        'Te_at_Rsep_eV': _finite_or_blank(spline_physics.get('Te_sep')),
        'Ne_at_Rsep_m3': _finite_or_blank(spline_physics.get('Ne_sep')),
        'lambda_Te_sep_m': _finite_or_blank(_scalar_at_sep(
            spline_physics, 'R_grid', 'lam_Te_grid')),
        'lambda_Ne_sep_m': _finite_or_blank(
            spline_physics.get('lambda_Ne_sep')),
        'lambda_q_sep_m': _finite_or_blank(_scalar_at_sep(
            spline_physics, 'R_grid', 'lam_q_grid')),
        'lambda_p_sep_m': _finite_or_blank(
            spline_physics.get('lambda_p_sep')),
        'Te_chi_squared': _finite_or_blank(te_metrics.get('weighted_sse')),
        'Te_reduced_chi_squared': _finite_or_blank(
            te_metrics.get('reduced_chi2')),
        'Ne_chi_squared': _finite_or_blank(ne_metrics.get('weighted_sse')),
        'Ne_reduced_chi_squared': _finite_or_blank(
            ne_metrics.get('reduced_chi2')),
        'physics_cache_hit': metadata.get('physics_cache_hit', ''),
        'connection_length_cache_hit': metadata.get(
            'connection_length_cache_hit', ''),
    }
    return row


def _failed_row(base_row, error, diagnostic_context=None):
    """Row for a failed fit; keeps spline outputs completed before the failure."""
    diagnostic_context = diagnostic_context or {}
    failed_stage = _failure_stage(diagnostic_context, error)
    partial = _partial_result(diagnostic_context)
    row = _successful_row(partial, base_row) if partial is not None else dict(base_row)
    row.update({
        'fit_success': False,
        'fit_error': f'{type(error).__name__}: {error}',
        **_stage_status_fields(diagnostic_context, failed_stage),
    })
    return row


def _intervals_for_shot(intervals, shot_id, requested_time):
    selected = intervals.loc[intervals['shot_id'].eq(int(shot_id))]
    if requested_time is not None:
        selected = selected.loc[
            selected['t_begin'].le(requested_time)
            & selected['t_end'].ge(requested_time)
        ]
    return selected


def _prepare_shot(shot_id, config, fit_config, physics_cache):
    data_dir = config['data_dir']
    diagnostic = fit_config['diagnostic']
    equilibrium = fit_config['equilibrium']
    preprocess_config = fit_config.get('preprocess', {})
    mapping_method = preprocess_config.get('mapping_method', 'ipfnpytools')
    r_edge_offset = preprocess_config.get(
        'r_shift', preprocess_config.get('r_edge_offset', 0.008))
    ts_data = load_ts_data(
        int(shot_id),
        str(data_dir),
        diagnostic=diagnostic,
        equilibrium=equilibrium,
    )

    def compute_mapping():
        return preprocess_ts_data(
            ts_data,
            interval=None,
            r_edge_offset=r_edge_offset,
            mapping_method=mapping_method,
            apply_filters=False,
        )

    mapped_shot_data, mapped_cache_hit = physics_cache.mapped_data(
        shot_id,
        diagnostic,
        equilibrium,
        mapping_method,
        r_edge_offset,
        compute_mapping,
    )
    mapped_shot_data, outlier_metadata = apply_shot_wide_statistical_outlier_filter(
        mapped_shot_data,
        config=preprocess_config.get('local_statistical_outlier_config'),
        rhop_filter=preprocess_config.get('rhop_filter'),
        channel_trim=preprocess_config.get('channel_trim'),
        report=False,
        return_metadata=True,
    )

    physics_context_state = {}

    def get_physics_context():
        if 'value' not in physics_context_state:
            physics_context_state['value'] = load_physics_context(
                int(shot_id),
                equilibrium=equilibrium,
                equilibrium_object=ts_data['equ'],
            )
        return physics_context_state['value']

    physics_context = (
        get_physics_context
        if fit_config.get('physics', {}).get('enabled', True)
        else None
    )
    return (
        ts_data,
        mapped_shot_data,
        outlier_metadata,
        physics_context,
        mapped_cache_hit,
    )


def _scan_one_interval(
    shot_id,
    interval,
    ts_data,
    mapped_shot_data,
    outlier_metadata,
    physics_context,
    mapped_cache_hit,
    strategies,
    knot_shifts,
    fit_config,
    physics_cache,
    writer,
    output_stream,
):
    interval_values = {
        't_begin': float(interval['t_begin']),
        't_end': float(interval['t_end']),
        'label': str(interval.get('label', '')),
    }
    for strategy in strategies:
        for knot_shift in knot_shifts:
            spline_config = copy.deepcopy(
                fit_config.get('fits', {}).get('spline', {}))
            spline_config['knot_strategy'] = strategy
            spline_config['knot_count_shift'] = int(knot_shift)
            spline_config.pop('R_knots', None)
            scan_config = copy.deepcopy(fit_config)
            scan_config.setdefault('fits', {})['spline'] = spline_config
            base_row = {
                'shot_id': int(shot_id),
                't_begin': interval_values['t_begin'],
                't_end': interval_values['t_end'],
                'interval_label': interval_values['label'],
                'knot_strategy': strategy,
                'knot_count_shift': int(knot_shift),
                'mapped_cache_hit': bool(mapped_cache_hit),
                'physics_cache_hit': '',
                'connection_length_cache_hit': '',
            }
            diagnostic_context = {}
            try:
                result = run_interval(
                    int(shot_id),
                    interval_values,
                    ts_data,
                    fit_config=scan_config,
                    connection_length_provider=get_LcSOL,
                    make_plots=False,
                    physics_cache=physics_cache,
                    diagnostic_context=diagnostic_context,
                    mapped_shot_data=mapped_shot_data,
                    physics_context=physics_context,
                    shot_wide_statistical_outlier_metadata=outlier_metadata,
                )
                row = _successful_row(result, base_row)
                row.update(_stage_status_fields(diagnostic_context))
            except Exception as error:
                row = _failed_row(base_row, error, diagnostic_context)
            writer.writerow(row)
            output_stream.flush()
            if row['fit_success']:
                status = 'ok'
            elif row.get('failed_stage'):
                status = f"failed at stage {row['failed_stage']}"
            else:
                status = f"failed: {row['fit_error']}"
            print(
                f"[shot {shot_id}] {strategy}, shift {knot_shift}, "
                f"{row.get('n_knots', '?')} knots: {status}",
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Scan spline knot strategies/shifts using the configured pipeline '
            'cache. Results are written only to one CSV.'
        )
    )
    parser.add_argument('--config', required=True, help='Pipeline YAML configuration.')
    parser.add_argument(
        '--shot-id', type=int, nargs='+', required=True,
        help='One or more shot IDs from the configured shot list.',
    )
    parser.add_argument(
        '--time', type=float,
        help='Optional time; scan only the interval containing this time.',
    )
    parser.add_argument(
        '--strategies', nargs='+', choices=KNOT_STRATEGIES,
        default=list(KNOT_STRATEGIES),
    )
    parser.add_argument(
        '--knot-shifts', nargs='+', type=int, required=True,
        help='Strategy-native knot_count_shift values to scan.',
    )
    parser.add_argument(
        '--output',
        help='CSV destination (default: timestamped file in the configured output directory).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    fit_config = _load_fit_config(config)

    data_dir = _resolve_path(config_dir, config.get('data_dir'), '.')
    shot_list_path = _resolve_path(config_dir, config.get('shot_list'), 'shot_list.csv')
    intervals_path = _resolve_path(
        config_dir, config.get('database', config.get('intervals')),
        'intervals.csv',
    )
    output_dir = _resolve_path(config_dir, config.get('output_dir'), 'results')
    config['data_dir'] = data_dir
    cache_config = copy.deepcopy(fit_config.get('cache', {}))
    cache_directory = _resolve_path(
        config_dir,
        cache_config.get('directory'),
        output_dir / 'cache',
    )
    cache_config['directory'] = cache_directory
    physics_cache = _make_cache({'cache': cache_config}, output_dir)
    disabled_caches = [
        kind for kind in ('mapped_data', 'physics', 'connection_length')
        if not physics_cache.is_enabled(kind)
    ]
    if disabled_caches:
        raise ValueError(
            'The selected config must enable mapped, physics-input, and '
            f'connection-length caches; disabled: {disabled_caches}'
        )

    shot_list = load_shot_list(str(shot_list_path))
    intervals = load_intervals(str(intervals_path))
    configured_shots = set(shot_list['shot_id'].dropna().astype(int))
    requested_shots = list(dict.fromkeys(args.shot_id))
    missing_shots = sorted(set(requested_shots) - configured_shots)
    if missing_shots:
        raise ValueError(f'Shot IDs are not in the configured shot list: {missing_shots}')
    selected_intervals = intervals.loc[
        intervals['shot_id'].isin(requested_shots)
    ].copy()
    if args.time is not None:
        selected_intervals = selected_intervals.loc[
            selected_intervals['t_begin'].le(args.time)
            & selected_intervals['t_end'].ge(args.time)
        ]
    if selected_intervals.empty:
        raise ValueError('No configured intervals match the requested shot/time.')

    strategies = list(dict.fromkeys(args.strategies))
    knot_shifts = list(dict.fromkeys(args.knot_shifts))
    if args.output:
        output_path = _resolve_path(config_dir, args.output, '')
    else:
        output_path = output_dir / (
            'spline_knot_scan_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Using shared cache: {cache_directory}', flush=True)
    print(f'Writing scan results to: {output_path}', flush=True)
    with output_path.open('w', newline='', encoding='utf-8') as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for shot_id in requested_shots:
            shot_intervals = selected_intervals.loc[
                selected_intervals['shot_id'].eq(shot_id)
            ]
            try:
                (
                    ts_data,
                    mapped_shot_data,
                    outlier_metadata,
                    physics_context,
                    mapped_cache_hit,
                ) = _prepare_shot(shot_id, config, fit_config, physics_cache)
            except EquilibriumUnavailableError:
                raise
            print(
                f'[shot {shot_id}] mapped data '
                f"{'cache hit' if mapped_cache_hit else 'cache miss'}",
                flush=True,
            )
            for _, interval in shot_intervals.iterrows():
                _scan_one_interval(
                    shot_id,
                    interval,
                    ts_data,
                    mapped_shot_data,
                    outlier_metadata,
                    physics_context,
                    mapped_cache_hit,
                    strategies,
                    knot_shifts,
                    fit_config,
                    physics_cache,
                    writer,
                    output_stream,
                )
    print(f'Completed knot scan: {output_path}', flush=True)


if __name__ == '__main__':
    main()
