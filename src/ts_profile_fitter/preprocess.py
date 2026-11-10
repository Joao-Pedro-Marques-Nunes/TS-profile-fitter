import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import aug_sfutils as sf
import sys

try:
    from ipfnpytools.trhopz_to_r import trhopz_to_r
    from ipfnpytools.trz_to_rhop import fast_trz_to_rhop
except ImportError:
    trhopz_to_r = None
    fast_trz_to_rhop = None


def _to_array(obj):
    if hasattr(obj, 'data'):
        return np.asarray(obj.data)
    try:
        return np.asarray(obj)
    except Exception:
        return np.asarray(getattr(obj, 'values', obj))


def _resample_z_mag(equ, target_times):
    z_mag = _to_array(equ.Zmag)
    equ_time = _to_array(equ.time)
    if equ_time.ndim != 1:
        equ_time = equ_time.ravel()
    if z_mag.ndim != 1:
        z_mag = z_mag.ravel()
    order = np.argsort(equ_time)
    equ_time = equ_time[order]
    z_mag = z_mag[order]
    interpolator = interp1d(
        equ_time, z_mag, kind='cubic', bounds_error=False,
        fill_value='extrapolate')
    return interpolator(target_times)


def _filter_nan_rows(rhop_matrix, *arrays):
    mask = ~np.isnan(rhop_matrix).any(axis=1)
    return mask, [arr[mask] for arr in arrays]


def _normalize_interval(interval):
    if interval is None:
        return None, None
    if hasattr(interval, 'get'):
        t_begin = interval.get('t_begin', None)
        t_end = interval.get('t_end', None)
    else:
        try:
            t_begin, t_end = interval
        except Exception:
            raise ValueError('Interval must be a mapping or sequence with start/end values.')
    return (None if t_begin is None else float(t_begin),
            None if t_end is None else float(t_end))


def _interval_mask(time, interval):
    t_begin, t_end = _normalize_interval(interval)
    if t_begin is None and t_end is None:
        return np.ones_like(time, dtype=bool)
    mask = np.ones_like(time, dtype=bool)
    if t_begin is not None:
        mask &= time >= t_begin
    if t_end is not None:
        mask &= time <= t_end
    return mask


def _filter_outliers_window(value, error,
                           value_min=None, value_max=None,
                           error_min=None, error_max=None,
                           rel_error_max=None,
                           remove_zeros=False,
                           report=False,
                           report_label=None):
    mask = np.isfinite(value) & np.isfinite(error)
    reasons = {}

    reasons['nonfinite'] = np.sum(~mask)

    if value_min is not None:
        bad = value < value_min
        reasons['value < min'] = np.sum(bad)
        mask &= ~bad

    if value_max is not None:
        bad = value > value_max
        reasons['value > max'] = np.sum(bad)
        mask &= ~bad

    if error_min is not None:
        bad = error < error_min
        reasons['error < min'] = np.sum(bad)
        mask &= ~bad

    if error_max is not None:
        bad = error > error_max
        reasons['error > max'] = np.sum(bad)
        mask &= ~bad

    if rel_error_max is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(error / value)
            bad = rel_error > rel_error_max
            bad |= ~np.isfinite(rel_error)
            reasons['rel_error > max'] = np.sum(bad)
            mask &= ~bad

    if remove_zeros:
        bad_value_zero = value == 0
        bad_error_zero = error == 0
        reasons['value == 0'] = np.sum(bad_value_zero)
        reasons['error == 0'] = np.sum(bad_error_zero)
        mask &= ~(bad_value_zero | bad_error_zero)

    filtered_value = np.where(mask, value, np.nan)
    filtered_error = np.where(mask, error, np.nan)

    if report:
        total_outliers = np.sum(~mask)
        label = f": {report_label}" if report_label else ''
        print(f"[Outlier Filtering Report{label}]")
        print(f"→ Total outliers removed: {total_outliers}")
        print(f"  (Note: A point may be removed for multiple reasons.)")
        for reason, count in reasons.items():
            print(f"  - {reason}: {count} points")

    return filtered_value, filtered_error, mask


def _outlier_rules_for(system, quantity, outlier_config):
    if outlier_config is None:
        return None
    config = {}
    specific_key = f"{quantity}_{system}"
    if specific_key in outlier_config:
        config.update(outlier_config[specific_key])
    if quantity in outlier_config:
        config.update(outlier_config[quantity])
    return config or None


def _normalize_rhop_filter_entry(rhop_filter):
    if not isinstance(rhop_filter, dict):
        raise ValueError("rhop_filter items must be mappings with 'rhop_min' and 'rhop_max'.")

    if 'rhop_min' in rhop_filter and 'rhop_max' in rhop_filter:
        rhop_min = rhop_filter['rhop_min']
        rhop_max = rhop_filter['rhop_max']
    elif 'min' in rhop_filter and 'max' in rhop_filter:
        rhop_min = rhop_filter['min']
        rhop_max = rhop_filter['max']
    else:
        raise ValueError("rhop_filter must define 'rhop_min' and 'rhop_max'.")

    coord = rhop_filter.get('coord', rhop_filter.get('rhop', 'rho_pol'))
    return {
        'coord': coord,
        'rhop_min': float(rhop_min),
        'rhop_max': float(rhop_max),
    }


def _parse_rhop_filter(rhop_filter):
    if rhop_filter is None:
        return None
    if not isinstance(rhop_filter, dict):
        raise ValueError("rhop_filter must be a mapping with optional 'core' and 'edge' sections.")

    if 'core' in rhop_filter or 'edge' in rhop_filter:
        parsed = {}
        for system in ['core', 'edge']:
            if system in rhop_filter:
                parsed[system] = _normalize_rhop_filter_entry(rhop_filter[system])
        return parsed or None

    return {'default': _normalize_rhop_filter_entry(rhop_filter)}


def _initialize_filter_tracking(df):
    df = df.copy()
    if 'keep' not in df.columns:
        df['keep'] = True
    else:
        df['keep'] = df['keep'].fillna(True)
    if 'filter_reason' not in df.columns:
        df['filter_reason'] = ''
    else:
        df['filter_reason'] = df['filter_reason'].fillna('')
    return df


def _append_filter_reason(df, index, reason):
    for idx in np.atleast_1d(index):
        current = df.at[idx, 'filter_reason']
        if current:
            df.at[idx, 'filter_reason'] = f"{current}; {reason}"
        else:
            df.at[idx, 'filter_reason'] = reason


def _apply_measurement_validity_filters(
        df, report=False, return_metadata=False):
    """Remove invalid TS values and uncertainties before statistical filtering."""
    filtered_df = _initialize_filter_tracking(df)
    initially_kept = filtered_df['keep'].to_numpy(dtype=bool)
    removed_mask = np.zeros(len(filtered_df), dtype=bool)
    filter_counts = {}
    reason_counts = {}

    for quantity in ('Te', 'Ne'):
        values = pd.to_numeric(
            filtered_df[quantity], errors='coerce').to_numpy(dtype=float)
        errors = pd.to_numeric(
            filtered_df[f'{quantity}_err'], errors='coerce').to_numpy(dtype=float)
        reason_masks = {
            f'{quantity}_nonfinite_value': ~np.isfinite(values),
            f'{quantity}_zero_value': values == 0,
            f'{quantity}_negative_value': values < 0,
            f'{quantity}_nonfinite_error': ~np.isfinite(errors),
            f'{quantity}_zero_error': errors == 0,
            f'{quantity}_negative_error': errors < 0,
        }
        quantity_invalid = np.zeros(len(filtered_df), dtype=bool)
        for reason, reason_mask in reason_masks.items():
            invalid_positions = initially_kept & reason_mask
            if not invalid_positions.any():
                continue
            reason_indices = filtered_df.index[invalid_positions]
            _append_filter_reason(filtered_df, reason_indices, reason)
            reason_counts[reason] = int(np.sum(invalid_positions))
            quantity_invalid |= invalid_positions

        if quantity_invalid.any():
            invalid_indices = filtered_df.index[quantity_invalid]
            _append_filter_reason(
                filtered_df, invalid_indices, f'{quantity}_outlier')
            filter_counts[f'{quantity}:invalid_measurement'] = int(
                np.sum(quantity_invalid))
            removed_mask |= quantity_invalid

    removed_positions = np.flatnonzero(removed_mask & initially_kept)
    removed_indices = filtered_df.index[removed_positions]
    filtered_df.loc[removed_indices, 'keep'] = False
    removed_rows = filtered_df.iloc[removed_positions].copy()
    filtered_df = filtered_df.loc[filtered_df['keep']].copy()
    metadata = {
        'removed_rows': removed_rows,
        'filter_counts': filter_counts,
        'reason_counts': reason_counts,
        'keep_mask': initially_kept & ~removed_mask,
    }

    if report and reason_counts:
        print('[Measurement Validity Filtering Report]')
        for reason, count in reason_counts.items():
            print(f'  - {reason}: {count} points')

    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def _apply_rhop_filter(df, rhop_filter=None, report=False, return_metadata=False):
    parsed = _parse_rhop_filter(rhop_filter)
    if parsed is None:
        if return_metadata:
            return df, {
                'removed_rows': df.iloc[0:0].copy(),
                'filter_counts': {},
                'channels_removed': {'core': 0, 'edge': 0},
                'channels_before': {
                    system: int(df.loc[df['system'] == system, 'channel_index'].nunique())
                    for system in ['core', 'edge']
                },
                'keep_mask': np.ones(len(df), dtype=bool),
            }
        return df

    filtered_df = _initialize_filter_tracking(df)
    removed_counts = {'core': 0, 'edge': 0}
    channels_removed = {'core': 0, 'edge': 0}
    channels_before = {
        system: int(filtered_df.loc[filtered_df['system'] == system, 'channel_index'].nunique())
        for system in ['core', 'edge']
    }
    removed_frames = []
    keep_mask = np.ones(len(filtered_df), dtype=bool)

    for system in ['core', 'edge']:
        config = parsed.get(system) or parsed.get('default')
        if config is None:
            continue

        coord = config['coord']
        if coord not in filtered_df.columns:
            raise ValueError(f"Requested rhop coordinate '{coord}' is not present in the preprocessing DataFrame.")

        system_mask = filtered_df['system'] == system
        if not np.any(system_mask):
            continue

        system_indices = np.flatnonzero(system_mask)
        coord_values = filtered_df.iloc[system_indices][coord].to_numpy()
        valid = (coord_values >= config['rhop_min']) & (coord_values <= config['rhop_max'])
        removed_positions = system_indices[~valid]
        if removed_positions.size:
            removed_idx = filtered_df.index[removed_positions]
            removed_counts[system] += int(removed_positions.size)
            channels_removed[system] = int(
                filtered_df.iloc[removed_positions]['channel_index'].nunique())
            removed_frames.append(filtered_df.loc[removed_idx].copy())
            keep_mask[removed_positions] = False
            filtered_df.loc[removed_idx, 'keep'] = False
            _append_filter_reason(filtered_df, removed_idx, 'rhop_filter')

    filtered_df = filtered_df.loc[filtered_df['keep']].copy()

    if report:
        print(
            f"Removed {channels_removed['core']} core TS channels "
            f"({removed_counts['core']} rows); removed "
            f"{channels_removed['edge']} edge TS channels "
            f"({removed_counts['edge']} rows).")

    removed_rows = pd.concat(removed_frames, ignore_index=True) if removed_frames else filtered_df.iloc[0:0].copy()
    metadata = {
        'filter_counts': removed_counts,
        'channels_removed': channels_removed,
        'channels_before': channels_before,
        'removed_rows': removed_rows,
        'keep_mask': keep_mask,
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def _apply_channel_trim(df, channel_trim=None, report=False, return_metadata=False):
    """Remove configured leading and trailing channels within each system."""
    if channel_trim is None:
        if return_metadata:
            return df, {
                'removed_rows': df.iloc[0:0].copy(),
                'filter_counts': {},
                'channels_removed': {'core': 0, 'edge': 0},
            }
        return df

    filtered_df = _initialize_filter_tracking(df)
    channels_removed = {'core': 0, 'edge': 0}
    filter_counts = {}
    removed_frames = []

    for system in ['core', 'edge']:
        config = channel_trim.get(system, {})
        first = int(config.get('first', 0))
        last = int(config.get('last', 0))
        if first < 0 or last < 0:
            raise ValueError('channel_trim first/last values must be non-negative.')

        system_mask = filtered_df['system'].eq(system) & filtered_df['keep']
        channel_positions = (
            filtered_df.loc[system_mask]
            .groupby('channel_index', sort=False)['r_midplane']
            .median()
            .sort_values()
        )
        channel_ids = channel_positions.index.to_numpy()
        removed_channels = np.unique(np.concatenate([
            channel_ids[:first],
            channel_ids[-last:] if last else channel_ids[:0],
        ]))
        if removed_channels.size == 0:
            continue

        removed_idx = filtered_df.index[
            system_mask & filtered_df['channel_index'].isin(removed_channels)
        ]
        filtered_df.loc[removed_idx, 'keep'] = False
        _append_filter_reason(filtered_df, removed_idx, 'channel_trim')
        removed_frames.append(filtered_df.loc[removed_idx].copy())
        channels_removed[system] = int(removed_channels.size)
        filter_counts[system] = int(removed_idx.size)

    filtered_df = filtered_df.loc[filtered_df['keep']].copy()
    if report:
        print(
            f"Trimmed {channels_removed['core']} core channels and "
            f"{channels_removed['edge']} edge channels."
        )
    removed_rows = pd.concat(removed_frames, ignore_index=True) if removed_frames else df.iloc[0:0].copy()
    metadata = {
        'removed_rows': removed_rows,
        'filter_counts': filter_counts,
        'channels_removed': channels_removed,
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def _apply_outlier_filters(df, outlier_config=None, report=False, return_metadata=False):
    if outlier_config is None:
        if return_metadata:
            return df, {'removed_rows': df.iloc[0:0].copy(), 'filter_counts': {}, 'keep_mask': np.ones(len(df), dtype=bool)}
        return df

    filtered_df = _initialize_filter_tracking(df)
    filter_counts = {}
    removed_frames = []
    keep_mask = np.ones(len(filtered_df), dtype=bool)

    for system in ['core', 'edge']:
        for quantity in ['Te', 'Ne']:
            config = _outlier_rules_for(system, quantity, outlier_config)
            if config is None:
                continue
            value_col = quantity
            error_col = f"{quantity}_err"
            system_mask = (filtered_df['system'] == system) & filtered_df['keep']
            values = np.asarray(filtered_df.loc[system_mask, value_col])
            errors = np.asarray(filtered_df.loc[system_mask, error_col])
            filtered_value, filtered_error, kept = _filter_outliers_window(
                values, errors,
                value_min=config.get('value_min'),
                value_max=config.get('value_max'),
                error_min=config.get('error_min'),
                error_max=config.get('error_max'),
                rel_error_max=config.get('rel_error_max'),
                remove_zeros=config.get('remove_zeros', False),
                report=report,
                report_label=f'{system}:{quantity}')
            row_idx = filtered_df.index[system_mask]
            removed_idx = row_idx[~kept]
            if removed_idx.size:
                keep_mask[filtered_df.index.get_indexer(removed_idx)] = False
                filtered_df.loc[removed_idx, 'keep'] = False
                _append_filter_reason(filtered_df, removed_idx, f'{quantity}_outlier')
                reason_masks = {
                    f'{quantity}_zero_value': (
                        values == 0 if config.get('remove_zeros', False)
                        else np.zeros(values.shape, dtype=bool)),
                    f'{quantity}_zero_error': (
                        errors == 0 if config.get('remove_zeros', False)
                        else np.zeros(errors.shape, dtype=bool)),
                    f'{quantity}_negative_value': values < 0,
                    f'{quantity}_negative_error': errors < 0,
                }
                for reason, reason_mask in reason_masks.items():
                    reason_idx = row_idx[(~kept) & reason_mask]
                    if reason_idx.size:
                        _append_filter_reason(filtered_df, reason_idx, reason)
                removed_frames.append(filtered_df.loc[removed_idx].copy())
                filter_counts[f'{system}:{quantity}'] = int(np.sum(~kept))
            filtered_df.loc[system_mask, value_col] = filtered_value
            filtered_df.loc[system_mask, error_col] = filtered_error

    filtered_df = filtered_df.loc[filtered_df['keep']].copy()

    removed_rows = pd.concat(removed_frames, ignore_index=True) if removed_frames else filtered_df.iloc[0:0].copy()
    metadata = {
        'filter_counts': filter_counts,
        'removed_rows': removed_rows,
        'keep_mask': keep_mask,
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def apply_cold_dense_edge_filters(df, cold_dense_config=None,
                                  report=False, return_metadata=False):
    """Remove edge points that are simultaneously cold and very dense."""
    if cold_dense_config is None or not cold_dense_config.get('enabled', True):
        if return_metadata:
            return df, {
                'removed_rows': df.iloc[0:0].copy(),
                'filter_counts': {},
                'keep_mask': np.ones(len(df), dtype=bool),
            }
        return df

    required = {'system', 'Te', 'Ne'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f'Cold-dense filtering requires columns: {sorted(missing)}')

    filtered_df = _initialize_filter_tracking(df)
    edge_mask = filtered_df['system'].eq('edge') & filtered_df['keep']
    remove_mask = (
        edge_mask
        & (filtered_df['Te'] < float(cold_dense_config.get('te_max', 1.1)))
        & (filtered_df['Ne'] > float(cold_dense_config.get('ne_min', 2e19)))
    )
    removed_idx = filtered_df.index[remove_mask]
    if removed_idx.size:
        filtered_df.loc[removed_idx, 'keep'] = False
        _append_filter_reason(
            filtered_df, removed_idx, 'cold_dense_edge_outlier')
        if report:
            print('[Cold-Dense Edge Filtering Report]')
            print(f'-> Removed cold/high-density points: {removed_idx.size}')

    keep_mask = np.ones(len(filtered_df), dtype=bool)
    keep_mask[filtered_df.index.get_indexer(removed_idx)] = False
    removed_rows = filtered_df.loc[removed_idx].copy()
    filtered_df = filtered_df.loc[filtered_df['keep']].copy()
    metadata = {
        'removed_rows': removed_rows,
        'filter_counts': {'edge:TeNe': int(removed_idx.size)}
        if removed_idx.size else {},
        'keep_mask': keep_mask,
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def apply_density_outside_separatrix_filter(
        df, config=None, report=False, return_metadata=False):
    """Remove high-density points beyond the separatrix in normalized radius."""
    if config is None or not config.get('enabled', True):
        if return_metadata:
            return df, {
                'removed_rows': df.iloc[0:0].copy(),
                'filter_counts': {},
                'keep_mask': np.ones(len(df), dtype=bool),
            }
        return df

    required = {'Ne', 'rho_pol'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f'Density-outside-separatrix filtering requires columns: {sorted(missing)}')

    filtered_df = _initialize_filter_tracking(df)
    density = pd.to_numeric(filtered_df['Ne'], errors='coerce').to_numpy(dtype=float)
    rho_pol = pd.to_numeric(
        filtered_df['rho_pol'], errors='coerce').to_numpy(dtype=float)
    finite_density = np.isfinite(density)
    density_median = (
        float(np.median(density[finite_density]))
        if finite_density.any() else np.nan
    )
    median_factor = float(config.get('median_factor', 0.5))
    rho_pol_min = float(config.get('rho_pol_min', 1.0))
    density_threshold = median_factor * density_median
    remove_mask = (
        filtered_df['keep'].to_numpy(dtype=bool)
        & finite_density
        & np.isfinite(rho_pol)
        & (density > density_threshold)
        & (rho_pol > rho_pol_min)
    )
    removed_idx = filtered_df.index[remove_mask]
    if removed_idx.size:
        filtered_df.loc[removed_idx, 'keep'] = False
        _append_filter_reason(
            filtered_df, removed_idx, 'high_density_outside_separatrix')
        if report:
            print('[Density-Outside-Separatrix Filtering Report]')
            print(f'-> Concatenated median density: {density_median:g}')
            print(f'-> Density threshold: {density_threshold:g}')
            print(f'-> Removed points: {removed_idx.size}')

    keep_mask = np.ones(len(filtered_df), dtype=bool)
    keep_mask[filtered_df.index.get_indexer(removed_idx)] = False
    removed_rows = filtered_df.loc[removed_idx].copy()
    filtered_df = filtered_df.loc[filtered_df['keep']].copy()
    metadata = {
        'density_median': density_median,
        'density_threshold': density_threshold,
        'rho_pol_min': rho_pol_min,
        'removed_rows': removed_rows,
        'filter_counts': {'high_density_outside_separatrix': int(removed_idx.size)}
        if removed_idx.size else {},
        'keep_mask': keep_mask,
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def apply_distance_outlier_filters(df, distance_outlier_config=None,
                                   report=False, return_metadata=False):
    """Remove isolated low-error measurements using same-channel neighbors.

    The comparison is temporal when ``time`` and ``channel_index`` are
    available, which avoids treating normal profile evolution as an outlier.
    A point is removed only when its value is far from the local median and it
    satisfies the configured high-value and low-relative-error guards.
    """
    if distance_outlier_config is None:
        if return_metadata:
            return df, {
                'removed_rows': df.iloc[0:0].copy(),
                'filter_counts': {},
                'keep_mask': np.ones(len(df), dtype=bool),
            }
        return df

    filtered_df = _initialize_filter_tracking(df)
    removed_frames = []
    filter_counts = {}
    keep_mask = np.ones(len(filtered_df), dtype=bool)
    has_channel_time = {'channel_index', 'time'}.issubset(filtered_df.columns)

    for system in ['core', 'edge']:
        for quantity in ['Te', 'Ne']:
            config = _outlier_rules_for(
                system, quantity, distance_outlier_config)
            if config is None or not config.get('enabled', True):
                continue

            value_col = quantity
            error_col = f'{quantity}_err'
            required = {value_col, error_col}
            if has_channel_time:
                required.update({'channel_index', 'time'})
            missing = required.difference(filtered_df.columns)
            if missing:
                raise ValueError(
                    f'Distance outlier filtering requires columns: {sorted(missing)}')

            system_mask = filtered_df['system'].eq(system) & filtered_df['keep']
            candidate_indices = []
            group_columns = ['channel_index'] if has_channel_time else ['system']
            grouped = filtered_df.loc[system_mask].groupby(
                group_columns, sort=False)
            for _, group in grouped:
                group = group.sort_values('time' if has_channel_time else 'r_midplane')
                values = group[value_col].to_numpy(dtype=float)
                errors = group[error_col].to_numpy(dtype=float)
                te_values = group['Te'].to_numpy(dtype=float) if quantity == 'Ne' and 'Te' in group else None
                te_errors = group['Te_err'].to_numpy(dtype=float) if quantity == 'Ne' and 'Te_err' in group else None
                finite = np.isfinite(values) & np.isfinite(errors)
                valid_positions = np.flatnonzero(finite)
                n_neighbors = int(config.get('n_neighbors', 5))
                min_points = max(int(config.get('min_points', n_neighbors + 1)), 3)
                if valid_positions.size < min_points or n_neighbors < 2:
                    continue

                max_neighbors = min(n_neighbors, valid_positions.size - 1)
                for position in valid_positions:
                    other_positions = valid_positions[valid_positions != position]
                    distances = np.abs(other_positions - position)
                    neighbor_positions = other_positions[
                        np.argsort(distances)[:max_neighbors]]
                    local_values = values[neighbor_positions]
                    local_median = np.median(local_values)
                    local_mad = np.median(np.abs(local_values - local_median))
                    scale = max(
                        1.4826 * local_mad,
                        float(config.get('min_scale', 1e-18)),
                    )
                    use_short_group_fallback = False
                    if quantity == 'Ne' and te_values is not None and te_errors is not None:
                        short_group_max = config.get('short_group_max_points')
                        te_floor = config.get('te_floor')
                        te_error_max = config.get('te_max_relative_error')
                        if (short_group_max is not None and te_floor is not None
                                and te_error_max is not None
                                and valid_positions.size <= int(short_group_max)):
                            with np.errstate(divide='ignore', invalid='ignore'):
                                te_relative_error = abs(
                                    te_errors[position] / te_values[position])
                            use_short_group_fallback = (
                                np.isfinite(te_values[position])
                                and np.isfinite(te_relative_error)
                                and te_values[position] <= float(te_floor)
                                and te_relative_error <= float(te_error_max)
                            )
                    reference = np.min(local_values) if use_short_group_fallback else local_median
                    residual = abs(values[position] - reference)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        relative_error = abs(errors[position] / values[position])
                    is_candidate = (
                        values[position] >= config.get('min_value', -np.inf)
                        and relative_error <= config.get(
                            'max_relative_error', np.inf)
                        and residual > float(config.get('zscore', 8.0)) * scale
                    )
                    if is_candidate:
                        candidate_indices.append(group.index[position])

            if candidate_indices:
                removed_idx = filtered_df.index[filtered_df.index.isin(
                    candidate_indices) & filtered_df['keep']]
                filtered_df.loc[removed_idx, 'keep'] = False
                _append_filter_reason(
                    filtered_df, removed_idx, f'{quantity}_distance_outlier')
                keep_mask[filtered_df.index.get_indexer(removed_idx)] = False
                removed_frames.append(filtered_df.loc[removed_idx].copy())
                filter_counts[f'{system}:{quantity}'] = int(removed_idx.size)
                if report:
                    print(
                        f'[Distance Outlier Filtering Report: {system}:{quantity}]'
                    )
                    print(f'-> Removed isolated points: {removed_idx.size}')

    filtered_df = filtered_df.loc[filtered_df['keep']].copy()
    removed_rows = (
        pd.concat(removed_frames, ignore_index=True)
        if removed_frames else df.iloc[0:0].copy()
    )
    metadata = {
        'removed_rows': removed_rows,
        'filter_counts': filter_counts,
        'keep_mask': keep_mask,
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def _upper_mad_threshold(values, threshold):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median + threshold * 1.4826 * mad


def apply_local_statistical_outlier_filters(
        df, config=None, report=False, return_metadata=False):
    """Filter high values and relative errors using local 2-D robust statistics.

    Radial half-widths are in metres and the time half-width is in seconds.
    A point is evaluated against other measurements from the same system (and
    shot, when present) inside the configured radial-time window. Insufficient
    radial or temporal support leaves that point unclassified. The target
    channel counts toward radial support; nearest rho_pol channels are added
    only when needed to meet the configured radial support.
    """
    if not config or not config.get('enabled', True):
        if return_metadata:
            return df, {
                'removed_rows': df.iloc[0:0].copy(),
                'filter_counts': {},
                'criteria_counts': {},
                'insufficient_support_counts': {},
                'fallback_neighbor_counts': {},
                'keep_mask': np.ones(len(df), dtype=bool),
            }
        return df

    required = {
        'system', 'time', 'r_midplane',
        'Te', 'Te_err', 'Ne', 'Ne_err',
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f'Local statistical outlier filtering requires columns: {sorted(missing)}')

    radial_widths = config.get('radial_half_width_m')
    if not isinstance(radial_widths, dict):
        raise ValueError(
            "local statistical outlier config must define 'radial_half_width_m' "
            "for core and edge.")

    default_threshold = float(config.get('threshold', 3.5))
    upper_value_threshold = float(config.get(
        'upper_value_threshold', default_threshold))
    relative_error_threshold = float(config.get(
        'relative_error_threshold', default_threshold))
    time_half_width = float(config.get('time_half_width_s', 0.05))
    min_neighbors = int(config.get('min_neighbors', 6))
    min_radial_neighbors = int(config.get('min_radial_neighbors', 2))
    min_temporal_neighbors = int(config.get('min_temporal_neighbors', 2))
    fallback_to_nearest_channel = bool(
        config.get('fallback_to_nearest_channel', True))
    for threshold_name, threshold_value in (
            ('upper_value_threshold', upper_value_threshold),
            ('relative_error_threshold', relative_error_threshold)):
        if not np.isfinite(threshold_value) or threshold_value <= 0:
            raise ValueError(
                f'{threshold_name} must be finite and positive.')
    if not np.isfinite(time_half_width) or time_half_width <= 0:
        raise ValueError('time_half_width_s must be finite and positive.')
    if min_neighbors < 1 or min_radial_neighbors < 1 or min_temporal_neighbors < 1:
        raise ValueError('Local statistical minimum neighbor counts must be positive.')

    radial_half_widths = {}
    for system in ('core', 'edge'):
        if system not in radial_widths:
            raise ValueError(
                f"radial_half_width_m must define a width for '{system}'.")
        width = float(radial_widths[system])
        if not np.isfinite(width) or width <= 0:
            raise ValueError(
                f"radial_half_width_m['{system}'] must be finite and positive.")
        radial_half_widths[system] = width

    filtered_df = _initialize_filter_tracking(df)
    initially_kept = filtered_df['keep'].to_numpy(dtype=bool)
    times = pd.to_numeric(filtered_df['time'], errors='coerce').to_numpy(dtype=float)
    radii = pd.to_numeric(
        filtered_df['r_midplane'], errors='coerce').to_numpy(dtype=float)
    channel_ids = (
        filtered_df['channel_index'].to_numpy()
        if 'channel_index' in filtered_df.columns else None
    )
    radial_positions = pd.to_numeric(
        filtered_df['rho_pol'] if 'rho_pol' in filtered_df.columns
        else filtered_df['r_midplane'],
        errors='coerce',
    ).to_numpy(dtype=float)
    values_by_quantity = {
        quantity: pd.to_numeric(
            filtered_df[quantity], errors='coerce').to_numpy(dtype=float)
        for quantity in ('Te', 'Ne')
    }
    errors_by_quantity = {
        quantity: pd.to_numeric(
            filtered_df[f'{quantity}_err'], errors='coerce').to_numpy(dtype=float)
        for quantity in ('Te', 'Ne')
    }

    grouping_columns = ['system']
    if 'shot' in filtered_df.columns:
        grouping_columns.append('shot')
    groups = filtered_df.groupby(
        grouping_columns, sort=False, dropna=False).indices

    removed_mask = np.zeros(len(filtered_df), dtype=bool)
    reasons_by_position = {}
    removed_positions_by_group = {}
    criteria_counts = {}
    insufficient_support_counts = {}
    insufficient_support_rows = []
    fallback_neighbor_counts = {}

    for group_key, positions in groups.items():
        system = group_key[0] if isinstance(group_key, tuple) else group_key
        if system not in radial_half_widths:
            continue

        positions = np.asarray(positions, dtype=int)
        positions = positions[initially_kept[positions]]
        coordinate_valid = (
            np.isfinite(times[positions]) & np.isfinite(radii[positions]))
        coordinate_positions = positions[coordinate_valid]
        if coordinate_positions.size == 0:
            continue

        radial_half_width = radial_half_widths[system]
        scaled_coordinates = np.column_stack((
            radii[coordinate_positions] / radial_half_width,
            times[coordinate_positions] / time_half_width,
        ))
        tree = cKDTree(scaled_coordinates)
        neighborhoods = tree.query_ball_point(
            scaled_coordinates, r=1.0, p=np.inf)
        time_tree = cKDTree(times[coordinate_positions, None])
        time_neighborhoods = time_tree.query_ball_point(
            times[coordinate_positions, None], r=time_half_width)
        fallback_point_positions = set()

        for quantity in ('Te', 'Ne'):
            key = f'{system}:{quantity}'
            quantity_values = values_by_quantity[quantity]
            quantity_errors = errors_by_quantity[quantity]
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_errors = quantity_errors / np.abs(quantity_values)
            relative_errors[
                ~np.isfinite(quantity_values)
                | ~np.isfinite(quantity_errors)
                | (quantity_errors < 0)
            ] = np.nan

            criteria = criteria_counts.setdefault(key, {
                'upper_value': 0,
                'relative_error': 0,
                'both': 0,
            })
            insufficient = insufficient_support_counts.setdefault(key, {
                'upper_value': 0,
                'relative_error': 0,
            })
            removed_for_group = removed_positions_by_group.setdefault(key, set())

            for local_position, local_neighbors in enumerate(neighborhoods):
                point_position = coordinate_positions[local_position]
                if not np.isfinite(quantity_values[point_position]):
                    continue

                neighbor_positions = coordinate_positions[
                    np.asarray(local_neighbors, dtype=int)]
                neighbor_positions = neighbor_positions[
                    neighbor_positions != point_position]

                if (fallback_to_nearest_channel
                        and channel_ids is not None
                    and pd.notna(channel_ids[point_position])
                    and np.isfinite(radial_positions[point_position])):
                    existing_channels = channel_ids[neighbor_positions]
                    existing_channels = existing_channels[
                        pd.notna(existing_channels)]
                    spatial_channels = set(existing_channels.tolist())
                    spatial_channels.add(channel_ids[point_position])
                    required_channels = min_radial_neighbors + 1
                    channels_to_add = max(
                        0, required_channels - len(spatial_channels))
                    if channels_to_add:
                        temporal_positions = coordinate_positions[
                            np.asarray(
                                time_neighborhoods[local_position], dtype=int)]
                        temporal_channels = channel_ids[temporal_positions]
                        radial_valid = np.isfinite(
                            radial_positions[temporal_positions])
                        fallback_candidates = temporal_positions[
                            radial_valid
                            & pd.notna(temporal_channels)
                            & ~np.isin(
                                temporal_channels, list(spatial_channels))
                        ]
                        if fallback_candidates.size:
                            candidate_channels = pd.unique(
                                channel_ids[fallback_candidates])
                            target_radial_position = radial_positions[
                                point_position]
                            channel_distances = [
                                np.median(np.abs(
                                    radial_positions[fallback_candidates][
                                        channel_ids[fallback_candidates]
                                        == candidate_channel]
                                    - target_radial_position
                                ))
                                for candidate_channel in candidate_channels
                            ]
                            nearest_order = np.argsort(channel_distances)[
                                :channels_to_add]
                            added_positions = [
                                fallback_candidates[
                                    channel_ids[fallback_candidates]
                                    == candidate_channels[channel_order]
                                ]
                                for channel_order in nearest_order
                            ]
                            if added_positions:
                                neighbor_positions = np.concatenate((
                                    neighbor_positions,
                                    *added_positions,
                                ))
                                fallback_point_positions.add(point_position)

                metric_flags = {}
                for metric, metric_values, point_value, metric_threshold in (
                    ('upper_value', quantity_values,
                     quantity_values[point_position], upper_value_threshold),
                    ('relative_error', relative_errors,
                     relative_errors[point_position], relative_error_threshold)):
                    if np.isnan(point_value):
                        continue
                    valid_neighbors = neighbor_positions[
                        np.isfinite(metric_values[neighbor_positions])]
                    if valid_neighbors.size < min_neighbors:
                        insufficient[metric] += 1
                        insufficient_support_rows.append({
                            'row_index': filtered_df.index[point_position],
                            'system': system,
                            'shot': filtered_df['shot'].iloc[point_position]
                            if 'shot' in filtered_df.columns else None,
                            'time': times[point_position],
                            'channel_index': channel_ids[point_position]
                            if channel_ids is not None else None,
                            'quantity': quantity,
                            'metric': metric,
                            'reason': 'min_neighbors',
                            'neighbor_count': int(valid_neighbors.size),
                            'radial_support': None,
                            'temporal_support': None,
                        })
                        continue

                    if 'channel_index' in filtered_df.columns:
                        neighbor_channels = channel_ids[valid_neighbors]
                        if pd.notna(channel_ids[point_position]):
                            neighbor_channels = np.append(
                                neighbor_channels, channel_ids[point_position])
                        radial_support = pd.Series(
                            neighbor_channels).nunique(dropna=True)
                    else:
                        radial_support = np.unique(radii[valid_neighbors]).size
                    temporal_support = np.unique(times[valid_neighbors]).size
                    if (radial_support < min_radial_neighbors
                            or temporal_support < min_temporal_neighbors):
                        insufficient[metric] += 1
                        reasons = []
                        if radial_support < min_radial_neighbors:
                            reasons.append('min_radial_neighbors')
                        if temporal_support < min_temporal_neighbors:
                            reasons.append('min_temporal_neighbors')
                        insufficient_support_rows.append({
                            'row_index': filtered_df.index[point_position],
                            'system': system,
                            'shot': filtered_df['shot'].iloc[point_position]
                            if 'shot' in filtered_df.columns else None,
                            'time': times[point_position],
                            'channel_index': channel_ids[point_position]
                            if channel_ids is not None else None,
                            'quantity': quantity,
                            'metric': metric,
                            'reason': '; '.join(reasons),
                            'neighbor_count': int(valid_neighbors.size),
                            'radial_support': int(radial_support),
                            'temporal_support': int(temporal_support),
                        })
                        continue

                    local_threshold = _upper_mad_threshold(
                        metric_values[valid_neighbors], metric_threshold)
                    comparison_tolerance = (
                        np.finfo(float).eps
                        * 16
                        * max(abs(local_threshold), 1.0)
                    )
                    metric_flags[metric] = (
                        point_value > local_threshold + comparison_tolerance)

                value_outlier = metric_flags.get('upper_value', False)
                relative_error_outlier = metric_flags.get('relative_error', False)
                if value_outlier:
                    criteria['upper_value'] += 1
                    reasons_by_position.setdefault(point_position, []).append(
                        f'{quantity}_statistical_upper_value_outlier')
                if relative_error_outlier:
                    criteria['relative_error'] += 1
                    reasons_by_position.setdefault(point_position, []).append(
                        f'{quantity}_statistical_relative_error_outlier')
                if value_outlier and relative_error_outlier:
                    criteria['both'] += 1
                if value_outlier or relative_error_outlier:
                    removed_mask[point_position] = True
                    removed_for_group.add(point_position)

        if fallback_point_positions:
            fallback_neighbor_counts[system] = (
                fallback_neighbor_counts.get(system, 0)
            + len(fallback_point_positions))

    filter_counts = {
        key: len(positions)
        for key, positions in removed_positions_by_group.items()
        if positions
    }
    reason_values = filtered_df['filter_reason'].astype(str).to_numpy(copy=True)
    for position, additions in reasons_by_position.items():
        current = reason_values[position]
        reason_values[position] = '; '.join(
            value for value in (current, *additions) if value)
    filtered_df['filter_reason'] = reason_values
    removed_positions = np.flatnonzero(removed_mask)
    keep_values = filtered_df['keep'].to_numpy(dtype=bool)
    keep_values[removed_positions] = False
    filtered_df['keep'] = keep_values

    metadata = {
        'removed_rows': filtered_df.iloc[removed_positions].copy(),
        'filter_counts': filter_counts,
        'criteria_counts': criteria_counts,
        'insufficient_support_counts': insufficient_support_counts,
        'insufficient_support_rows': pd.DataFrame(
            insufficient_support_rows,
            columns=[
                'row_index', 'system', 'shot', 'time', 'channel_index',
                'quantity', 'metric', 'reason', 'neighbor_count',
                'radial_support', 'temporal_support',
            ],
        ),
        'fallback_neighbor_counts': fallback_neighbor_counts,
        'keep_mask': ~removed_mask,
    }
    if report:
        for key, counts in criteria_counts.items():
            insufficient = insufficient_support_counts[key]
            print(f'[Local Statistical Outlier Report: {key}]')
            print(
                f"Removed {filter_counts.get(key, 0)} rows; upper-value: "
                f"{counts['upper_value']}, relative-error: "
                f"{counts['relative_error']}, both: {counts['both']}."
            )
            print(
                'Skipped for insufficient support; upper-value: '
                f"{insufficient['upper_value']}, relative-error: "
                f"{insufficient['relative_error']}."
            )

    result = filtered_df.loc[filtered_df['keep']].copy()
    if return_metadata:
        return result, metadata
    return result


def _build_flat_dataframe(system, time, r, z, rho, r_midplane,
                          Te, Te_err, Ne, Ne_err):
    time = np.asarray(time)
    rho = np.asarray(rho)
    r_midplane = np.asarray(r_midplane)
    Te = np.asarray(Te)
    Te_err = np.asarray(Te_err)
    Ne = np.asarray(Ne)
    Ne_err = np.asarray(Ne_err)

    if rho.ndim != 2 or r_midplane.ndim != 2:
        raise ValueError('rho and r_midplane must be 2D arrays.')
    n_time, n_chan = rho.shape
    time_mesh = np.repeat(time[:, None], n_chan, axis=1)

    r_arr = np.asarray(r)
    if r_arr.ndim == 1:
        if r_arr.size == n_chan:
            r_mesh = np.repeat(r_arr[None, :], n_time, axis=0)
        elif r_arr.size == n_time:
            r_mesh = np.repeat(r_arr[:, None], n_chan, axis=1)
        else:
            raise ValueError(f'Unsupported r shape {r_arr.shape} for rho shape {rho.shape}')
    elif r_arr.ndim == 2:
        if r_arr.shape == (n_time, n_chan):
            r_mesh = r_arr
        elif r_arr.shape == (n_time, 1):
            r_mesh = np.repeat(r_arr, n_chan, axis=1)
        elif r_arr.shape == (1, n_chan):
            r_mesh = np.repeat(r_arr, n_time, axis=0)
        else:
            raise ValueError(f'Unsupported r shape {r_arr.shape} for rho shape {rho.shape}')
    else:
        raise ValueError(f'Unsupported r ndim {r_arr.ndim} for rho shape {rho.shape}')

    z_arr = np.asarray(z)
    if z_arr.ndim == 1:
        if z_arr.size != n_chan:
            raise ValueError(f'Unsupported z shape {z_arr.shape} for rho shape {rho.shape}')
        z_mesh = np.repeat(z_arr[None, :], n_time, axis=0)
    elif z_arr.ndim == 2:
        if z_arr.shape == (n_time, n_chan):
            z_mesh = z_arr
        elif z_arr.shape == (1, n_chan):
            z_mesh = np.repeat(z_arr, n_time, axis=0)
        else:
            raise ValueError(f'Unsupported z shape {z_arr.shape} for rho shape {rho.shape}')
    else:
        raise ValueError(f'Unsupported z ndim {z_arr.ndim} for rho shape {rho.shape}')

    return pd.DataFrame({
        'system': np.repeat([system], n_time * n_chan),
        'channel_index': np.tile(np.arange(n_chan), n_time),
        'time': time_mesh.ravel(),
        'r_channel': r_mesh.ravel(),
        'z_channel': z_mesh.ravel(),
        'rho_pol': rho.ravel(),
        'r_midplane': r_midplane.ravel(),
        'Te': Te.ravel(),
        'Te_err': Te_err.ravel(),
        'Ne': Ne.ravel(),
        'Ne_err': Ne_err.ravel(),
    })


def compute_rhop_matrix_ipfnpytools(time, r, z, equ, r_offset=0.0):
    if fast_trz_to_rhop is None:
        raise ImportError(
            "ipfnpytools mapping helpers are not available; install ipfnpytools to use this method.")

    time = np.asarray(time, dtype=float)
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    if time.ndim == 1:
        time = time[:, None]
    if r.ndim == 1:
        r = (r + r_offset)[:, None]
    if z.ndim == 1:
        z = z[None, :]

    # Debug: always flush so notebook / terminal shows these before any downstream error
    print('Dimensions before mapping: time:', time.shape, 'r:', r.shape, 'z:', z.shape, flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    return np.asarray(
        fast_trz_to_rhop(time, r, z, eq_shotfile=equ),
        dtype=float,
    )


def compute_rhop_matrix_sfutils(time, r, z, equ, r_offset=0.0):
    time = np.asarray(time, dtype=float).ravel()
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    n_time = time.size

    if r.ndim == 1:
        if z.ndim == 1 and r.size == z.size:
            r_grid = (r + r_offset)[None, :]
        elif r.size == n_time:
            r_grid = (r + r_offset)[:, None]
        else:
            raise ValueError(
                f'1D radius length {r.size} must match either the time '
                f'length {n_time} or the channel length.')
    elif r.ndim == 2:
        r_grid = np.asarray(r, dtype=float)
        if r_grid.shape[0] not in (1, n_time):
            raise ValueError(
                f'Radius array shape {r.shape} is incompatible with '
                f'{n_time} time samples.')
    else:
        raise ValueError(f'Unsupported radius array shape for rho mapping: {r.shape}')

    if z.ndim == 1:
        z_grid = z[None, :]
    elif z.ndim == 2:
        z_grid = np.asarray(z, dtype=float)
        if z_grid.shape[0] not in (1, n_time):
            raise ValueError(
                f'Height array shape {z.shape} is incompatible with '
                f'{n_time} time samples.')
    else:
        raise ValueError(f'Unsupported z array shape for rho mapping: {z.shape}')

    try:
        r_grid, z_grid = np.broadcast_arrays(r_grid, z_grid)
    except ValueError as exc:
        raise ValueError(
            f'Radius and height arrays cannot be broadcast together: '
            f'r={r.shape}, z={z.shape}') from exc
    if r_grid.shape[0] != n_time:
        raise ValueError(
            f'Coordinate grids have {r_grid.shape[0]} time rows, expected '
            f'{n_time}.')

    return equ.rz2rho(
        r_grid, z_grid, t_in=time, coord_out='rho_pol', extrapolate=True)


def compute_rhop_matrix(time, r, z, equ, r_offset=0.0, mapping_method='sfutils'):
    mapping_method = mapping_method.lower()
    if mapping_method == 'ipfnpytools':
        return compute_rhop_matrix_ipfnpytools(time, r, z, equ, r_offset)
    if mapping_method == 'sfutils':
        return compute_rhop_matrix_sfutils(time, r, z, equ, r_offset)
    raise ValueError(f"Unsupported mapping_method: {mapping_method}")


def compute_r_midplane_ipfnpytools(time, rhop_matrix, z_mag_resampled, equ):
    if trhopz_to_r is None:
        raise ImportError(
            "ipfnpytools mapping helpers are not available; install ipfnpytools to use this method.")

    time = np.asarray(time, dtype=float)
    rhop_matrix = np.asarray(rhop_matrix, dtype=float)
    z_mag_resampled = np.asarray(z_mag_resampled, dtype=float)

    if time.ndim == 1:
        time = time[:, None]
    if z_mag_resampled.ndim == 1:
        z_mag_resampled = z_mag_resampled[:, None]

    r_lfs = trhopz_to_r(
        time,
        rhop_matrix,
        z_mag_resampled,
        eq_shotfile=equ,
    )[1] # To select between inner and outer midplane mapping, we choose the outer midplane (index 1). This is a common choice for midplane radius calculations in tokamak plasmas.
    return np.asarray(r_lfs)


def compute_r_midplane_sfutils(time, rhop_matrix, z_mag_resampled, equ):
    time = np.asarray(time, dtype=float).ravel()
    rhop_matrix = np.asarray(rhop_matrix, dtype=float)
    z_mag_resampled = np.asarray(z_mag_resampled, dtype=float)
    if z_mag_resampled.ndim == 1:
        z_mag_resampled = z_mag_resampled[:, None]

    if rhop_matrix.ndim == 1:
        rhop_matrix = rhop_matrix[None, :]
    if rhop_matrix.shape[0] != time.size:
        if rhop_matrix.shape[1] == time.size:
            rhop_matrix = rhop_matrix.T
        else:
            raise ValueError(f'Rho matrix shape {rhop_matrix.shape} is incompatible with time size {time.size}')

    try:
        r_midplane, _ = equ.rhoTheta2rz(
            rhop_matrix, 0, t_in=time, coord_in='rho_pol', n_line=1)
        r_midplane = np.asarray(r_midplane)
        if r_midplane.ndim == 3 and r_midplane.shape[1] == 1:
            return r_midplane[:, 0, :]
        if r_midplane.ndim == 2 and r_midplane.shape[0] == time.size:
            return r_midplane
        return np.asarray(r_midplane)
    except Exception:
        if rhop_matrix.ndim == 2 and rhop_matrix.shape[0] == time.size:
            return np.repeat(np.asarray(rhop_matrix[:, :1], dtype=float), rhop_matrix.shape[1], axis=1)
        return np.asarray(rhop_matrix)


def compute_r_midplane(time, rhop_matrix, z_mag_resampled, equ, mapping_method='sfutils'):
    mapping_method = mapping_method.lower()
    if mapping_method == 'ipfnpytools':
        return compute_r_midplane_ipfnpytools(time, rhop_matrix, z_mag_resampled, equ)
    if mapping_method == 'sfutils':
        return compute_r_midplane_sfutils(time, rhop_matrix, z_mag_resampled, equ)
    raise ValueError(f"Unsupported mapping_method: {mapping_method}")


def _apply_preprocessing_filters(
        all_df, outlier_config=None, rhop_filter=None, channel_trim=None,
        report_outliers=False, local_statistical_outlier_config=None):
    all_df, rhop_metadata = _apply_rhop_filter(
        all_df, rhop_filter=rhop_filter, report=True, return_metadata=True)
    all_df, trim_metadata = _apply_channel_trim(
        all_df, channel_trim=channel_trim, report=True, return_metadata=True)
    validity_metadata = None
    statistical_metadata = None
    statistical_enabled = (
        local_statistical_outlier_config
        and local_statistical_outlier_config.get('enabled', True))
    if statistical_enabled:
        all_df, validity_metadata = _apply_measurement_validity_filters(
            all_df, report=report_outliers, return_metadata=True)
    if local_statistical_outlier_config is not None:
        all_df, statistical_metadata = apply_local_statistical_outlier_filters(
            all_df,
            config=local_statistical_outlier_config,
            report=report_outliers,
            return_metadata=True,
        )
    if outlier_config is None:
        outlier_metadata = {
            'removed_rows': all_df.iloc[0:0].copy(),
            'filter_counts': {},
            'keep_mask': np.ones(len(all_df), dtype=bool),
        }
    else:
        all_df, outlier_metadata = _apply_outlier_filters(
            all_df, outlier_config=outlier_config, report=report_outliers,
            return_metadata=True)
    removed_frames = [
        metadata['removed_rows']
        for metadata in (
                rhop_metadata, trim_metadata, validity_metadata,
                statistical_metadata, outlier_metadata)
        if metadata is not None and not metadata['removed_rows'].empty
    ]
    metadata = {
        'removed_rows': (
            pd.concat(removed_frames, ignore_index=True)
            if removed_frames else all_df.iloc[0:0].copy()
        ),
        'filter_counts': {
            **rhop_metadata['filter_counts'],
            **trim_metadata['filter_counts'],
            **(validity_metadata or {}).get('filter_counts', {}),
            **outlier_metadata['filter_counts'],
        },
        'channels_removed': {
            system: rhop_metadata['channels_removed'][system]
            + trim_metadata['channels_removed'][system]
            for system in ['core', 'edge']
        },
        'channels_before': rhop_metadata['channels_before'],
        'keep_mask': outlier_metadata['keep_mask'],
        'measurement_validity_metadata': validity_metadata,
        'local_statistical_outlier_metadata': statistical_metadata,
    }
    return all_df, metadata


def apply_shot_wide_statistical_outlier_filter(
        ts_df, config=None, rhop_filter=None, channel_trim=None,
        report=False, return_metadata=False):
    """Apply local statistical filtering using the complete mapped shot."""
    source_df = ts_df.copy().reset_index(drop=True)
    if not config or not config.get('enabled', True):
        if return_metadata:
            return source_df, None
        return source_df

    valid_source_df, validity_metadata = _apply_measurement_validity_filters(
        source_df, report=report, return_metadata=True)
    support_df = _apply_rhop_filter(valid_source_df, rhop_filter=rhop_filter)
    support_df = _apply_channel_trim(support_df, channel_trim=channel_trim)
    _, metadata = apply_local_statistical_outlier_filters(
        support_df,
        config=config,
        report=report,
        return_metadata=True,
    )
    statistical_removed_rows = metadata['removed_rows']
    validity_removed_rows = validity_metadata['removed_rows']
    removed_indices = np.union1d(
        statistical_removed_rows.index.to_numpy(dtype=int),
        validity_removed_rows.index.to_numpy(dtype=int),
    )
    result = source_df.drop(index=removed_indices).copy()
    keep_mask = np.ones(len(source_df), dtype=bool)
    keep_mask[removed_indices] = False
    metadata['keep_mask'] = keep_mask
    metadata['statistical_removed_rows'] = statistical_removed_rows
    metadata['validity_filter_counts'] = validity_metadata['filter_counts']
    metadata['validity_filter_scope'] = 'shot'
    metadata['removed_rows'] = pd.concat((
        validity_removed_rows,
        statistical_removed_rows,
    )).sort_index()
    metadata['scope'] = 'shot'
    if return_metadata:
        return result, metadata
    return result


def convert_vta_to_r_midplane(
        ts_dict, interval=None, r_edge_offset=0.008, mapping_method='sfutils',
    outlier_config=None, rhop_filter=None, channel_trim=None,
    report_outliers=False, return_metadata=False, apply_filters=True,
    local_statistical_outlier_config=None):
    """Convert VTA core/edge coordinates to midplane radius and return a flat DataFrame."""
    vta = ts_dict['vta']
    equ = ts_dict['equ']

    t_c_all = np.asarray(_to_array(vta('TIM_CORE')))
    t_e_all = np.asarray(_to_array(vta('TIM_EDGE')))

    r_core = _to_array(vta.getobject('R_core'))
    z_core = _to_array(vta.getobject('Z_core'))
    r_edge = _to_array(vta.getobject('R_edge'))
    z_edge = _to_array(vta.getobject('Z_edge'))

    t_c_mask = _interval_mask(t_c_all, interval)
    t_e_mask = _interval_mask(t_e_all, interval)

    def _filter_time_rows(arr, mask):
        arr = np.asarray(arr)
        if arr.shape[0] != mask.shape[0]:
            raise ValueError(
                'Time-dependent arrays must match the supplied time mask length.')
        return arr[mask]

    Te_c = _filter_time_rows(vta.getobject('Te_c'), t_c_mask)
    SigTe_c = _filter_time_rows(vta.getobject('SigTe_c'), t_c_mask)
    Ne_c = _filter_time_rows(vta.getobject('Ne_c'), t_c_mask)
    SigNe_c = _filter_time_rows(vta.getobject('SigNe_c'), t_c_mask)

    Te_e = _filter_time_rows(vta.getobject('Te_e'), t_e_mask)
    SigTe_e = _filter_time_rows(vta.getobject('SigTe_e'), t_e_mask)
    Ne_e = _filter_time_rows(vta.getobject('Ne_e'), t_e_mask)
    SigNe_e = _filter_time_rows(vta.getobject('SigNe_e'), t_e_mask)

    if np.asarray(r_core).shape[0] == t_c_all.shape[0]:
        r_core = _filter_time_rows(r_core, t_c_mask)
    if np.asarray(r_edge).shape[0] == t_e_all.shape[0]:
        r_edge = _filter_time_rows(r_edge, t_e_mask)

    t_c = t_c_all[t_c_mask]
    t_e = t_e_all[t_e_mask]

    rhop_core = compute_rhop_matrix(t_c, r_core, z_core, equ, r_offset=0.0, mapping_method=mapping_method)
    rhop_edge = compute_rhop_matrix(t_e, r_edge, z_edge, equ, r_offset=r_edge_offset, mapping_method=mapping_method)

    core_mask, filtered_core = _filter_nan_rows(rhop_core, t_c, Te_c, SigTe_c, Ne_c, SigNe_c)
    edge_mask, filtered_edge = _filter_nan_rows(rhop_edge, t_e, Te_e, SigTe_e, Ne_e, SigNe_e)

    t_c_f, Te_c_f, SigTe_c_f, Ne_c_f, SigNe_c_f = filtered_core[0], filtered_core[1], filtered_core[2], filtered_core[3], filtered_core[4]
    t_e_f, Te_e_f, SigTe_e_f, Ne_e_f, SigNe_e_f = filtered_edge[0], filtered_edge[1], filtered_edge[2], filtered_edge[3], filtered_edge[4]

    rhop_core_f = rhop_core[core_mask]
    rhop_edge_f = rhop_edge[edge_mask]

    r_core_f = r_core
    if np.asarray(r_core).ndim >= 1 and np.asarray(r_core).shape[0] == core_mask.size:
        r_core_f = np.asarray(r_core)[core_mask]
    r_edge_f = r_edge
    if np.asarray(r_edge).ndim >= 1 and np.asarray(r_edge).shape[0] == edge_mask.size:
        r_edge_f = np.asarray(r_edge)[edge_mask]

    z_mag_core = _resample_z_mag(equ, t_c_f)
    z_mag_edge = _resample_z_mag(equ, t_e_f)

    r_midplane_core = compute_r_midplane(t_c_f, rhop_core_f, z_mag_core, equ, mapping_method=mapping_method)
    r_midplane_edge = compute_r_midplane(t_e_f, rhop_edge_f, z_mag_edge, equ, mapping_method=mapping_method)

    core_df = _build_flat_dataframe(
        'core', t_c_f, r_core_f, z_core, rhop_core_f, r_midplane_core,
        Te_c_f, SigTe_c_f, Ne_c_f, SigNe_c_f)
    edge_df = _build_flat_dataframe(
        'edge', t_e_f, r_edge_f, z_edge, rhop_edge_f, r_midplane_edge,
        Te_e_f, SigTe_e_f, Ne_e_f, SigNe_e_f)

    all_df = pd.concat([core_df, edge_df], ignore_index=True)
    if not apply_filters:
        return all_df
    all_df, metadata = _apply_preprocessing_filters(
        all_df,
        outlier_config=outlier_config,
        local_statistical_outlier_config=local_statistical_outlier_config,
        rhop_filter=rhop_filter,
        channel_trim=channel_trim,
        report_outliers=report_outliers,
    )
    if return_metadata:
        return all_df, metadata
    return all_df


def preprocess_ts_data(
    ts_dict, interval=None, r_edge_offset=0.008, mapping_method='sfutils',
    outlier_config=None, rhop_filter=None, channel_trim=None,
    report_outliers=False, return_metadata=False, apply_filters=True,
    local_statistical_outlier_config=None):
    """Preprocess VTA TS data and produce a DataFrame with R_midplane coordinates."""
    if (interval is not None and apply_filters
            and local_statistical_outlier_config
            and local_statistical_outlier_config.get('enabled', True)):
        mapped_shot = convert_vta_to_r_midplane(
            ts_dict,
            interval=None,
            r_edge_offset=r_edge_offset,
            mapping_method=mapping_method,
            apply_filters=False,
        )
        mapped_shot, statistical_metadata = (
            apply_shot_wide_statistical_outlier_filter(
                mapped_shot,
                config=local_statistical_outlier_config,
                rhop_filter=rhop_filter,
                channel_trim=channel_trim,
                report=report_outliers,
                return_metadata=True,
            )
        )
        interval_mask = _interval_mask(
            mapped_shot['time'].to_numpy(dtype=float), interval)
        interval_df = mapped_shot.loc[interval_mask].copy().reset_index(drop=True)
        removed_rows = statistical_metadata['removed_rows']
        removed_rows = removed_rows.loc[_interval_mask(
            removed_rows['time'].to_numpy(dtype=float), interval)].copy()
        interval_before_statistical = pd.concat(
            [interval_df, removed_rows], ignore_index=True)
        filtered_interval, metadata = _apply_preprocessing_filters(
            interval_df,
            outlier_config=outlier_config,
            rhop_filter=rhop_filter,
            channel_trim=channel_trim,
            report_outliers=report_outliers,
            local_statistical_outlier_config=None,
        )
        metadata['channels_before'] = {
            system: int(interval_before_statistical.loc[
                interval_before_statistical['system'].eq(system),
                'channel_index',
            ].nunique())
            for system in ('core', 'edge')
        }
        statistical_metadata = dict(statistical_metadata)
        statistical_metadata['removed_rows'] = removed_rows
        if 'statistical_removed_rows' in statistical_metadata:
            statistical_removed_rows = statistical_metadata[
                'statistical_removed_rows']
            statistical_metadata['statistical_removed_rows'] = (
                statistical_removed_rows.loc[_interval_mask(
                    statistical_removed_rows['time'].to_numpy(dtype=float),
                    interval,
                )].copy()
            )
        statistical_metadata['removed_rows_scope'] = 'interval'
        metadata['local_statistical_outlier_metadata'] = statistical_metadata
        if not removed_rows.empty:
            metadata['removed_rows'] = pd.concat(
                [metadata['removed_rows'], removed_rows], ignore_index=True)
        if return_metadata:
            return filtered_interval, metadata
        return filtered_interval

    return convert_vta_to_r_midplane(
        ts_dict,
        interval=interval,
        r_edge_offset=r_edge_offset,
        mapping_method=mapping_method,
        outlier_config=outlier_config,
        local_statistical_outlier_config=local_statistical_outlier_config,
        rhop_filter=rhop_filter,
        channel_trim=channel_trim,
        report_outliers=report_outliers,
        return_metadata=return_metadata,
        apply_filters=apply_filters)


def apply_preprocessing_filters(
    ts_df, outlier_config=None, rhop_filter=None, channel_trim=None,
    report_outliers=False, local_statistical_outlier_config=None):
    """Apply interval-level filters to an already mapped TS DataFrame."""
    return _apply_preprocessing_filters(
        ts_df,
        outlier_config=outlier_config,
        local_statistical_outlier_config=local_statistical_outlier_config,
        rhop_filter=rhop_filter,
        channel_trim=channel_trim,
        report_outliers=report_outliers,
    )


def map_to_r_midplane(ts_df):
    """Sort data by midplane radius for fitting."""
    df = ts_df.copy()
    if 'r_midplane' not in df.columns:
        raise ValueError('Missing r_midplane values for TS data.')
    return df.sort_values('r_midplane')


def augment_measurement_errors(ts_df, quantity='Te', fixed_error=None,
                               relative_error=None, error_column=None,
                               output_column=None, inplace=False):
    """Add independent fixed and relative uncertainty contributions.

    The returned error is computed in the same units as ``quantity`` using
    quadrature:

    ``sqrt(original_error**2 + fixed_error**2 + (relative_error * abs(value))**2)``

    ``relative_error`` is a fraction, so ``0.05`` means a 5 percent
    contribution. The original error column is preserved unless
    ``output_column`` explicitly names that same column and ``inplace`` is
    true.
    """
    if not isinstance(ts_df, pd.DataFrame):
        raise TypeError('ts_df must be a pandas DataFrame.')

    if quantity not in ts_df.columns:
        raise ValueError(f"Missing quantity column '{quantity}'.")

    if error_column is None:
        error_column = f'{quantity}_err'
    if error_column not in ts_df.columns:
        raise ValueError(f"Missing error column '{error_column}'.")

    if fixed_error is None:
        fixed_error = 0.0
    if relative_error is None:
        relative_error = 0.0

    fixed_error = float(fixed_error)
    relative_error = float(relative_error)
    if not np.isfinite(fixed_error) or fixed_error < 0:
        raise ValueError('fixed_error must be finite and nonnegative.')
    if not np.isfinite(relative_error) or relative_error < 0:
        raise ValueError('relative_error must be finite and nonnegative.')

    if output_column is None:
        output_column = f'{error_column}_augmented'
    if not isinstance(output_column, str) or not output_column:
        raise ValueError('output_column must be a non-empty string.')
    if output_column == error_column and not inplace:
        raise ValueError(
            'Cannot overwrite the source error column when inplace=False.')

    values = ts_df[quantity].to_numpy(dtype=float)
    original_error = ts_df[error_column].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Quantity column '{quantity}' must contain finite values.")
    if np.any(~np.isfinite(original_error)) or np.any(original_error < 0):
        raise ValueError(
            f"Error column '{error_column}' must contain finite nonnegative values.")

    augmented_error = np.sqrt(
        original_error ** 2
        + fixed_error ** 2
        + (relative_error * np.abs(values)) ** 2
    )

    result = ts_df if inplace else ts_df.copy()
    result[output_column] = augmented_error
    return result


def filter_elm_cycles(ts_df, elm_times_begin, elm_times_end,
                      cycle_start_fraction=0.75,
                      cycle_end_fraction=0.90,
                      return_metadata=False):
    """Keep the late portion of each inter-ELM cycle.

    For each consecutive pair of ELM events, the retained interval runs from
    ``cycle_start_fraction`` to ``cycle_end_fraction`` of the time between
    the first ELM end and the next ELM start. Core and edge rows are masked
    independently using their shared ``time`` column.
    """
    if 'time' not in ts_df.columns:
        raise ValueError('Missing time values for ELM cycle filtering.')

    begins = np.asarray(elm_times_begin, dtype=float).ravel()
    ends = np.asarray(elm_times_end, dtype=float).ravel()
    if begins.size != ends.size:
        raise ValueError('ELM start and end arrays must have the same length.')
    if not 0 <= cycle_start_fraction < cycle_end_fraction <= 1:
        raise ValueError(
            'ELM cycle fractions must satisfy 0 <= start < end <= 1.')
    if np.any(~np.isfinite(begins)) or np.any(~np.isfinite(ends)):
        raise ValueError('ELM start and end arrays must contain finite values.')
    if np.any(ends < begins):
        raise ValueError('Each ELM end time must be greater than its start time.')

    time = ts_df['time'].to_numpy(dtype=float)
    keep_mask = np.zeros(len(ts_df), dtype=bool)
    cycle_windows = []
    for end, next_begin in zip(ends[:-1], begins[1:]):
        gap = next_begin - end
        if gap <= 0:
            continue
        cycle_begin = end + cycle_start_fraction * gap
        cycle_end = end + cycle_end_fraction * gap
        cycle_windows.append((cycle_begin, cycle_end))
        keep_mask |= (time >= cycle_begin) & (time <= cycle_end)

    filtered = ts_df.loc[keep_mask].copy()
    if return_metadata:
        return filtered, {
            'rows_removed': int(np.sum(~keep_mask)),
            'rows_before': int(len(ts_df)),
            'rows_after': int(len(filtered)),
            'cycle_windows': [
                {'t_begin': float(begin), 't_end': float(end)}
                for begin, end in cycle_windows
            ],
            'cycle_start_fraction': float(cycle_start_fraction),
            'cycle_end_fraction': float(cycle_end_fraction),
        }
    return filtered
