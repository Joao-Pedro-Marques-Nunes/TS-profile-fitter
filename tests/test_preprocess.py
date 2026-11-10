import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ts_profile_fitter.preprocess import (
    _apply_channel_trim,
    apply_cold_dense_edge_filters,
    apply_density_outside_separatrix_filter,
    apply_distance_outlier_filters,
    apply_preprocessing_filters,
    compute_rhop_matrix,
    compute_r_midplane,
    filter_elm_cycles,
    preprocess_ts_data,
)


def test_cold_dense_edge_filter_removes_only_combined_condition():
    data = pd.DataFrame({
        'system': ['edge', 'edge', 'core', 'edge'],
        'Te': [1.0, 1.0, 1.0, 2.0],
        'Ne': [2.1e19, 1.9e19, 3e19, 3e19],
    })

    filtered, metadata = apply_cold_dense_edge_filters(
        data,
        {'enabled': True, 'te_max': 1.1, 'ne_min': 2e19},
        return_metadata=True,
    )

    assert filtered.index.tolist() == [1, 2, 3]
    assert metadata['filter_counts'] == {'edge:TeNe': 1}
    assert metadata['removed_rows'].index.tolist() == [0]
    assert metadata['removed_rows'].iloc[0]['filter_reason'] == (
        'cold_dense_edge_outlier')


def test_density_outside_separatrix_filter_uses_pooled_median_and_strict_cuts():
    data = pd.DataFrame({
        'system': ['core', 'core', 'edge', 'edge'],
        'Ne': [2e19, 3e19, 5e19, 20e19],
        'rho_pol': [1.2, 1.2, 1.2, 1.0],
    })

    filtered, metadata = apply_density_outside_separatrix_filter(
        data, {'enabled': True}, return_metadata=True)

    assert filtered.index.tolist() == [0, 3]
    assert metadata['density_median'] == 4e19
    assert metadata['density_threshold'] == 2e19
    assert metadata['filter_counts'] == {
        'high_density_outside_separatrix': 2}
    assert metadata['removed_rows']['filter_reason'].tolist() == [
        'high_density_outside_separatrix',
        'high_density_outside_separatrix',
    ]


def test_distance_outlier_filter_removes_isolated_low_error_point():
    times = np.arange(7, dtype=float)
    data = pd.DataFrame({
        'system': ['edge'] * 7,
        'channel_index': [1] * 7,
        'time': times,
        'r_midplane': [2.2] * 7,
        'Ne': [2e19, 2.1e19, 8.9e19, 2.0e19, 2.2e19, 2.1e19, 2e19],
        'Ne_err': [1e18] * 7,
        'Te': [20.0] * 7,
        'Te_err': [2.0] * 7,
    })
    data.loc[2, 'Ne_err'] = 2e17

    filtered, metadata = apply_distance_outlier_filters(
        data,
        {'Ne_edge': {
            'n_neighbors': 5,
            'min_points': 6,
            'min_value': 5e19,
            'max_relative_error': 0.05,
            'zscore': 8,
            'min_scale': 1e17,
        }},
        return_metadata=True,
    )

    assert len(filtered) == 6
    assert metadata['filter_counts'] == {'edge:Ne': 1}
    assert 'Ne_distance_outlier' in metadata['removed_rows'].iloc[0]['filter_reason']


def test_distance_outlier_filter_handles_short_temporal_group():
    data = pd.DataFrame({
        'system': ['edge'] * 4,
        'channel_index': [1] * 4,
        'time': np.arange(4, dtype=float),
        'r_midplane': [2.2] * 4,
        'Ne': [2.0e19, 2.1e19, 8.9e19, 2.0e19],
        'Ne_err': [1.0e18, 1.0e18, 2.0e17, 1.0e18],
        'Te': [20.0] * 4,
        'Te_err': [2.0] * 4,
    })

    filtered, metadata = apply_distance_outlier_filters(
        data,
        {'Ne_edge': {
            'n_neighbors': 5,
            'min_points': 4,
            'min_value': 5e19,
            'max_relative_error': 0.05,
            'zscore': 5,
            'min_scale': 1e17,
        }},
        return_metadata=True,
    )

    assert len(filtered) == 3
    assert metadata['filter_counts'] == {'edge:Ne': 1}


def test_distance_outlier_filter_uses_temperature_floor_for_short_group():
    data = pd.DataFrame({
        'system': ['edge'] * 5,
        'channel_index': [1] * 5,
        'time': np.arange(5, dtype=float),
        'r_midplane': [2.2] * 5,
        'Ne': [3.3e19, 7.1e19, 1.0e19, 7.3e19, 0.4e19],
        'Ne_err': [3e17, 2e17, 2e18, 2e17, 1e18],
        'Te': [1.0005, 1.0005, 10.0, 1.0005, 12.0],
        'Te_err': [0.01, 0.01, 2.0, 0.01, 2.4],
    })

    filtered, metadata = apply_distance_outlier_filters(
        data,
        {'Ne_edge': {
            'n_neighbors': 5,
            'min_points': 3,
            'min_value': 2e19,
            'max_relative_error': 0.05,
            'zscore': 5,
            'min_scale': 1e18,
            'short_group_max_points': 5,
            'te_floor': 1.1,
            'te_max_relative_error': 0.05,
        }},
        return_metadata=True,
    )

    assert len(filtered) == 5
    assert metadata['filter_counts'] == {}


def test_heuristic_outlier_reasons_distinguish_zero_and_negative_inputs():
    data = pd.DataFrame({
        'system': ['core'] * 4,
        'channel_index': range(4),
        'Te': [0.0, -1.0, 10.0, 10.0],
        'Te_err': [1.0, 1.0, -1.0, 1.0],
        'Ne': [1.0] * 4,
        'Ne_err': [0.1] * 4,
    })

    _, metadata = apply_preprocessing_filters(
        data,
        outlier_config={
            'Te': {'remove_zeros': True, 'value_min': 0, 'error_min': 0},
        },
    )

    reasons = metadata['removed_rows'].set_index('channel_index')['filter_reason']
    assert 'Te_zero_value' in reasons.loc[0]
    assert 'Te_negative_value' in reasons.loc[1]
    assert 'Te_negative_error' in reasons.loc[2]


def _make_local_statistical_outlier_grid():
    rows = []
    row_by_point = {}
    for system in ('core', 'edge'):
        base_radius = 1.0 if system == 'core' else 2.0
        for time_index in range(7):
            for channel_index in range(7):
                temperature = 100.0 + 15.0 * channel_index + 2.0 * time_index
                density = 1e19 * (
                    1.0
                    + (0.1 if system == 'core' else 0.35) * channel_index
                    + 0.01 * time_index
                )
                row_by_point[(system, time_index, channel_index)] = len(rows)
                rows.append({
                    'shot': 123,
                    'system': system,
                    'channel_index': channel_index,
                    'time': 0.1 + 0.01 * time_index,
                    'r_midplane': base_radius + 0.005 * channel_index,
                    'Te': temperature,
                    'Te_err': 0.02 * temperature,
                    'Ne': density,
                    'Ne_err': 0.02 * density,
                })
    return pd.DataFrame(rows), row_by_point


def test_local_statistical_outlier_filter_uses_2d_support_and_both_criteria():
    data, row_by_point = _make_local_statistical_outlier_grid()
    expected_removed = {
        row_by_point[('core', 3, 3)],
        row_by_point[('edge', 3, 3)],
        row_by_point[('core', 1, 5)],
        row_by_point[('edge', 5, 1)],
    }
    data.loc[row_by_point[('core', 3, 3)], 'Te'] = 10000.0
    data.loc[row_by_point[('edge', 3, 3)], 'Ne'] = 5e20
    core_relative_error_row = row_by_point[('core', 1, 5)]
    data.loc[core_relative_error_row, 'Ne_err'] = (
        0.5 * data.loc[core_relative_error_row, 'Ne'])
    edge_relative_error_row = row_by_point[('edge', 5, 1)]
    data.loc[edge_relative_error_row, 'Te_err'] = (
        0.5 * data.loc[edge_relative_error_row, 'Te'])

    filtered, metadata = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config={
            'enabled': True,
            'threshold': 3.5,
            'radial_half_width_m': {'core': 0.011, 'edge': 0.006},
            'time_half_width_s': 0.011,
            'min_neighbors': 6,
            'min_radial_neighbors': 2,
            'min_temporal_neighbors': 2,
        },
    )

    statistical_metadata = metadata['local_statistical_outlier_metadata']
    assert set(statistical_metadata['removed_rows'].index) == expected_removed
    assert len(filtered) == len(data) - 4
    assert statistical_metadata['filter_counts'] == {
        'core:Te': 1,
        'core:Ne': 1,
        'edge:Te': 1,
        'edge:Ne': 1,
    }
    assert statistical_metadata['criteria_counts']['core:Te']['upper_value'] == 1
    assert statistical_metadata['criteria_counts']['edge:Ne']['upper_value'] == 1
    assert statistical_metadata['criteria_counts']['core:Ne']['relative_error'] == 1
    assert statistical_metadata['criteria_counts']['edge:Te']['relative_error'] == 1
    assert statistical_metadata['fallback_neighbor_counts'] == {}


def test_local_statistical_filter_accepts_separate_value_and_error_thresholds():
    data, row_by_point = _make_local_statistical_outlier_grid()
    high_value_row = row_by_point[('edge', 3, 3)]
    high_relative_error_row = row_by_point[('core', 1, 5)]
    data.loc[high_value_row, 'Ne'] = 5e20
    data.loc[high_relative_error_row, 'Ne_err'] = (
        0.5 * data.loc[high_relative_error_row, 'Ne'])

    relative_error_by_channel = {
        channel_index: 0.02 + 0.01 * channel_index
        for channel_index in range(7)
    }
    core_rows = data['system'].eq('core')
    data.loc[core_rows, 'Ne_err'] = (
        data.loc[core_rows, 'Ne']
        * data.loc[core_rows, 'channel_index'].map(relative_error_by_channel)
    )
    data.loc[high_relative_error_row, 'Ne_err'] = (
        0.5 * data.loc[high_relative_error_row, 'Ne'])

    filtered, metadata = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config={
            'enabled': True,
            'upper_value_threshold': 3.5,
            'relative_error_threshold': 100.0,
            'radial_half_width_m': {'core': 0.011, 'edge': 0.006},
            'time_half_width_s': 0.011,
            'min_neighbors': 6,
            'min_radial_neighbors': 2,
            'min_temporal_neighbors': 2,
        },
    )

    statistical_metadata = metadata['local_statistical_outlier_metadata']
    removed_indices = set(statistical_metadata['removed_rows'].index)
    assert high_value_row in removed_indices
    assert high_relative_error_row not in removed_indices
    assert statistical_metadata['criteria_counts']['edge:Ne']['upper_value'] == 1
    assert statistical_metadata['criteria_counts']['core:Ne']['relative_error'] == 0
    assert len(filtered) == len(data) - len(removed_indices)


def test_invalid_measurements_are_removed_before_statistical_filtering():
    data, row_by_point = _make_local_statistical_outlier_grid()
    target_index = row_by_point[('core', 3, 3)]
    zero_value_index = row_by_point[('core', 2, 2)]
    negative_value_index = row_by_point[('edge', 2, 2)]
    zero_error_index = row_by_point[('core', 4, 2)]
    negative_error_index = row_by_point[('edge', 4, 2)]
    data.loc[target_index, 'Te'] = 10000.0
    data.loc[zero_value_index, 'Te'] = 0.0
    data.loc[negative_value_index, 'Ne'] = -1e19
    data.loc[zero_error_index, 'Te_err'] = 0.0
    data.loc[negative_error_index, 'Ne_err'] = -1.0

    filtered, metadata = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config={
            'enabled': True,
            'threshold': 3.5,
            'radial_half_width_m': {'core': 0.011, 'edge': 0.006},
            'time_half_width_s': 0.011,
            'min_neighbors': 6,
            'min_radial_neighbors': 2,
            'min_temporal_neighbors': 2,
        },
    )

    invalid_indices = {
        zero_value_index,
        negative_value_index,
        zero_error_index,
        negative_error_index,
    }
    assert invalid_indices.isdisjoint(filtered.index)
    validity_removed = metadata['measurement_validity_metadata'][
        'removed_rows']
    validity_reasons = validity_removed['filter_reason']
    assert 'Te_zero_value' in validity_reasons.loc[zero_value_index]
    assert 'Ne_negative_value' in validity_reasons.loc[negative_value_index]
    assert 'Te_zero_error' in validity_reasons.loc[zero_error_index]
    assert 'Ne_negative_error' in validity_reasons.loc[negative_error_index]
    statistical_removed = metadata['local_statistical_outlier_metadata'][
        'removed_rows']
    assert target_index in statistical_removed.index
    assert invalid_indices.isdisjoint(statistical_removed.index)


def test_local_statistical_outlier_filter_falls_back_to_nearest_channel():
    rows = []
    target_index = None
    for time_index in range(9):
        time = 0.1 + 0.01 * time_index
        for channel_index, (r_midplane, rho_pol, temperature) in enumerate((
                (2.0, 0.98, 100.0),
            (2.03, 1.01, 90.0),
            (2.02, 0.99, 95.0),
                (2.04, 1.02, 110.0),
        )):
            if channel_index == 0 and time_index == 4:
                target_index = len(rows)
                temperature = 10000.0
            rows.append({
                'system': 'edge',
                'channel_index': channel_index,
                'time': time,
                'r_midplane': r_midplane,
                'rho_pol': rho_pol,
                'Te': temperature,
                'Te_err': 0.02 * temperature,
                'Ne': 1e19 * (1.0 - 0.1 * channel_index),
                'Ne_err': 2e17,
            })
    data = pd.DataFrame(rows)
    config = {
        'enabled': True,
        'threshold': 3.5,
        'radial_half_width_m': {'core': 0.005, 'edge': 0.005},
        'time_half_width_s': 0.05,
        'min_neighbors': 30,
        'min_radial_neighbors': 3,
        'min_temporal_neighbors': 2,
    }

    without_fallback, _ = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config={
            **config,
            'fallback_to_nearest_channel': False,
        },
    )
    with_fallback, metadata = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config=config,
    )

    assert target_index in without_fallback.index
    assert target_index not in with_fallback.index
    assert metadata['local_statistical_outlier_metadata'][
        'fallback_neighbor_counts']['edge'] > 0
    assert metadata['local_statistical_outlier_metadata'][
        'insufficient_support_counts']['edge:Te']['upper_value'] == 0


def test_interval_preprocessing_uses_full_shot_statistical_neighbors(monkeypatch):
    rows = []
    for time_index in range(9):
        time = 0.1 + 0.01 * time_index
        for channel_index, (r_midplane, rho_pol, temperature) in enumerate((
                (2.0, 0.98, 100.0),
            (2.03, 1.01, 100.0),
        )):
            if channel_index == 0 and time_index == 4:
                temperature = 10000.0
            elif channel_index == 1 and time_index == 4:
                temperature = 0.0
            rows.append({
                'system': 'edge',
                'channel_index': channel_index,
                'time': time,
                'r_midplane': r_midplane,
                'rho_pol': rho_pol,
                'Te': temperature,
                'Te_err': 0.02 * temperature,
                'Ne': 1e19,
                'Ne_err': 2e17,
            })
    mapped_shot = pd.DataFrame(rows)
    mapping_calls = []

    def mapped_data_for_shot(ts_dict, **kwargs):
        mapping_calls.append(kwargs)
        return mapped_shot.copy()

    monkeypatch.setattr(
        'ts_profile_fitter.preprocess.convert_vta_to_r_midplane',
        mapped_data_for_shot,
    )
    filtered, metadata = preprocess_ts_data(
        {},
        interval={'t_begin': 0.14, 't_end': 0.14},
        local_statistical_outlier_config={
            'enabled': True,
            'threshold': 3.5,
            'radial_half_width_m': {'core': 0.005, 'edge': 0.005},
            'time_half_width_s': 0.05,
            'min_neighbors': 6,
            'min_radial_neighbors': 2,
            'min_temporal_neighbors': 2,
        },
        return_metadata=True,
    )

    assert mapping_calls[0]['interval'] is None
    assert mapping_calls[0]['apply_filters'] is False
    assert not (
        filtered['time'].eq(0.14)
        & filtered['channel_index'].eq(0)
    ).any()
    statistical_metadata = metadata['local_statistical_outlier_metadata']
    assert statistical_metadata['scope'] == 'shot'
    assert statistical_metadata['removed_rows_scope'] == 'interval'
    assert metadata['channels_before']['edge'] == 2
    statistical_only_rows = statistical_metadata['statistical_removed_rows']
    assert statistical_only_rows[
        ['time', 'channel_index']].values.tolist() == [[0.14, 0]]
    interval_removed = statistical_metadata['removed_rows'].loc[
        statistical_metadata['removed_rows']['time'].eq(0.14)
    ].set_index('channel_index')['filter_reason']
    assert 'Te_statistical_upper_value_outlier' in interval_removed.loc[0]
    assert 'Te_zero_value' in interval_removed.loc[1]


def test_local_statistical_outlier_filter_keeps_low_values():
    data, row_by_point = _make_local_statistical_outlier_grid()
    low_value_row = row_by_point[('core', 3, 3)]
    data.loc[low_value_row, 'Te'] = 1.0
    data.loc[low_value_row, 'Te_err'] = 0.02

    filtered, metadata = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config={
            'enabled': True,
            'threshold': 3.5,
            'radial_half_width_m': {'core': 0.011, 'edge': 0.006},
            'time_half_width_s': 0.011,
            'min_neighbors': 6,
            'min_radial_neighbors': 2,
            'min_temporal_neighbors': 2,
        },
    )

    assert low_value_row in filtered.index
    assert metadata['local_statistical_outlier_metadata']['criteria_counts'][
        'core:Te']['upper_value'] == 0


def test_local_statistical_outlier_filter_skips_insufficient_radial_support():
    data = pd.DataFrame({
        'system': ['edge'] * 5,
        'channel_index': [1] * 5,
        'time': 0.1 + 0.01 * np.arange(5),
        'r_midplane': [2.2] * 5,
        'Te': [100.0, 110.0, 10000.0, 105.0, 100.0],
        'Te_err': [2.0, 2.2, 200.0, 2.1, 2.0],
        'Ne': [1e19, 1.1e19, 1e22, 1.05e19, 1e19],
        'Ne_err': [2e17, 2.2e17, 2e20, 2.1e17, 2e17],
    })

    filtered, metadata = apply_preprocessing_filters(
        data,
        local_statistical_outlier_config={
            'enabled': True,
            'radial_half_width_m': {'core': 0.02, 'edge': 0.02},
            'time_half_width_s': 0.025,
            'min_neighbors': 3,
            'min_radial_neighbors': 2,
            'min_temporal_neighbors': 2,
        },
    )

    statistical_metadata = metadata['local_statistical_outlier_metadata']
    assert filtered.index.tolist() == data.index.tolist()
    assert statistical_metadata['filter_counts'] == {}
    assert statistical_metadata['insufficient_support_counts']['edge:Ne'][
        'upper_value'] > 0
    target_support = statistical_metadata['insufficient_support_rows'].loc[
        lambda rows: rows['row_index'].eq(2)
        & rows['quantity'].eq('Ne')
        & rows['metric'].eq('upper_value')
    ]
    assert len(target_support) == 1
    assert target_support.iloc[0]['reason'] == 'min_radial_neighbors'
    assert target_support.iloc[0]['radial_support'] == 1


def test_channel_trim_uses_spatial_channel_order():
    rows = []
    for system, positions in [('core', [1.4, 1.1, 1.3, 1.2]), ('edge', [2.0, 2.1])]:
        for channel_index, r_midplane in enumerate(positions):
            rows.append({
                'system': system,
                'channel_index': channel_index,
                'r_midplane': r_midplane,
            })
    data = pd.DataFrame(rows)

    trimmed, metadata = _apply_channel_trim(
        data,
        channel_trim={'core': {'first': 2, 'last': 1}},
        return_metadata=True,
    )

    assert trimmed.loc[trimmed['system'].eq('core'), 'channel_index'].tolist() == [2]
    assert metadata['channels_removed'] == {'core': 3, 'edge': 0}


def test_filter_elm_cycles_keeps_legacy_inter_elm_window():
    data = pd.DataFrame({'time': np.arange(11, dtype=float), 'value': np.arange(11)})

    filtered, metadata = filter_elm_cycles(
        data,
        elm_times_begin=[1.0, 7.0],
        elm_times_end=[2.0, 8.0],
        return_metadata=True,
    )

    assert filtered['time'].tolist() == [6.0]
    assert metadata['rows_removed'] == 10
    assert metadata['rows_after'] == 1


class DummyEquilibrium:
    def __init__(self):
        self.Rmesh = np.array([0.8, 0.9, 1.0], dtype=float)
        self.Zmesh = np.array([-0.2, 0.0, 0.2], dtype=float)
        self.time = np.array([0.0, 1.0], dtype=float)
        self.pfm = np.zeros((2, 3, 3), dtype=float)
        self.psi0 = np.zeros(2, dtype=float)
        self.psix = np.ones(2, dtype=float)
        for it in range(2):
            for iz in range(3):
                for ir in range(3):
                    self.pfm[it, iz, ir] = 1.0 + 0.1 * ir + 0.05 * iz + 0.01 * it

    def rz2rho(self, r, z, t_in, coord_out='rho_pol', extrapolate=True):
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        return r + z * 0.0


def test_sfutils_mapping_broadcasts_time_dependent_radius_over_channels():
    eq = DummyEquilibrium()
    time = np.array([0.0, 1.0, 2.0], dtype=float)
    r = np.array([0.85, 0.86, 0.87], dtype=float)
    z = np.array([-0.1, 0.0], dtype=float)

    rho = compute_rhop_matrix(time, r, z, eq, mapping_method='sfutils')

    assert rho.shape == (3, 2)
    np.testing.assert_allclose(rho[:, 0], r)
    np.testing.assert_allclose(rho[:, 1], r)


def test_ipfnpytools_mapping_returns_finite_2d_arrays():
    eq = DummyEquilibrium()
    time = np.array([0.0, 1.0], dtype=float)
    r = np.array([0.85, 0.95], dtype=float)
    z = np.array([0.0, 0.0], dtype=float)

    rho = compute_rhop_matrix(time, r, z, eq, mapping_method='ipfnpytools')
    assert rho.shape == (2, 2)
    assert np.isfinite(rho).all()

    z_mag = np.zeros_like(time)
    r_midplane = compute_r_midplane(time, rho, z_mag, eq, mapping_method='ipfnpytools')
    assert r_midplane.shape == rho.shape
    assert np.isfinite(r_midplane).all()
