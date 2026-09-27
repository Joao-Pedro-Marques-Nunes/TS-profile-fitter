import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ts_profile_fitter.preprocess import (
    _apply_channel_trim,
    apply_cold_dense_edge_filters,
    apply_distance_outlier_filters,
    compute_rhop_matrix,
    compute_r_midplane,
    filter_elm_cycles,
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
