import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ts_profile_fitter.fitting import (
    compute_channel_knots,
    count_modified_tanh_parameters,
    fit_ipfn_spline_ne_Te_TS,
    fit_modified_tanh_profile,
    fit_mbsplt_ne_Te_TS,
    modified_tanh,
)
from ts_profile_fitter.separatrix import make_profile


def test_fit_modified_tanh_profile_returns_finite_params():
    x0_true = 0.2
    w_true = 0.4
    tail_true = np.array([0.5, 0.15])
    top_true = np.array([1.8, -0.1])

    x = np.linspace(-1.5, 1.5, 41)
    dx = x - x0_true
    tanh_term = np.tanh(dx / w_true)
    tail = tail_true[0] + tail_true[1] * dx
    top = top_true[0] + top_true[1] * dx
    y = 0.5 * (1 + tanh_term) * tail + 0.5 * (1 - tanh_term) * top

    x_core = x[x < 0.0]
    x_edge = x[x >= 0.0]
    y_core = y[x < 0.0]
    y_edge = y[x >= 0.0]

    params, _ = fit_modified_tanh_profile(
        np.concatenate([x_core, x_edge]),
        np.concatenate([y_core, y_edge]),
        np.concatenate([np.ones_like(y_core) * 0.01, np.ones_like(y_edge) * 0.01]),
        deg_tail=1,
        deg_top=1,
    )

    assert np.isfinite(params).all()
    assert params.shape[0] == 6


def test_count_modified_tanh_parameters_matches_constraints():
    assert count_modified_tanh_parameters(deg_tail=1, deg_top=3) == 8
    assert count_modified_tanh_parameters(
        deg_tail=1,
        deg_top=3,
        fix_a_tail=0,
    ) == 7
    assert count_modified_tanh_parameters(
        deg_tail=1,
        deg_top=3,
        fix_x0=2.135,
        fix_w=0.15,
        fix_a_tail=0,
    ) == 5


def test_make_profile_mtanh_derivatives_match_profile_grid():
    R = np.linspace(1.8, 2.2, 50)
    Te = modified_tanh(R, 2.0, 0.05, 0.2, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0)
    Ne = np.linspace(1e19, 1e18, R.size)

    profile = make_profile(R, Te, Ne, fit_method='mtanh')

    assert profile['dTe_dR'].shape == R.shape
    assert profile['dNe_dR'].shape == R.shape
    assert np.isfinite(profile['dTe_dR']).all()
    assert np.isfinite(profile['dNe_dR']).all()

def test_fit_mbsplt_ne_Te_TS_returns_spline_outputs():
    ds = pd.DataFrame({
        'R_mid': np.array([0.8, 0.9, 1.0, 1.2, 1.4]),
        'ne': np.array([1.0, 0.7, 0.45, 0.2, 0.08]),
        'ne_err': np.array([0.05, 0.04, 0.03, 0.02, 0.01]),
        'Te': np.array([3.0, 2.2, 1.5, 0.9, 0.4]),
        'Te_err': np.array([0.1, 0.08, 0.06, 0.04, 0.02]),
    })

    ds_out, xt, p_ne, pcov_ne, p_Te, pcov_Te = fit_mbsplt_ne_Te_TS(
        ds,
        dR_fine=0.05,
        sigma_conv=0.0,
        R_knots=np.array([0.8, 0.95, 1.1, 1.4]),
        max_step_scale=None,
        use_errors=False,
        loss='linear',
    )

    assert isinstance(ds_out, pd.DataFrame)
    assert 'R_mid_spl' in ds_out.columns
    assert 'ne_spl' in ds_out.columns
    assert 'Te_spl' in ds_out.columns
    assert np.isfinite(p_ne).all()
    assert np.isfinite(p_Te).all()
    assert len(xt) == 4
    assert pcov_ne is not None
    assert pcov_Te is not None


def test_fit_ipfn_spline_ne_Te_TS_returns_shared_profile_outputs():
    ds = pd.DataFrame({
        'R_mid': np.linspace(0.8, 1.4, 12),
        'ne': np.linspace(1.0, 0.1, 12),
        'ne_err': np.full(12, 0.03),
        'Te': np.linspace(3.0, 0.3, 12),
        'Te_err': np.full(12, 0.05),
    })

    ds_out, knots, p_ne, pcov_ne, p_Te, pcov_Te = fit_ipfn_spline_ne_Te_TS(
        ds,
        dR_fine=0.05,
        R_knots=np.array([0.8, 0.95, 1.1, 1.4]),
        maxiter=50,
    )

    assert isinstance(ds_out, pd.DataFrame)
    assert set([
        'R_mid_spl', 'ne_spl', 'dne_spl', 'lam_ne_spl',
        'Te_spl', 'dTe_spl', 'lam_Te_spl',
    ]).issubset(ds_out.columns)
    assert np.isfinite(ds_out['ne_spl']).all()
    assert np.isfinite(ds_out['Te_spl']).all()
    assert np.all(np.diff(ds_out['ne_spl']) <= 1e-8)
    assert np.all(np.diff(ds_out['Te_spl']) <= 1e-8)
    assert np.isfinite(p_ne).all()
    assert np.isfinite(p_Te).all()
    assert np.nanmax(ds_out['ne_spl']) < 2.0
    assert np.nanmax(ds_out['Te_spl']) < 4.0
    assert pcov_ne is None
    assert pcov_Te is None
    assert np.array_equal(knots, np.array([0.8, 0.95, 1.1, 1.4]))


def test_fit_ipfn_spline_preserves_interior_endpoint_strategy_knots():
    ds = pd.DataFrame({
        'R_mid': np.linspace(0.8, 1.4, 12),
        'ne': np.linspace(1.0, 0.1, 12),
        'ne_err': np.full(12, 0.03),
        'Te': np.linspace(3.0, 0.3, 12),
        'Te_err': np.full(12, 0.05),
    })
    knots = np.array([0.82, 0.95, 1.1, 1.38])

    ds_out, returned_knots, p_ne, _, p_Te, _ = fit_ipfn_spline_ne_Te_TS(
        ds,
        dR_fine=0.05,
        R_knots=knots,
        mono=None,
        maxiter=50,
    )

    expected_full_knots = np.r_[np.repeat(0.8, 4), knots, np.repeat(1.4, 4)]
    assert np.array_equal(returned_knots, knots)
    assert np.allclose(ds_out.attrs['t_full'], expected_full_knots)
    assert ds_out['R_mid_spl'].iloc[0] == 0.8
    assert ds_out['R_mid_spl'].iloc[-1] == 1.4
    assert len(p_ne) == len(knots) + 4
    assert len(p_Te) == len(knots) + 4


def test_compute_channel_knots_applies_count_shift():
    ds = pd.DataFrame({
        'R_mid': np.linspace(0.8, 1.4, 8),
        'system': ['core'] * 4 + ['edge'] * 4,
    })

    knots = compute_channel_knots(
        ds,
        channels_before={'core': 4, 'edge': 4},
        channels_removed={'core': 0, 'edge': 0},
        knot_count_shift=1,
    )

    assert knots.shape == (5,)
    assert np.allclose(knots, np.quantile(ds['R_mid'], np.linspace(0, 1, 5)))


def test_compute_channel_knots_edge_biased_uses_raw_edge_positions():
    ds = pd.DataFrame({
        'system': ['core'] * 12 + ['edge'] * 12,
        'channel_index': np.tile(np.repeat(np.arange(6), 2), 2),
        'rho_pol': np.repeat(np.r_[np.linspace(0.8, 1.3, 6), np.linspace(1.4, 1.9, 6)], 2),
        'R_mid': np.r_[
            np.repeat(np.linspace(0.8, 1.3, 6), 2) + np.tile([0.0, 0.02], 6),
            np.repeat(np.linspace(1.4, 1.9, 6), 2) + np.tile([0.0, 0.02], 6),
        ],
    })

    knots = compute_channel_knots(
        ds,
        channels_before={'core': 6, 'edge': 6},
        strategy='edge_biased_quantile',
    )

    core_positions = np.linspace(0.81, 1.31, 6)
    edge_positions = ds.loc[ds['system'].eq('edge'), 'R_mid'].to_numpy()
    expected = np.sort(np.r_[core_positions, np.quantile(edge_positions, [0, 0.5, 1])])
    assert knots.shape == (9,)
    assert np.allclose(knots, expected)

    shifted_knots = compute_channel_knots(
        ds,
        channels_before={'core': 6, 'edge': 6},
        knot_count_shift=1,
        strategy='edge_biased_quantile',
    )
    shifted_expected = np.sort(np.r_[
        core_positions,
        np.quantile(edge_positions, np.linspace(0, 1, 4)),
    ])
    assert shifted_knots.shape == (10,)
    assert np.allclose(shifted_knots, shifted_expected)


def test_compute_channel_knots_hybrid_uses_core_medians_and_edge_quantiles():
    ds = pd.DataFrame({
        'system': ['core'] * 6 + ['edge'] * 8,
        'channel_index': [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3],
        'rho_pol': [
            0.8, 0.8, 0.9, 0.9, 1.0, 1.0,
            1.01, 1.01, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3,
        ],
        'R_mid': [
            1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
            2.1, 2.11, 2.3, 2.31, 2.5, 2.51, 2.7, 2.71,
        ],
    })

    knots = compute_channel_knots(
        ds,
        channels_before={'core': 3, 'edge': 4},
        strategy='hybrid',
    )

    expected = np.array([1.1, 1.5, 1.9, 2.105, 2.705])
    assert np.allclose(knots, expected)


def test_compute_channel_knots_hybrid_shift_changes_edge_budget_only():
    ds = pd.DataFrame({
        'system': ['core'] * 6 + ['edge'] * 8,
        'channel_index': [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3],
        'rho_pol': [
            0.8, 0.8, 0.9, 0.9, 1.0, 1.0,
            1.01, 1.01, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3,
        ],
        'R_mid': [
            1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
            2.1, 2.11, 2.3, 2.31, 2.5, 2.51, 2.7, 2.71,
        ],
    })

    knots = compute_channel_knots(
        ds,
        channels_before={'core': 3, 'edge': 4},
        knot_count_shift=1,
        strategy='hybrid',
    )

    assert np.allclose(knots[:3], [1.1, 1.5, 1.9])
    assert np.allclose(knots[3:], [2.105, 2.405, 2.705])


def test_compute_channel_knots_hybrid_merges_close_core_channels_by_rhop(capsys):
    ds = pd.DataFrame({
        'system': ['core'] * 6 + ['edge'] * 8,
        'channel_index': [
            12, 12, 3, 3, 40, 40,
            0, 0, 1, 1, 2, 2, 3, 3,
        ],
        'rho_pol': [
            0.8, 0.8, 0.81, 0.81, 1.0, 1.0,
            1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4,
        ],
        'R_mid': [
            1.0, 1.0, 1.02, 1.02, 1.4, 1.4,
            1.5, 1.51, 1.7, 1.71, 1.9, 1.91, 2.1, 2.11,
        ],
    })

    knots = compute_channel_knots(
        ds,
        channels_before={'core': 3, 'edge': 4},
        strategy='hybrid',
    )

    assert np.allclose(knots, [1.01, 1.4, 1.505, 2.105])
    assert (
        'Merged core channels 12 and 3 '
        '(rho_pol gap=0.01, threshold=0.05)'
    ) in capsys.readouterr().out


def test_compute_channel_knots_shift_uses_strategy_specific_count_rules():
    ds = pd.DataFrame({
        'system': ['core'] * 6 + ['edge'] * 8,
        'channel_index': [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3],
        'rho_pol': [
            0.8, 0.8, 0.9, 0.9, 1.0, 1.0,
            1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4,
        ],
        'R_mid': [
            1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
            2.1, 2.11, 2.3, 2.31, 2.5, 2.51, 2.7, 2.71,
        ],
    })

    for strategy in (
        'global_quantile', 'balanced_quantile',
        'edge_biased_quantile', 'hybrid',
    ):
        knots = compute_channel_knots(
            ds,
            channels_before={'core': 3, 'edge': 4},
            strategy=strategy,
            knot_count_shift=1,
        )
        expected_counts = {
            'global_quantile': 5,
            'balanced_quantile': 5,
            'edge_biased_quantile': 6,
            'hybrid': 6,
        }
        assert knots.size == expected_counts[strategy], strategy


def test_compute_channel_knots_hybrid_uses_ceiling_for_odd_edge_count():
    ds = pd.DataFrame({
        'system': ['core'] * 6 + ['edge'] * 10,
        'channel_index': [
            0, 0, 1, 1, 2, 2,
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
        ],
        'rho_pol': [
            0.8, 0.8, 0.9, 0.9, 1.0, 1.0,
            1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5,
        ],
        'R_mid': [
            1.0, 1.02, 1.2, 1.22, 1.4, 1.42,
            1.6, 1.6, 1.8, 1.8, 2.0, 2.0, 2.2, 2.2, 2.4, 2.4,
        ],
    })

    knots = compute_channel_knots(
        ds,
        channels_before={'core': 3, 'edge': 5},
        strategy='hybrid',
    )

    assert np.allclose(knots, [1.01, 1.21, 1.41, 1.6, 2.0, 2.4])
