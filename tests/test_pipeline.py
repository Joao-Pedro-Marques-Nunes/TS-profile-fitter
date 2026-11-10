import json

import pandas as pd
import numpy as np
import pytest

from ts_profile_fitter.cache import PhysicsCache
from ts_profile_fitter.pipeline import (
    _failure_stage,
    _partial_result,
    _pedestal_profile_values,
    _result_summary,
    _stage_columns,
    _stage_status_fields,
    _write_cache_reference_row,
    enrich_database,
    run_database,
)
from ts_profile_fitter.loading import (
    _read_eqh_outer_midplane,
    _read_eqh_outer_midplane_fields,
)
from ts_profile_fitter.separatrix import (
    PowerBalanceUnavailableError,
    estimate_separatrix,
    make_profile,
)
from scripts.scan_spline_knots import _failed_row, _successful_row


def test_enrich_database_restores_analysis_keys_and_keeps_fit_results():
    intervals = pd.DataFrame({
        'shot_id': [34383],
        't_begin': [1.0],
        't_end': [1.1],
        'label': ['EDA H-mode'],
    })
    records = [{
        'shot_id': 34383,
        't_begin': 1.0,
        't_end': 1.1,
        'fit_status': 'ok',
        'spline_Te_sep': 25.0,
    }]

    enriched = enrich_database(intervals, records)

    assert enriched.loc[0, 'shot'] == 34383
    assert enriched.loc[0, 't_begin (s)'] == 1.0
    assert enriched.loc[0, 't_end (s)'] == 1.1
    assert enriched.loc[0, 'spline_Te_sep'] == 25.0
    assert {'shot_id', 't_begin', 't_end'}.isdisjoint(enriched.columns)


def test_enrich_database_restores_analysis_keys_when_records_are_empty():
    intervals = pd.DataFrame({
        'shot_id': [34383],
        't_begin': [1.0],
        't_end': [1.1],
    })

    enriched = enrich_database(intervals, [])

    assert {'shot', 't_begin (s)', 't_end (s)'}.issubset(enriched.columns)
    assert {'shot_id', 't_begin', 't_end'}.isdisjoint(enriched.columns)
    assert {'fit_status', 'fit_error'}.issubset(enriched.columns)


def test_disabled_physics_cache_writes_reference_csv(tmp_path):
    cache = PhysicsCache(
        tmp_path / 'cache',
        physics_inputs_enabled=False,
        on_new_cache=lambda kind, key, value: _write_cache_reference_row(
            tmp_path, kind, key, value),
    )
    physics_inputs = {
        'metadata': {'time': 1.05, 'f_ELM': 0.0, 'sources': {}},
        'power_balance': {'P_abs': 1000.0, 'P_rsep': 100.0, 'dWdt': 50.0},
        'geometry': {
            'R_out': 1.5,
            'B_pol': 0.2,
            'B_tor': 2.5,
            'Z_eff': 1.0,
        },
    }

    value, cache_hit = cache.physics_inputs(
        34383,
        {'t_begin': 1.0, 't_end': 1.1},
        'EQH',
        lambda: physics_inputs,
    )

    reference_path = tmp_path / 'db_Lgil2024_physics_input.csv'
    reference = pd.read_csv(reference_path)
    assert value == physics_inputs
    assert not cache_hit
    assert len(reference) == 1
    assert reference.loc[0, 'shot'] == 34383
    assert reference.loc[0, 'P_abs [W]'] == 1000.0


def test_physics_cache_version_skips_legacy_nan_geometry(tmp_path):
    cache = PhysicsCache(tmp_path / 'cache', physics_inputs_enabled=True)
    old_key = {
        'version': 'physics_inputs_v1',
        'shot_id': 34383,
        't_begin': 1.0,
        't_end': 1.1,
        'equilibrium': 'EQH',
    }
    cache._write(
        cache._path('physics', old_key),
        {'status': 'ok', 'value': {'geometry': {'R_out': float('nan')}}},
    )
    fresh_inputs = {'geometry': {'R_out': 1.5}}

    value, cache_hit = cache.physics_inputs(
        34383,
        {'t_begin': 1.0, 't_end': 1.1},
        'EQH',
        lambda: fresh_inputs,
    )

    assert value == fresh_inputs
    assert not cache_hit


def test_eqh_outer_midplane_helpers_pass_array_inputs():
    class ArrayOnlyEquilibrium:
        def rhoTheta2rz(self, rho, theta, t_in=None, coord_in='rho_pol'):
            assert isinstance(rho, np.ndarray)
            assert isinstance(theta, np.ndarray)
            assert rho.shape == (1,)
            assert theta.shape == (1,)
            return np.array([2.14]), np.array([0.03])

        def rz2brzt(self, r, z, t_in=None):
            assert isinstance(r, np.ndarray)
            assert isinstance(z, np.ndarray)
            return np.array([-0.1]), np.array([0.2]), np.array([-2.5])

    equilibrium = ArrayOnlyEquilibrium()

    assert _read_eqh_outer_midplane(equilibrium, 1.05) == pytest.approx(2.14)
    assert _read_eqh_outer_midplane_fields(equilibrium, 1.05) == pytest.approx(
        (2.14, np.hypot(0.1, 0.2), -2.5)
    )


def test_result_summary_includes_separatrix_pressure_and_lambdas():
    state = {
        'Rsep': 1.5,
        'Ne_sep': 2e19,
        'Te_sep': 100.0,
        'R_grid': [1.0, 2.0],
        'lam_Te_grid': [0.01, 0.03],
        'lam_q_grid': [0.002, 0.004],
        'lambda_p_sep': 0.015,
        'lambda_Ne_sep': 0.03,
    }

    summary = _result_summary({'physics': {'spline': state, 'mtanh': state}})

    assert summary['spline_p_sep'] == 2e19 * 100.0 * 1.602176634e-22
    assert summary['mtanh_p_sep'] == 2e19 * 100.0 * 1.602176634e-22
    assert summary['spline_lambda_Te_sep'] == pytest.approx(0.02)
    assert summary['spline_lambda_q_sep'] == pytest.approx(0.003)
    assert summary['spline_lambda_p_sep'] == 0.015
    assert summary['spline_lambda_Ne_sep'] == 0.03
    assert summary['mtanh_lambda_Te_sep'] == pytest.approx(0.02)
    assert summary['mtanh_lambda_q_sep'] == pytest.approx(0.003)
    assert summary['mtanh_lambda_p_sep'] == 0.015
    assert summary['mtanh_lambda_Ne_sep'] == 0.03


def test_pedestal_profile_values_evaluate_linear_profiles_at_radius():
    R = np.linspace(3.0, 4.0, 101)
    values = _pedestal_profile_values(
        R,
        Te=100.0 - 50.0 * (R - 3.0),
        Ne=4e19 - 2e19 * (R - 3.0),
        R_ped=3.5,
    )

    assert values['Te_ped'] == pytest.approx(75.0)
    assert values['Ne_ped'] == pytest.approx(3e19)
    assert values['p_ped'] == pytest.approx(3e19 * 75.0 * 1.602176634e-22)
    assert values['lambda_Te_ped'] == pytest.approx(1.5)
    assert values['lambda_Ne_ped'] == pytest.approx(1.5)
    assert values['lambda_p_ped'] == pytest.approx(0.75)
    assert np.isnan(_pedestal_profile_values(R, R, R, R_ped=5.0)['Te_ped'])


def test_result_summary_includes_pedestal_top_values():
    state = {
        'Ne_ped': 1e19,
        'p_ped': 0.5,
        'Te_ped': 200.0,
        'lambda_Te_ped': 0.02,
        'lambda_Ne_ped': 0.03,
        'lambda_p_ped': 0.012,
    }
    summary = _result_summary({'_pedestal': {
        'R_ped': 3.9,
        'spline': state,
        'mtanh': state,
    }})

    assert summary['Rped_eqh'] == 3.9
    assert summary['spline_Te_ped'] == 200.0
    assert summary['mtanh_lambda_p_ped'] == 0.012
    assert np.isnan(_result_summary(None, status='failed')['spline_Ne_ped'])


def test_scan_result_row_includes_spline_decay_lengths_at_separatrix():
    result = {
        'spline': {
            'knots': [1.0, 1.5, 2.0],
            'method': 'ipfn_spline',
            'Te_params': [100.0, 200.0],
            'ne_params': [1.0e19, 2.0e19],
        },
        'fit_metrics': {'spline': {
            'Te': {'weighted_sse': 4.0, 'reduced_chi2': 1.0},
            'Ne': {'weighted_sse': 6.0, 'reduced_chi2': 1.5},
        }},
        'physics': {
            'eqh': {'Rsep_eqh': 2.0},
            'spline': {
                'Rsep': 1.5,
                'Te_sep': 150.0,
                'Ne_sep': 1.5e19,
                'R_grid': [1.0, 2.0],
                'lam_Te_grid': [0.01, 0.03],
                'lam_q_grid': [0.02, 0.04],
                'lambda_Ne_sep': 0.05,
                'lambda_p_sep': 0.06,
            },
        },
    }

    row = _successful_row(result, {})

    assert row['fit_success']
    assert row['lambda_Te_sep_m'] == pytest.approx(0.02)
    assert row['lambda_Ne_sep_m'] == pytest.approx(0.05)
    assert row['lambda_q_sep_m'] == pytest.approx(0.03)
    assert row['lambda_p_sep_m'] == pytest.approx(0.06)


def test_estimate_separatrix_computes_electron_pressure_scale_length(monkeypatch):
    def find_sep(scalars, profiles, model, options):
        return {'x_sep': 1.0}, None

    def model(model_state):
        model_state.update({
            'lam_q': 0.01,
            'q_par_P_SOL': 1.0,
            'q_par_SH': 2.0,
        })
        return 0.0

    monkeypatch.setattr(
        'ts_profile_fitter.separatrix._pysepest_components',
        lambda: (find_sep, model, object()),
    )
    profile = make_profile(
        R=[0.0, 1.0, 2.0],
        Te=[4.0, 3.0, 2.0],
        Ne=[8.0, 6.0, 4.0],
    )

    state = estimate_separatrix(
        profile,
        power_balance={'P_abs': 1000.0, 'P_rsep': 100.0, 'dWdt': 50.0},
        geometry={
            'R_out': 2.0,
            'B_pol': 0.2,
            'B_tor': 2.5,
            'L_par': 10.0,
            'Z_eff': 1.0,
        },
        x_grid=[0.2, 0.6, 1.0, 1.4, 1.8],
    )

    assert state['lambda_p_sep'] == pytest.approx(1.5)
    assert state['lambda_Ne_sep'] == pytest.approx(3.0)


def test_failure_stage_separates_power_balance_from_pipeline_stage():
    assert _failure_stage(
        {'stage': 'physics'},
        PowerBalanceUnavailableError('dWdt unavailable'),
    ) == 'power_balance'
    assert _failure_stage({'stage': 'physics'}, RuntimeError('boom')) == 'physics'
    assert _failure_stage({}, RuntimeError('boom')) == 'unknown'


def test_stage_status_fields_mark_failed_completed_and_not_run_groups():
    assert _stage_status_fields(
        {'stage': 'mtanh', 'completed_stages': ['spline']}, 'mtanh',
    ) == {
        'failed_stage': 'mtanh',
        'spline_status': 'ok',
        'mtanh_status': 'failed',
        'physics_status': 'not_run',
    }
    assert _stage_status_fields(
        {'completed_stages': ['spline', 'mtanh', 'pedestal']}, 'power_balance',
    ) == {
        'failed_stage': 'power_balance',
        'spline_status': 'ok',
        'mtanh_status': 'ok',
        'physics_status': 'failed',
    }
    assert _stage_status_fields(
        {'completed_stages': ['spline', 'mtanh', 'pedestal', 'physics']},
    ) == {
        'failed_stage': '',
        'spline_status': 'ok',
        'mtanh_status': 'ok',
        'physics_status': 'ok',
    }


def test_stage_columns_default_for_records_without_stage_fields():
    assert _stage_columns({}) == {
        'failed_stage': '',
        'spline_status': 'not_run',
        'mtanh_status': 'not_run',
        'physics_status': 'not_run',
    }


def test_partial_result_keeps_spline_outputs_and_summarises_as_failed():
    assert _partial_result({'stage': 'spline'}) is None

    context = {
        'stage': 'mtanh',
        'completed_stages': ['spline'],
        'spline_fit': {
            'n_points': 4,
            'spline': {'method': 'ondrej_spline', 'knots': [3.6, 3.9]},
            'fit_metrics': {'Te': {'reduced_chi2': 0.5}},
        },
    }
    partial = _partial_result(context)

    assert partial['n_points'] == 4
    assert partial['spline']['method'] == 'ondrej_spline'
    assert partial['fit_metrics'] == {
        'mtanh': {},
        'spline': {'Te': {'reduced_chi2': 0.5}},
    }
    assert partial['physics'] is None

    summary = _result_summary(
        partial, status='failed', error='mtanh fit did not converge')
    assert summary['fit_status'] == 'failed'
    assert summary['spline_Te_reduced_chi2'] == 0.5
    assert np.isnan(summary['mtanh_Te_reduced_chi2'])
    assert summary['n_knots'] == 2


def test_failed_scan_row_keeps_spline_outputs_when_mtanh_fails():
    context = {
        'stage': 'mtanh',
        'completed_stages': ['spline'],
        'spline_fit': {
            'n_points': 4,
            'spline': {
                'method': 'ondrej_spline',
                'knots': [3.6, 3.9],
                'Te_params': [100.0, 200.0],
                'ne_params': [1.0e19, 2.0e19],
            },
            'fit_metrics': {
                'Te': {'weighted_sse': 4.0, 'reduced_chi2': 1.0},
                'Ne': {'weighted_sse': 6.0, 'reduced_chi2': 1.5},
            },
        },
    }

    row = _failed_row(
        {'shot_id': 34383},
        RuntimeError('mtanh fit did not converge'),
        context,
    )

    assert row['fit_success'] is False
    assert row['fit_error'] == 'RuntimeError: mtanh fit did not converge'
    assert row['failed_stage'] == 'mtanh'
    assert row['spline_status'] == 'ok'
    assert row['mtanh_status'] == 'failed'
    assert row['physics_status'] == 'not_run'
    assert row['spline_method'] == 'ondrej_spline'
    assert row['Te_chi_squared'] == 4.0

    row = _failed_row(
        {'shot_id': 34383},
        RuntimeError('spline did not converge'),
        {'stage': 'spline'},
    )

    assert row['fit_success'] is False
    assert row['failed_stage'] == 'spline'
    assert row['spline_status'] == 'failed'
    assert row['mtanh_status'] == 'not_run'


def test_enrich_database_adds_stage_columns_when_records_are_empty():
    intervals = pd.DataFrame({
        'shot_id': [34383],
        't_begin': [1.0],
        't_end': [1.1],
    })

    enriched = enrich_database(intervals, [])

    assert {
        'failed_stage', 'spline_status', 'mtanh_status', 'physics_status',
    }.issubset(enriched.columns)


def test_run_database_writes_stage_columns_and_failed_by_stage(tmp_path, monkeypatch):
    shot_list = pd.DataFrame({'shot_id': [34383]})
    intervals = pd.DataFrame({
        'shot_id': [34383, 34383],
        't_begin': [1.0, 2.0],
        't_end': [1.1, 2.1],
        'label': ['mtanh case', 'power balance case'],
    })
    run_records = [
        {
            'shot_id': 34383, 't_begin': 1.0, 't_end': 1.1,
            'label': 'mtanh case',
            'status': 'failed', 'warning': '',
            'error': 'mtanh fit did not converge',
            'failed_stage': 'mtanh',
            'spline_status': 'ok',
            'mtanh_status': 'failed',
            'physics_status': 'not_run',
            '_partial_result': None,
            '_interval_runtime_seconds': 0.1,
        },
        {
            'shot_id': 34383, 't_begin': 2.0, 't_end': 2.1,
            'label': 'power balance case',
            'status': 'failed', 'warning': '',
            'error': 'Power-balance term(s) unavailable',
            'failed_stage': 'power_balance',
            'spline_status': 'ok',
            'mtanh_status': 'ok',
            'physics_status': 'failed',
            '_partial_result': None,
            '_interval_runtime_seconds': 0.1,
        },
    ]
    monkeypatch.setattr(
        'ts_profile_fitter.pipeline.run_shot',
        lambda *args, **kwargs: list(run_records),
    )

    enriched = run_database(
        shot_list, intervals, tmp_path / 'data', tmp_path / 'out')

    assert list(enriched['failed_stage']) == ['mtanh', 'power_balance']
    assert list(enriched['spline_status']) == ['ok', 'ok']
    assert list(enriched['mtanh_status']) == ['failed', 'ok']
    assert list(enriched['physics_status']) == ['not_run', 'failed']
    assert list(enriched['fit_status']) == ['failed', 'failed']
    summary = json.loads(
        (tmp_path / 'out' / 'run_summary.json').read_text(encoding='utf-8'))
    assert summary['intervals']['failed_by_stage'] == {
        'mtanh': 1,
        'power_balance': 1,
    }