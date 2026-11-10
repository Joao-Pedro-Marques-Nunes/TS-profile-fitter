"""Physics-based separatrix estimation using the pysepest framework."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from scipy.interpolate import interp1d


class PowerBalanceUnavailableError(ValueError):
    """Raised when the power-balance terms needed for P_SOL are not usable."""


def _pysepest_components():
    try:
        from pysepest.finder import find_sep
        from pysepest.models import conduction_limited_local_lam_q
        from pysepest.optimizers import first_0_crossing
    except ImportError as exc:
        raise ImportError(
            'pysepest is required for physics-based separatrix estimation. '
            'Make the local pysepest package available to the Python environment.'
        ) from exc
    return find_sep, conduction_limited_local_lam_q, first_0_crossing


def _clean_profile(R, values, name):
    R = np.asarray(R, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if R.size != values.size:
        raise ValueError(f'{name} must have the same length as R.')
    mask = np.isfinite(R) & np.isfinite(values)
    if np.count_nonzero(mask) < 3:
        raise ValueError(f'{name} requires at least three finite profile points.')
    R = R[mask]
    values = values[mask]
    order = np.argsort(R)
    R = R[order]
    values = values[order]
    R, unique_indices = np.unique(R, return_index=True)
    return R, values[unique_indices]


def make_profile(R, Te, Ne=None, dTe_dR=None, dNe_dR=None, fit_method=None):
    """Build a standardized fitted-profile dictionary.

    Temperatures must be in eV, densities in m^-3, and radius in meters.
    Derivatives are calculated from the supplied profile unless provided.
    """
    R, Te = _clean_profile(R, Te, 'Te')
    if Ne is not None:
        R_ne, Ne = _clean_profile(R, Ne, 'Ne')
        if not np.array_equal(R, R_ne):
            raise ValueError('Te and Ne must be evaluated on the same R grid.')

    if dTe_dR is None:
        dTe_dR = np.gradient(Te, R)
    else:
        dTe_dR = np.asarray(dTe_dR, dtype=float).ravel()
        if dTe_dR.size != R.size:
            raise ValueError('dTe_dR must have the same length as R.')

    profile = {
        'R': R,
        'Te': Te,
        'dTe_dR': dTe_dR,
        'fit_method': fit_method,
    }
    if Ne is not None:
        profile['Ne'] = Ne
        profile['dNe_dR'] = (
            # TODO: use the analytic derivative of the modified-tanh model for
            # density when mtanh parameters are available.
            np.gradient(Ne, R) if dNe_dR is None else np.asarray(dNe_dR, dtype=float).ravel()
        )
        if profile['dNe_dR'].size != R.size:
            raise ValueError('dNe_dR must have the same length as R.')
    return profile


def _profile_interpolator(R, values):
    return interp1d(R, values, bounds_error=True, assume_sorted=True)


def _valid_profile_region(profile, x_grid, min_lambda=1e-8):
    Te = _profile_interpolator(profile['R'], profile['Te'])(x_grid)
    dTe_dR = _profile_interpolator(profile['R'], profile['dTe_dR'])(x_grid)
    lam_Te = -Te / dTe_dR
    return np.isfinite(Te) & np.isfinite(lam_Te) & (lam_Te > min_lambda)


def estimate_separatrix(
    profile: Mapping,
    power_balance: Mapping,
    geometry: Mapping,
    model_config: Optional[Mapping] = None,
    x_grid=None,
    optimizer=None,
    vectorize=False,
):
    """Estimate a physics-based separatrix with pysepest.

    ``power_balance`` values are in watts: ``P_abs``, ``P_rsep``, and
    ``dWdt``. ``geometry`` contains ``R_out``, ``B_pol``, ``B_tor``, and
    ``L_par``. Geometry values may be scalars or callables evaluated at R.
    """
    model_config = dict(model_config or {})
    power_terms = {
        'P_abs': power_balance.get('P_abs', np.nan),
        'P_rsep': power_balance.get('P_rsep', np.nan),
        'dWdt': power_balance.get('dWdt', np.nan),
    }
    unavailable_terms = []
    numeric_terms = {}
    for name, value in power_terms.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = np.nan
        numeric_terms[name] = numeric_value
        if not np.isfinite(numeric_value):
            unavailable_terms.append(name)
    if unavailable_terms:
        terms = ', '.join(unavailable_terms)
        raise PowerBalanceUnavailableError(
            'Power-balance term(s) unavailable for this interval: '
            f'{terms}. Expected finite P_abs, P_rsep, and dWdt values.'
        )
    P_sol = numeric_terms['P_abs'] - numeric_terms['P_rsep'] - numeric_terms['dWdt']
    if not np.isfinite(P_sol):
        raise PowerBalanceUnavailableError(
            'Power-balance terms produced a non-finite P_SOL after subtraction.'
        )

    defaults = {
        'f_e': 0.8,
        'f_out': 0.65,
        'f_cond': 0.8,
        'q_distribution': 1.0,
        'min_lambda': 1e-8,
    }
    defaults.update(model_config)
    P_e_SOL = P_sol * defaults['f_e'] * defaults['f_out'] * defaults['f_cond']

    R = np.asarray(profile['R'], dtype=float)
    if x_grid is None:
        x_grid = np.linspace(R.min(), R.max(), 2000)
    x_grid = np.asarray(x_grid, dtype=float).ravel()
    x_grid = x_grid[(x_grid >= R.min()) & (x_grid <= R.max())]
    valid = _valid_profile_region(profile, x_grid, defaults['min_lambda'])
    x_grid = x_grid[valid]
    if x_grid.size < 3:
        raise ValueError('No sufficient physical Te-gradient region remains for separatrix search.')

    def value_at(name, x):
        value = geometry[name]
        return value(x) if callable(value) else value

    profiles = {
        'Te': _profile_interpolator(R, profile['Te']),
        'lam_Te': _profile_interpolator(R, -profile['Te'] / profile['dTe_dR']),
    }
    if 'Ne' in profile:
        profiles['Ne'] = _profile_interpolator(R, profile['Ne'])
        profiles['lam_Ne'] = _profile_interpolator(R, -profile['Ne'] / profile['dNe_dR'])

    scalars = {
        'P_e_SOL': P_e_SOL,
        'R': geometry['R_out'],
        'B_pol': geometry['B_pol'],
        'B_tor': geometry['B_tor'],
        'L_par': geometry['L_par'],
        'Zeff': geometry['Z_eff'],
        'q_distribution': defaults['q_distribution'],
    }

    find_sep, model, first_0_crossing = _pysepest_components()
    if optimizer is None:
        optimizer = first_0_crossing

    state, optimizer_result = find_sep(
        scalars,
        profiles,
        model,
        {'optimizer': optimizer, 'x_grid': x_grid, 'vectorize': vectorize},
    )
    Rsep = state.get('x_sep', np.nan)
    if np.isfinite(Rsep):
        state['Rsep'] = Rsep
        state['Te_sep'] = float(profiles['Te'](Rsep))
        if 'Ne' in profiles:
            state['Ne_sep'] = float(profiles['Ne'](Rsep))
            lambda_te = float(profiles['lam_Te'](Rsep))
            lambda_ne = float(profiles['lam_Ne'](Rsep))
            state['lambda_Ne_sep'] = lambda_ne
            if (np.isfinite(lambda_te) and lambda_te != 0
                    and np.isfinite(lambda_ne) and lambda_ne != 0):
                inverse_lambda_p = 1.0 / lambda_te + 1.0 / lambda_ne
                state['lambda_p_sep'] = (
                    1.0 / inverse_lambda_p
                    if np.isfinite(inverse_lambda_p) and inverse_lambda_p != 0
                    else np.nan
                )
            else:
                state['lambda_p_sep'] = np.nan
    else:
        state['Rsep'] = np.nan

    def evaluate_model(x):
        model_state = scalars.copy()
        model_state['Te'] = float(profiles['Te'](x))
        model_state['lam_Te'] = float(profiles['lam_Te'](x))
        mismatch = model(model_state)
        model_state['model_mismatch'] = mismatch
        return model_state

    model_grid = [evaluate_model(x) for x in x_grid]

    state.update({
        'P_SOL': P_sol,
        'P_e_SOL': P_e_SOL,
        'fit_method': profile.get('fit_method'),
        'R_grid': x_grid,
        'mismatch_grid': np.asarray([item['model_mismatch'] for item in model_grid]),
        'Te_grid': np.asarray([item['Te'] for item in model_grid]),
        'lam_Te_grid': np.asarray([item['lam_Te'] for item in model_grid]),
        'lam_q_grid': np.asarray([item['lam_q'] for item in model_grid]),
        'q_par_P_SOL_grid': np.asarray([item['q_par_P_SOL'] for item in model_grid]),
        'q_par_SH_grid': np.asarray([item['q_par_SH'] for item in model_grid]),
        'optimizer_result': optimizer_result,
    })
    return state


__all__ = ['make_profile', 'estimate_separatrix', 'PowerBalanceUnavailableError']
