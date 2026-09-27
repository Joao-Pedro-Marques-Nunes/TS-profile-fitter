import numpy as np

from .connection_length import get_connection_length
from .separatrix import estimate_separatrix, make_profile


def compute_separatrix(profile_x, profile_y):
    """Estimate the separatrix location from a fitted profile."""
    # TODO: implement a physics-based separatrix estimator
    ix = np.argmax(np.gradient(profile_y, profile_x))
    return profile_x[ix]


def compute_profile_quantities(x, y, fit_model=None):
    """Compute derived profile quantities from fitted profiles."""
    results = {
        'x_min': np.min(x),
        'x_max': np.max(x),
        'y_min': np.min(y),
        'y_max': np.max(y),
    }
    if fit_model is not None:
        results['separatrix'] = compute_separatrix(x, fit_model(x))
    return results


def compute_equilibrium_separatrix(equilibrium, interval, rho_sep=1.0):
    """Compute the outer-midplane equilibrium separatrix radius.

    The equilibrium surface is evaluated at ``rho_sep`` and at the magnetic
    axis height interpolated to the interval midpoint.
    """
    if hasattr(interval, 'get'):
        t_begin = interval.get('t_begin')
        t_end = interval.get('t_end')
    else:
        t_begin, t_end = interval
    if t_begin is None or t_end is None:
        raise ValueError('interval must define t_begin and t_end.')

    t_mid = 0.5 * (float(t_begin) + float(t_end))
    equilibrium_time = equilibrium.time
    z_mag = equilibrium.Zmag
    if equilibrium_time.size != z_mag.size or equilibrium_time.size < 2:
        raise ValueError('Equilibrium time and Zmag arrays are incompatible.')

    from scipy.interpolate import make_interp_spline
    from ipfnpytools.trhopz_to_r import trhopz_to_r

    z_mag_mid = make_interp_spline(
        equilibrium_time, z_mag, k=3)(t_mid)
    r_midplane = trhopz_to_r(
        t_mid,
        rho_sep,
        z_mag_mid,
        eq_shotfile=equilibrium,
    )[1]

    return {
        'Rsep_eqh': float(np.asarray(r_midplane).ravel()[-1]),
        'rho_sep': float(rho_sep),
        'tmid': t_mid,
        'z_mag': float(z_mag_mid),
    }


def compute_physics_separatrix(
    R,
    Te,
    Ne=None,
    power_balance=None,
    geometry=None,
    model_config=None,
    dTe_dR=None,
    dNe_dR=None,
    fit_method=None,
    x_grid=None,
    optimizer=None,
):
    """Compute a physics-based separatrix from any evaluated profile model.

    ``power_balance`` must contain SI-power values ``P_abs``, ``P_rsep`` and
    ``dWdt`` in watts. ``geometry`` must contain ``R_out``, ``B_pol``,
    ``B_tor`` and ``L_par``.
    """
    if power_balance is None or geometry is None:
        raise ValueError('power_balance and geometry are required.')

    profile = make_profile(
        R,
        Te,
        Ne=Ne,
        dTe_dR=dTe_dR,
        dNe_dR=dNe_dR,
        fit_method=fit_method,
    )
    return estimate_separatrix(
        profile,
        power_balance,
        geometry,
        model_config=model_config,
        x_grid=x_grid,
        optimizer=optimizer,
    )


__all__ = [
    'compute_separatrix',
    'compute_profile_quantities',
    'compute_equilibrium_separatrix',
    'compute_physics_separatrix',
    'get_connection_length',
]
