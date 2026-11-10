import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, optimize, signal
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline


def mtanh(x, x0, width, amin, amax, offset):
    return amin + (amax - amin) * 0.5 * (1 + np.tanh((x - x0) / width)) + offset


def modified_tanh(x, x0, w, a_tail, b_tail, c_tail, d_tail, a_top, b_top, c_top, d_top):
    dx = x - x0
    T = np.tanh(dx / w)
    tail = a_tail + b_tail * dx + c_tail * dx**2 + d_tail * dx**3
    top = a_top + b_top * dx + c_top * dx**2 + d_top * dx**3
    return 0.5 * (1 + T) * tail + 0.5 * (1 - T) * top


def count_modified_tanh_parameters(deg_tail=3, deg_top=3,
                                   fix_x0=None, fix_w=None,
                                   fix_a_tail=None):
    """Return the number of free parameters in a configured modified tanh."""
    if not 0 <= deg_tail <= 3 or not 0 <= deg_top <= 3:
        raise ValueError('Polynomial degrees must be between 0 and 3.')

    n_parameters = 2 + (deg_tail + 1) + (deg_top + 1)
    n_parameters -= int(fix_x0 is not None)
    n_parameters -= int(fix_w is not None)
    n_parameters -= int(fix_a_tail is not None)
    return n_parameters


def fit_mtanh_profile(x, y, yerr=None, p0=None, bounds=None):
    """Fit an mtanh profile to profile data."""
    if p0 is None:
        p0 = [np.median(x), 0.1, np.min(y), np.max(y), 0.0]
    if bounds is None:
        bounds = (-np.inf, np.inf)
    popt, pcov = curve_fit(mtanh, x, y, sigma=yerr, p0=p0, bounds=bounds)
    return popt, pcov


def _prepare_joint_fit_data(x, y, y_err):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    if x.size == 0 or y.size == 0 or y_err.size == 0:
        raise ValueError('Joint fitting input arrays must not be empty.')

    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(y_err).all():
        raise ValueError('Joint fitting input arrays contain non-finite values; clean the data before fitting.')

    return x, y, y_err


def fit_modified_tanh_profile(x, y, y_err=None, p0=None,
                              deg_tail=3, deg_top=3,
                              fix_x0=None, fix_w=None,
                              fix_a_tail=None, d=None,
                              y_scale=None):
    """Fit a modified tanh profile to a single sorted profile dataset."""
    if y_err is None:
        y_err = np.ones_like(y, dtype=float)

    x_data, y_data, y_err = _prepare_joint_fit_data(x, y, y_err)
    if y_scale is None:
        y_reference = np.max(np.abs(y_data))
        y_scale = 10.0 ** np.floor(np.log10(y_reference)) if y_reference > 0 else 1.0
    y_scale = float(y_scale)
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('y_scale must be a finite positive number.')
    y_data = y_data / y_scale
    y_err = y_err / y_scale

    if x_data.size == 0 or y_data.size == 0:
        return np.full(10, np.nan), None

    if len(x_data) < 3:
        return np.full(10, np.nan), None

    p0_was_provided = p0 is not None
    if p0 is None:
        p0 = [
            np.median(x_data),
            0.05,
            np.min(y_data),
            0.0,
            0.0,
            0.0,
            np.max(y_data),
            0.0,
            0.0,
            0.0,
        ]
    if len(p0) != 10:
        raise ValueError('Initial guess p0 must have exactly 10 elements.')

    p0 = np.asarray(p0, dtype=float).copy()
    if p0_was_provided:
        p0[2:] /= y_scale
    fit_mask = np.array([True] * 10, dtype=bool)

    if deg_tail < 3:
        fit_mask[5] = False
    if deg_tail < 2:
        fit_mask[4] = False
    if deg_tail < 1:
        fit_mask[3] = False
    if fix_a_tail is not None:
        fit_mask[2] = False
        p0[2] = float(fix_a_tail) / y_scale

    if deg_top < 3:
        fit_mask[9] = False
    if deg_top < 2:
        fit_mask[8] = False
    if deg_top < 1:
        fit_mask[7] = False

    if fix_x0 is not None:
        fit_mask[0] = False
        p0[0] = float(fix_x0)
    if fix_w is not None:
        fit_mask[1] = False
        p0[1] = float(fix_w)

    fit_indices = np.where(fit_mask)[0]
    p0_reduced = [p0[idx] for idx in fit_indices]

    def wrapped_modified_tanh(x, *p_reduced):
        full_params = p0.copy()
        for i, idx in enumerate(fit_indices):
            full_params[idx] = p_reduced[i]
        return modified_tanh(x, *full_params)

    try:
        popt_reduced, pcov = curve_fit(
            wrapped_modified_tanh,
            x_data,
            y_data,
            sigma=y_err,
            absolute_sigma=True,
            p0=p0_reduced,
        )
    except Exception as exc:
        raise RuntimeError(
            'Modified-tanh fit failed; inspect data scale, uncertainties, '
            'initial parameters, and polynomial degrees.'
        ) from exc

    popt_full = p0.copy()
    for i, idx in enumerate(fit_indices):
        popt_full[idx] = popt_reduced[i]
    popt_full[2:] *= y_scale
    if pcov is not None:
        parameter_scales = np.ones(len(fit_indices), dtype=float)
        parameter_scales[np.asarray(fit_indices) >= 2] = y_scale
        pcov = pcov * np.outer(parameter_scales, parameter_scales)
    return popt_full, pcov


def fit_spline_profile(x, y, bc_type='natural'):
    """Fit a smooth spline to profile data."""
    return CubicSpline(x, y, bc_type=bc_type)


def find_x_means(x, dx_max, use_edges=True):
    """Find knot locations by grouping points separated by at least dx_max."""
    x_s = np.sort(np.asarray(x, dtype=float))
    if x_s.size == 0:
        return np.array([], dtype=float)
    dx = np.diff(x_s)
    i_jump = (dx > dx_max).nonzero()[0]
    i_jump_loc = signal.argrelmax(dx)[0]
    i_jump = np.union1d(i_jump, i_jump_loc)
    i_jump += 1
    x_t = np.array([np.mean(chunk) for chunk in np.split(x_s, i_jump)])
    if use_edges:
        x_t[0] = x_s[0]
        x_t[-1] = x_s[-1]
    return x_t


def compute_channel_knots(
    ds_ts,
    channels_before,
    channels_removed=None,
    knot_count_shift=0,
    strategy='global_quantile',
    core_merge_rhop_spacing_factor=0.5,
):
    """Compute spline knots, including core-anchored edge strategies."""
    required_columns = {'R_mid'}
    missing = required_columns.difference(ds_ts.columns)
    if missing:
        raise ValueError(f"TS data is missing required knot columns: {sorted(missing)}")

    channels_removed = channels_removed or {'core': 0, 'edge': 0}
    retained_core = channels_before['core'] - channels_removed.get('core', 0)
    retained_edge = channels_before['edge'] - channels_removed.get('edge', 0)
    if strategy not in {
        'global_quantile',
        'balanced_quantile',
        'edge_biased_quantile',
        'hybrid',
    }:
        raise ValueError(
            f"Unknown knot strategy {strategy!r}; expected 'global_quantile' "
            "'balanced_quantile', 'edge_biased_quantile', or 'hybrid'."
        )

    if strategy in {'hybrid', 'edge_biased_quantile'}:
        core_anchored_columns = {'system', 'channel_index', 'rho_pol'}
        missing = core_anchored_columns.difference(ds_ts.columns)
        if missing:
            raise ValueError(
                'Core-anchored knot placement requires columns: '
                f'{sorted(missing)}'
            )

        try:
            core_merge_rhop_spacing_factor = float(
                core_merge_rhop_spacing_factor)
        except (TypeError, ValueError) as error:
            raise ValueError(
                'core_merge_rhop_spacing_factor must be a finite non-negative number.'
            ) from error
        if (not np.isfinite(core_merge_rhop_spacing_factor)
                or core_merge_rhop_spacing_factor < 0):
            raise ValueError(
                'core_merge_rhop_spacing_factor must be a finite non-negative number.'
            )

        core_anchored_data = ds_ts[
            ['system', 'channel_index', 'R_mid', 'rho_pol']
        ].copy()
        core_anchored_data['R_mid'] = pd.to_numeric(
            core_anchored_data['R_mid'], errors='coerce')
        core_anchored_data['rho_pol'] = pd.to_numeric(
            core_anchored_data['rho_pol'], errors='coerce')
        core_anchored_data = core_anchored_data.dropna(
            subset=['system', 'channel_index', 'R_mid', 'rho_pol'])
        core_anchored_data = core_anchored_data.loc[
            np.isfinite(core_anchored_data['R_mid'])
            & np.isfinite(core_anchored_data['rho_pol'])]
        core_channels = (
            core_anchored_data.loc[core_anchored_data['system'].eq('core')]
            .groupby('channel_index', sort=False)
            .agg(R_mid=('R_mid', 'median'), rho_pol=('rho_pol', 'median'))
            .sort_values('rho_pol', kind='stable')
        )
        edge_data = core_anchored_data.loc[
            core_anchored_data['system'].eq('edge')
        ]
        edge_channel_count = edge_data['channel_index'].nunique()
        if strategy == 'hybrid':
            edge_positions = (
                edge_data.groupby('channel_index', sort=False)['R_mid']
                .median()
                .to_numpy(dtype=float)
            )
        else:
            edge_positions = edge_data['R_mid'].to_numpy(dtype=float)
        if core_channels.empty or edge_channel_count == 0:
            raise ValueError(
                'Core-anchored knot placement requires retained core and edge channels.'
            )

        core_channel_ids = core_channels.index.to_numpy()
        core_rhop = core_channels['rho_pol'].to_numpy(dtype=float)
        core_channel_positions = core_channels['R_mid'].to_numpy(dtype=float)
        core_rhop_gaps = np.diff(core_rhop)
        positive_rhop_gaps = core_rhop_gaps[core_rhop_gaps > 0]
        typical_rhop_gap = (
            float(np.median(positive_rhop_gaps))
            if positive_rhop_gaps.size else 0.0
        )
        merge_threshold = (
            core_merge_rhop_spacing_factor * typical_rhop_gap)
        core_groups = [[0]]
        for position_index, rhop_gap in enumerate(core_rhop_gaps, start=1):
            if typical_rhop_gap > 0 and rhop_gap < merge_threshold:
                print(
                    'Merged core channels '
                    f'{core_channel_ids[position_index - 1]} and '
                    f'{core_channel_ids[position_index]} '
                    f'(rho_pol gap={rhop_gap:.6g}, '
                    f'threshold={merge_threshold:.6g})'
                )
                core_groups[-1].append(position_index)
            else:
                core_groups.append([position_index])
        core_positions = np.asarray([
            np.median(core_channel_positions[group])
            for group in core_groups
        ], dtype=float)

        n_edge_knots = int(np.ceil(edge_channel_count / 2)) + int(knot_count_shift)
        if n_edge_knots < 1:
            raise ValueError(
                'Hybrid knot placement requires at least one edge knot; '
                f'got {n_edge_knots}.'
            )
        if n_edge_knots > np.unique(edge_positions).size:
            raise ValueError(
                f"Cannot compute {n_edge_knots} edge knots from "
                f"{np.unique(edge_positions).size} unique edge-channel positions."
            )

        n_knots = core_positions.size + n_edge_knots
        if n_knots < 4:
            raise ValueError(f"Cubic spline fitting requires at least 4 knots; got {n_knots}.")

        edge_quantiles = (
            np.array([0.5])
            if n_edge_knots == 1
            else np.linspace(0.0, 1.0, n_edge_knots)
        )
        knots = np.sort(np.concatenate([
            core_positions,
            np.quantile(edge_positions, edge_quantiles),
        ]))
        if np.unique(knots).size != n_knots:
            raise ValueError(
                'Core and edge knot positions must be distinct; core-channel '
                'medians and edge quantiles overlap.'
            )
        return knots

    base_count = int(np.ceil((retained_core + retained_edge) / 2))
    n_knots = base_count + int(knot_count_shift)
    if n_knots < 4:
        raise ValueError(f"Cubic spline fitting requires at least 4 knots; got {n_knots}.")

    if strategy == 'global_quantile':
        radial_positions = ds_ts['R_mid'].to_numpy(dtype=float)
        radial_positions = radial_positions[np.isfinite(radial_positions)]
        if np.unique(radial_positions).size < n_knots:
            raise ValueError(
                f"Cannot compute {n_knots} channel knots from "
                f"{np.unique(radial_positions).size} unique radial positions."
            )
        return np.quantile(radial_positions, np.linspace(0, 1, n_knots))

    if 'system' not in ds_ts.columns:
        raise ValueError("Balanced quantile knots require a 'system' column.")

    core_positions = ds_ts.loc[ds_ts['system'].eq('core'), 'R_mid'].to_numpy(float)
    edge_positions = ds_ts.loc[ds_ts['system'].eq('edge'), 'R_mid'].to_numpy(float)
    core_positions = core_positions[np.isfinite(core_positions)]
    edge_positions = edge_positions[np.isfinite(edge_positions)]
    n_core = n_knots // 2
    n_edge = n_knots - n_core
    if np.unique(core_positions).size < n_core:
        raise ValueError(
            f"Cannot compute {n_core} core knots from "
            f"{np.unique(core_positions).size} unique core radial positions."
        )
    if np.unique(edge_positions).size < n_edge:
        raise ValueError(
            f"Cannot compute {n_edge} edge knots from "
            f"{np.unique(edge_positions).size} unique edge radial positions."
        )

    knots = np.concatenate([
        np.quantile(core_positions, np.linspace(0, 1, n_core)),
        np.quantile(edge_positions, np.linspace(0, 1, n_edge)),
    ])
    knots = np.unique(np.sort(knots))
    if knots.size < 4:
        raise ValueError(
            f"Balanced quantile knots contain only {knots.size} unique positions; "
            "cubic spline fitting requires at least 4."
        )
    return knots


def prepare_monotonic_decr_bspline_basis(x_t, x_f, nu=0, k=3, decr=True):
    """Build a monotonic decreasing B-spline basis matrix on a fine grid."""
    x_t = np.asarray(x_t, dtype=float)
    x_f = np.asarray(x_f, dtype=float)
    df = len(x_t)
    c = np.eye(df)
    t = interpolate._bsplines._not_a_knot(x_t, k)
    bspl = interpolate.BSpline(t, c, k)
    bmat = bspl(x_f, nu)
    if decr:
        return np.cumsum(bmat, axis=1)
    return bmat


def gaussian(dx, bw):
    """Gaussian kernel with standard deviation bw."""
    if bw <= 0:
        return np.zeros_like(dx, dtype=float)
    return np.exp(-0.5 * (dx / bw) ** 2) / (np.sqrt(2 * np.pi) * bw)


def _fit_spl_new(bmat, x_f, x, y, y_err, max_step_scale=3, use_errors=True, loss='linear', f_scale=1.0):
    """Fit monotonic decreasing spline coefficients with optional regularization and robust loss."""
    bmat = np.asarray(bmat, dtype=float)
    x_f = np.asarray(x_f, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    if bmat.ndim != 2:
        raise ValueError('Spline basis matrix must be 2D.')

    m = bmat.shape[1]
    y_max = np.quantile(y, 0.98)
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = np.max(np.abs(y))
        if not np.isfinite(y_max) or y_max <= 0:
            y_max = 1.0

    bmat_x = interpolate.interp1d(x_f, bmat, axis=0, bounds_error=False, fill_value='extrapolate')(x)
    if max_step_scale is not None:
        I_m = np.eye(m)
        D_1 = I_m[:-1, :] - I_m[1:, :]
        bmat_x = np.vstack([bmat_x, D_1])
        y_aug = np.hstack([y, np.zeros(m - 1)])
        step_scale = y_max / max_step_scale
        y_err_aug = np.hstack([np.maximum(y_err, np.finfo(float).eps), np.full(m - 1, step_scale)])
        y = y_aug
        y_err = y_err_aug

    if use_errors:
        p, pcov = optimize.curve_fit(
            lambda xval, *p: np.dot(bmat_x, p),
            x,
            y,
            sigma=y_err,
            absolute_sigma=True,
            p0=np.full(m, y_max / m),
            bounds=(0, np.inf),
            loss=loss,
            f_scale=f_scale,
            jac=lambda xval, *p: bmat_x,
        )
    else:
        p, pcov = optimize.curve_fit(
            lambda xval, *p: np.dot(bmat_x, p),
            x,
            y,
            p0=np.full(m, y_max / m),
            bounds=(0, np.inf),
            loss=loss,
            f_scale=f_scale,
            jac=lambda xval, *p: bmat_x,
        )
    return p, pcov


def _save_spl_eval(ds, q, p, bmat, bmat_dx, bmat_conv, pcov):
    """Store spline evaluations and derivatives on the provided DataFrame."""
    if not isinstance(ds, pd.DataFrame):
        raise TypeError('ds must be a pandas DataFrame.')

    if bmat_conv is not None:
        ds[f'{q}e_spl_conv'] = np.dot(bmat_conv, p)
    ds[f'{q}e_spl'] = np.dot(bmat, p)
    ds[f'd{q}e_spl'] = np.dot(bmat_dx, p)
    ds[f'lam_{q}e_spl'] = -ds[f'{q}e_spl'] / ds[f'd{q}e_spl']
    if pcov is None:
        ds[f'{q}e_err_spl'] = np.full(len(ds), np.nan)
    else:
        ds[f'{q}e_err_spl'] = np.diagonal(bmat.dot(pcov.dot(bmat.T)))


def fit_mbsplt_ne_Te_TS(ds_TS, dR_fine=5e-4, sigma_conv=5e-3, R_knots=None, dR_channel_max=3e-3, max_step_scale=3, use_errors=True, loss='linear', f_scale=1.0, ne_scale=1e19, Te_scale=1e3):
    """Fit Thomson scattering density and temperature with scaled B-splines.

    The optimizer works on ``ne / ne_scale`` and ``Te / Te_scale`` with the
    corresponding uncertainties scaled identically. Returned profiles,
    coefficients, and covariance matrices remain in physical units.
    """
    required_columns = {'R_mid', 'ne', 'ne_err', 'Te', 'Te_err'}
    if not isinstance(ds_TS, pd.DataFrame):
        raise TypeError('ds_TS must be a pandas DataFrame.')
    missing = required_columns.difference(ds_TS.columns)
    if missing:
        raise ValueError(f"ds_TS is missing required columns: {sorted(missing)}")
    for name, scale in [('ne_scale', ne_scale), ('Te_scale', Te_scale)]:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'{name} must be a finite positive number.')

    ds = ds_TS.copy()

    if R_knots is None:
        xt = find_x_means(ds['R_mid'], dR_channel_max)
    else:
        xt = np.asarray(R_knots, dtype=float)

    x_f = np.arange(xt[0], xt[-1] + dR_fine, dR_fine)
    ds_out = pd.DataFrame({'R_mid_spl': x_f})
    bmat = prepare_monotonic_decr_bspline_basis(xt, x_f, k=3)
    bmat_dx = prepare_monotonic_decr_bspline_basis(xt, x_f, nu=1, k=3)

    if sigma_conv > 0:
        x_kr = np.arange(-sigma_conv * 3, 3 * sigma_conv, dR_fine)
        gkernel = gaussian(x_kr, sigma_conv) * dR_fine
        bmat_conv = ndimage.convolve(bmat, gkernel[:, np.newaxis], mode='nearest')
    else:
        bmat_conv = bmat

    p_ne, pcov_ne = _fit_spl_new(
        bmat_conv,
        x_f,
        np.asarray(ds['R_mid']),
        np.asarray(ds['ne']) / ne_scale,
        np.asarray(ds['ne_err']) / ne_scale,
        max_step_scale=max_step_scale,
        use_errors=use_errors,
        loss=loss,
        f_scale=f_scale,
    )
    p_ne *= ne_scale
    pcov_ne *= ne_scale ** 2
    _save_spl_eval(ds_out, 'n', p_ne, bmat, bmat_dx, bmat_conv, pcov_ne)

    if sigma_conv > 0:
        bmat_ne_conv = (
            ndimage.convolve(bmat * ds_out['ne_spl'][:, np.newaxis], gkernel[:, np.newaxis], mode='nearest')
            / ds_out['ne_spl_conv'][:, np.newaxis]
        )
    else:
        bmat_ne_conv = bmat

    p_Te, pcov_Te = _fit_spl_new(
        bmat_ne_conv,
        x_f,
        np.asarray(ds['R_mid']),
        np.asarray(ds['Te']) / Te_scale,
        np.asarray(ds['Te_err']) / Te_scale,
        max_step_scale=max_step_scale,
        use_errors=use_errors,
        loss=loss,
        f_scale=f_scale,
    )
    p_Te *= Te_scale
    pcov_Te *= Te_scale ** 2
    _save_spl_eval(ds_out, 'T', p_Te, bmat, bmat_dx, bmat_ne_conv, pcov_Te)
    return ds_out, xt, p_ne, pcov_ne, p_Te, pcov_Te


def fit_ipfn_spline_ne_Te_TS(
    ds_TS,
    dR_fine=5e-4,
    R_knots=None,
    dR_channel_max=3e-3,
    ne_scale=1e19,
    Te_scale=1e3,
    s_factor=1.0,
    s=None,
    mono='decreasing',
    bc_type=None,
    maxiter=200,
    minimize_kw=None,
):
    """Fit independent IPFN constrained splines to density and temperature.

    Input values and uncertainties are scaled by ``ne_scale`` and
    ``Te_scale`` before fitting, while evaluated profiles and coefficients
    are returned in physical units.

    The returned profile DataFrame intentionally contains only fields shared
    with the project spline output that are consumed downstream. IPFN does
    not provide the covariance or convolution fields produced by the basis
    spline routine, so the corresponding tuple entries are ``None`` and
    those DataFrame columns are omitted.
    """
    try:
        from ipfnpytools.spline import constrained_spline
    except ImportError as exc:
        raise ImportError(
            'fit_ipfn_spline_ne_Te_TS requires ipfnpytools.spline.'
        ) from exc

    required_columns = {'R_mid', 'ne', 'ne_err', 'Te', 'Te_err'}
    if not isinstance(ds_TS, pd.DataFrame):
        raise TypeError('ds_TS must be a pandas DataFrame.')
    missing = required_columns.difference(ds_TS.columns)
    if missing:
        raise ValueError(f'ds_TS is missing required columns: {sorted(missing)}')
    if not np.isfinite(dR_fine) or dR_fine <= 0:
        raise ValueError('dR_fine must be a finite positive number.')
    for name, scale in [('ne_scale', ne_scale), ('Te_scale', Te_scale)]:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'{name} must be a finite positive number.')
    if not np.isfinite(s_factor) or s_factor < 0:
        raise ValueError('s_factor must be finite and non-negative.')
    if s is not None and (not np.isfinite(s) or s < 0):
        raise ValueError('s must be finite and non-negative when provided.')

    ds = ds_TS.dropna(subset=required_columns).copy()
    if ds.empty:
        raise ValueError('ds_TS contains no complete spline fitting rows.')

    x = ds['R_mid'].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('R_mid must contain finite values.')
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    knots = find_x_means(x, dR_channel_max) if R_knots is None else np.asarray(R_knots, dtype=float)
    if knots.ndim != 1 or knots.size < 4 or not np.isfinite(knots).all():
        raise ValueError('R_knots must contain at least four finite values.')
    if np.any(np.diff(knots) <= 0):
        raise ValueError('R_knots must be strictly increasing.')
    if knots[0] < np.min(x) or knots[-1] > np.max(x):
        raise ValueError('R_knots must lie within the fitting data range.')
    interior_knots = knots[1:-1]
    #interior_knots = knots[(knots > x_min) & (knots < x_max)]

    fit_kw = dict(minimize_kw or {})
    options = dict(fit_kw.pop('options', {}))
    options.setdefault('maxiter', maxiter)
    fit_kw['options'] = options

    def fit_quantity(value_column, error_column, scale):
        values = ds[value_column].to_numpy(dtype=float)
        errors = ds[error_column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(errors).all():
            raise ValueError(f'{value_column} and {error_column} must be finite.')
        if np.any(errors <= 0):
            raise ValueError(f'{error_column} must be strictly positive.')
        scaled_s = None if s is None else s / scale ** 2
        return constrained_spline(
            x=x,
            y=values / scale,
            w=scale / errors,
            t=interior_knots,
            mono=mono,
            bc_type=bc_type,
            sort_and_merge_duplicates=True,
            s_factor=s_factor,
            s=scaled_s,
            minimize_kw=fit_kw,
        )

    ne_fit = fit_quantity('ne', 'ne_err', ne_scale)
    te_fit = fit_quantity('Te', 'Te_err', Te_scale)

    x_f = np.arange(x_min, x_max, dR_fine)
    if x_f.size == 0 or x_f[-1] < x_max:
        x_f = np.append(x_f, x_max)
    ds_out = pd.DataFrame({'R_mid_spl': x_f})
    for quantity, spline, scale in [
        ('ne', ne_fit, ne_scale),
        ('Te', te_fit, Te_scale),
    ]:
        values = np.asarray(spline(x_f), dtype=float) * scale
        derivatives = np.asarray(spline.derivative()(x_f), dtype=float) * scale
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_values = -values / derivatives
        ds_out[f'{quantity}_spl'] = values
        ds_out[f'd{quantity}_spl'] = derivatives
        ds_out[f'lam_{quantity}_spl'] = lambda_values

    # Full clamped knot vector: the interior knots plus the data min/max repeated k+1 times.
    t_full = np.r_[np.repeat(x_min, 4), interior_knots, np.repeat(x_max, 4)]
    ds_out.attrs['t_full'] = t_full
    for quantity, spline, scale in [
        ('ne', ne_fit, ne_scale),
        ('Te', te_fit, Te_scale),
    ]:
        rebuilt = interpolate.BSpline(
            t_full, spline.get_coeffs(), 3, extrapolate=False)(x_f) * scale
        if not np.allclose(rebuilt, ds_out[f'{quantity}_spl'], rtol=1e-9, atol=1e-9 * scale):
            raise RuntimeError(f'{quantity} spline cannot be rebuilt from t_full and coefficients.')

    return (
        ds_out,
        knots,
        ne_fit.get_coeffs() * ne_scale,
        None,
        te_fit.get_coeffs() * Te_scale,
        None,
    )
