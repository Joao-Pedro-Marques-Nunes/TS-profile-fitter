import os
from importlib import import_module
import sys
import numpy as np
import pandas as pd
import aug_sfutils as sf
from aug_sfutils import mapeq
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt
from statsmodels.stats.weightstats import DescrStatsW
sys.path.insert(0, '/shares/departments/AUG/users/dsil/pymodules')
from get_functions import butter_bandpass_filter, smooth


class EquilibriumUnavailableError(RuntimeError):
    """Raised when the requested equilibrium shotfile is unavailable."""


def load_shot_list(path):
    """Load a shot list from CSV or Excel."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_intervals(path):
    """Load time interval definitions for Thomson scattering shots."""
    if path.endswith('.pkl'):
        df = pd.read_pickle(path)
        return _normalize_intervals(df)
    if path.endswith('.csv'):
        return _normalize_intervals(pd.read_csv(path))
    return _normalize_intervals(pd.read_excel(path))


def _normalize_intervals(df):
    """Return interval tables with the canonical pipeline column names."""
    columns = set(df.columns)
    conflicting_pairs = (
        ('shot_id', 'shot'),
        ('t_begin', 't_begin (s)'),
        ('t_end', 't_end (s)'),
        ('label', 'profile_type'),
    )
    conflicts = [f'{canonical} and {legacy}'
                 for canonical, legacy in conflicting_pairs
                 if canonical in columns and legacy in columns]
    if conflicts:
        raise ValueError(
            'Interval table contains ambiguous duplicate columns: '
            + ', '.join(conflicts))

    aliases = {
        'shot': 'shot_id',
        't_begin (s)': 't_begin',
        't_end (s)': 't_end',
        'profile_type': 'label',
    }
    normalized = df.rename(columns={key: value for key, value in aliases.items()
                                    if key in df.columns}).copy()
    required = {'shot_id', 't_begin', 't_end'}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f'Interval table is missing required columns: {missing}')
    if 'label' not in normalized.columns:
        normalized['label'] = ''
    normalized['shot_id'] = normalized['shot_id'].astype(int)
    normalized['t_begin'] = normalized['t_begin'].astype(float)
    normalized['t_end'] = normalized['t_end'].astype(float)
    return normalized


def _to_array(obj):
    if hasattr(obj, 'data'):
        return np.asarray(obj.data)
    try:
        return np.asarray(obj)
    except Exception:
        return np.asarray(getattr(obj, 'values', obj))


def _build_vta_dataframe(vta):
    """Build a DataFrame with VTA core/edge TS signals."""
    t_c = _to_array(vta('TIM_CORE'))
    t_e = _to_array(vta('TIM_EDGE'))

    r_core = _to_array(vta.getobject('R_core'))
    z_core = _to_array(vta.getobject('Z_core'))
    r_edge = _to_array(vta.getobject('R_edge'))
    z_edge = _to_array(vta.getobject('Z_edge'))

    Te_c = _to_array(vta.getobject('Te_c'))
    SigTe_c = _to_array(vta.getobject('SigTe_c'))
    Ne_c = _to_array(vta.getobject('Ne_c'))
    SigNe_c = _to_array(vta.getobject('SigNe_c'))

    Te_e = _to_array(vta.getobject('Te_e'))
    SigTe_e = _to_array(vta.getobject('SigTe_e'))
    Ne_e = _to_array(vta.getobject('Ne_e'))
    SigNe_e = _to_array(vta.getobject('SigNe_e'))

    def _flatten(system, t, r, z, Te, SigTe, Ne, SigNe):
        t = np.asarray(t, dtype=float).ravel()
        r = np.asarray(r, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        Te = np.asarray(Te, dtype=float)
        SigTe = np.asarray(SigTe, dtype=float)
        Ne = np.asarray(Ne, dtype=float)
        SigNe = np.asarray(SigNe, dtype=float)

        if Te.ndim != 2 or Ne.ndim != 2:
            raise ValueError(
                f"Expected 2D signal arrays for {system}, got shape {Te.shape} and {Ne.shape}")
        if Te.shape != Ne.shape or SigTe.shape != Te.shape or SigNe.shape != Te.shape:
            raise ValueError(
                f"Signal arrays must share the same shape for {system}, got Te={Te.shape}, Ne={Ne.shape}, SigTe={SigTe.shape}, SigNe={SigNe.shape}")
        if Te.shape[0] != t.size:
            raise ValueError(
                f"Time length does not match signal rows for {system}")

        n_channels = Te.shape[1]
        if r.ndim == 2:
            if r.shape == (t.size, n_channels):
                r_mesh = r
            else:
                raise ValueError(f"Radius array shape {r.shape} is incompatible for {system}")
        elif r.size == n_channels:
            r_mesh = np.repeat(r[None, :], t.size, axis=0)
        elif r.size == t.size:
            r_mesh = np.repeat(r[:, None], n_channels, axis=1)
        elif r.size == 1:
            r_mesh = np.full((t.size, n_channels), r[0], dtype=float)
        else:
            raise ValueError(f"Radius length {r.size} is incompatible for {system}")

        if z.ndim == 2:
            if z.shape == (t.size, n_channels):
                z_mesh = z
            else:
                raise ValueError(f"Height array shape {z.shape} is incompatible for {system}")
        elif z.size == n_channels:
            z_mesh = np.repeat(z[None, :], t.size, axis=0)
        elif z.size == t.size:
            z_mesh = np.repeat(z[:, None], n_channels, axis=1)
        elif z.size == 1:
            z_mesh = np.full((t.size, n_channels), z[0], dtype=float)
        else:
            raise ValueError(f"Height length {z.size} is incompatible for {system}")

        t_mesh = np.repeat(t[:, None], n_channels, axis=1)

        return pd.DataFrame({
            'system': system,
            'time': t_mesh.ravel(),
            'r': r_mesh.ravel(),
            'z': z_mesh.ravel(),
            'Te': Te.ravel(),
            'Te_err': SigTe.ravel(),
            'Ne': Ne.ravel(),
            'Ne_err': SigNe.ravel(),
        })

    df_core = _flatten('core', t_c, r_core, z_core, Te_c, SigTe_c, Ne_c, SigNe_c)
    df_edge = _flatten('edge', t_e, r_edge, z_edge, Te_e, SigTe_e, Ne_e, SigNe_e)
    df = pd.concat([df_core, df_edge], ignore_index=True)
    df['r_midplane'] = df['r']
    return df


def load_vta(shot_id, diagnostic='VTA'):
    """Load the VTA shotfile for a single shot."""
    vta = sf.SFREAD(shot_id, diagnostic)
    if not getattr(vta, 'status', False):
        raise RuntimeError(
            f"Unable to open VTA shotfile for shot {shot_id}, diagnostic {diagnostic}")
    return vta


def load_equilibrium(shot_id, diag='EQH'):
    """Load the equilibrium object for a shot."""
    equilibrium = sf.EQU(shot_id, diag=diag)
    try:
        time = np.asarray(equilibrium.time)
        z_mag = np.asarray(equilibrium.Zmag)
    except (AttributeError, TypeError, ValueError) as exc:
        raise EquilibriumUnavailableError(
            f'Unable to open usable {diag} equilibrium for shot {shot_id}'
        ) from exc
    if time.size < 2 or z_mag.size != time.size:
        raise EquilibriumUnavailableError(
            f'Unable to open usable {diag} equilibrium for shot {shot_id}'
        )
    return equilibrium


def load_ts_data(shot_id, data_dir=None, diagnostic='VTA', equilibrium='EQH'):
    """Load TS data and the equilibrium object for a single shot."""
    vta = load_vta(shot_id, diagnostic=diagnostic)
    equ = load_equilibrium(shot_id, diag=equilibrium)
    df = _build_vta_dataframe(vta)
    return {
        'shot_id': shot_id,
        'vta': vta,
        'equ': equ,
        'data': df,
    }


def load_elm_times(shot_id, diagnostic='ELM'):
    """Load ELM start and end times for a shot."""
    elm = _open_optional_shotfile(shot_id, diagnostic)
    if elm is None:
        raise RuntimeError(
            f"Unable to open {diagnostic} shotfile for shot {shot_id}")
    try:
        begins = _to_array(elm.getobject('t_begELM')).astype(float).ravel()
        ends = _to_array(elm.getobject('t_endELM')).astype(float).ravel()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read ELM start/end times for shot {shot_id}") from exc
    if begins.size != ends.size:
        raise ValueError('ELM start and end arrays must have the same length.')
    return begins, ends


def _interval_bounds(interval):
    if hasattr(interval, 'get'):
        t_begin = interval.get('t_begin')
        t_end = interval.get('t_end')
    else:
        t_begin, t_end = interval
    if t_begin is None or t_end is None:
        raise ValueError('interval must define t_begin and t_end.')
    return float(t_begin), float(t_end)


def _read_interval_signal(shotfile, name, t_begin, t_end, scale=1.0):
    values = _to_array(shotfile.getobject(name, tbeg=t_begin, tend=t_end))
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmean(finite) * scale)


def _read_optional_interval_signal(shotfile, name, t_begin, t_end, scale=1.0):
    if shotfile is None or not getattr(shotfile, 'status', False):
        return np.nan
    try:
        return _read_interval_signal(shotfile, name, t_begin, t_end, scale)
    except Exception:
        return np.nan


def _open_optional_shotfile(shot_id, diagnostic, **kwargs):
    try:
        shotfile = sf.SFREAD(shot_id, diagnostic, **kwargs)
    except Exception:
        return None
    return shotfile if getattr(shotfile, 'status', False) else None


def load_physics_context(
    shot_id,
    equilibrium='EQH',
    equilibrium_object=None,
    tth_experiment=None,
    bpt_experiment='DAVIDP',
):
    """Open reusable shot-level objects for interval physics calculations."""
    equ = equilibrium_object
    if equ is None:
        equ = load_equilibrium(shot_id, diag=equilibrium)
    tth_kwargs = {} if tth_experiment is None else {'exp': tth_experiment}
    return {
        'equilibrium': equ,
        'TTH': _open_optional_shotfile(shot_id, 'TTH', **tth_kwargs),
        'BPT': _open_optional_shotfile(
            shot_id, 'BPT', experiment=bpt_experiment),
        'TOT': _open_optional_shotfile(shot_id, 'TOT'),
        'ELM': _open_optional_shotfile(shot_id, 'ELM'),
        'GQH': _open_optional_shotfile(shot_id, 'GQH'),
        'FPC': _open_optional_shotfile(shot_id, 'FPC'),
        'MBI': _open_optional_shotfile(shot_id, 'MBI'),
        'IDZ': _open_optional_shotfile(shot_id, 'IDZ'),
    }


def _load_fast_wmhd_dwdt(
    shot_id,
    t_begin,
    t_end,
    equilibrium,
    t_beg_elm,
    elm_fraction,
):
    """Estimate dWmhd/dt with Davide's fast-Wmhd Field method."""

    gqh_kwargs = {
        name: getattr(equilibrium, name)
        for name in ('exp', 'diag', 'ed')
        if getattr(equilibrium, name, None) is not None
    }
    gqh = _open_optional_shotfile(shot_id, 'GQH', **gqh_kwargs)
    if gqh is None:
        raise RuntimeError('Fast GQH Wmhd is unavailable.')

    wmhd = _to_array(gqh.getobject(
        'Wmhd', tbeg=t_begin, tend=t_end)).astype(float).ravel()
    wmhd_time = _to_array(gqh.gettimebase(
        'Wmhd', tbeg=t_begin, tend=t_end)).astype(float).ravel()
    finite = np.isfinite(wmhd) & np.isfinite(wmhd_time)
    wmhd = wmhd[finite]
    wmhd_time = wmhd_time[finite]
    order = np.argsort(wmhd_time)
    wmhd_time = wmhd_time[order]
    wmhd = wmhd[order]
    if wmhd.size < 2 or np.any(np.diff(wmhd_time) <= 0.0):
        raise RuntimeError('Fast Wmhd timebase is invalid.')

    dt = float(np.mean(np.diff(wmhd_time)))
    n_samples = int(np.ceil((wmhd_time[-1] - wmhd_time[0]) / dt))
    if not np.isfinite(dt) or dt <= 0.0 or n_samples < 2:
        raise RuntimeError('Fast Wmhd timebase is invalid.')
    wmhd_interp_time = np.linspace(wmhd_time.min(), wmhd_time.max(), n_samples)
    wmhd_interp = interp1d(wmhd_time, wmhd)(wmhd_interp_time)

    from get_functions import dWdt_Field

    dwdt_mw, dwdt_err_mw = dWdt_Field(
        wmhd_interp_time,
        wmhd_interp,
        t_beg_elm,
        shot_id,
        float(elm_fraction[0]),
        float(elm_fraction[1]),
    )
    return float(dwdt_mw) * 1e6, float(dwdt_err_mw) * 1e6


def _read_timeseries_signal(shotfile, name):
    if shotfile is None or not getattr(shotfile, 'status', False):
        return None, None
    try:
        values = _to_array(shotfile.getobject(name)).astype(float).ravel()
        times = _to_array(shotfile.gettimebase(name)).astype(float).ravel()
    except Exception:
        return None, None
    if values.size != times.size:
        return None, None
    finite = np.isfinite(values) & np.isfinite(times)
    if not np.any(finite):
        return None, None
    times = times[finite]
    values = values[finite]
    order = np.argsort(times)
    return times[order], values[order]


def load_power_signals(shot_id, interval=None):
    """Load shot-level power, stored energy, and ELM timing signals.

    Signals retain their native SI units and timebases. Missing optional
    diagnostics are returned as ``None`` so plotting can show what exists.
    """
    tot = _open_optional_shotfile(shot_id, 'TOT')
    tth = _open_optional_shotfile(shot_id, 'TTH')
    gqh = _open_optional_shotfile(shot_id, 'GQH')
    bpt = _open_optional_shotfile(shot_id, 'BPT', experiment='DAVIDP')
    elm = _open_optional_shotfile(shot_id, 'ELM')
    signals = {}
    for key, shotfile, signal_name in (
        ('P_TOT', tot, 'P_TOT'),
        ('P_Net', tth, 'P_NET'),
        ('dWdt_TOT', tot, 'dWmhd/dt'),
        ('W_mhd', gqh, 'Wmhd'),
        ('Pr_sep', bpt, 'Pr_sep'),
        ('f_ELM', elm, 'f_ELM'),
    ):
        time, values = _read_timeseries_signal(shotfile, signal_name)
        signals[key] = {'time': time, 'values': values, 'source': signal_name}
    signals['P_abs'] = signals['P_Net']

    elm_times = None
    if elm is not None:
        try:
            elm_times = _to_array(elm.getobject('t_begELM')).astype(float).ravel()
            elm_times = np.sort(elm_times[np.isfinite(elm_times)])
        except Exception:
            elm_times = None
    elm_frequency = np.nan
    if interval is not None and elm is not None:
        t_begin, t_end = _interval_bounds(interval)
        elm_frequency = _read_optional_interval_signal(
            elm, 'f_ELM', t_begin, t_end)
    wmhd_time = signals['W_mhd']['time']
    if wmhd_time is not None and wmhd_time.size > 1:
        steps = np.diff(wmhd_time)
        steps = steps[np.isfinite(steps) & (steps > 0.0)]
        wmhd_frequency = float(1.0 / np.median(steps)) if steps.size else np.nan
    else:
        wmhd_frequency = np.nan
    if elm_times is not None and elm_times.size > 1:
        periods = np.diff(elm_times)
        periods = periods[np.isfinite(periods) & (periods > 0.0)]
        if not np.isfinite(elm_frequency):
            elm_frequency = float(1.0 / np.median(periods)) if periods.size else np.nan
    elif not np.isfinite(elm_frequency):
        elm_frequency = np.nan
    return {
        'signals': signals,
        't_beg_elm': elm_times,
        'elm_frequency': elm_frequency,
        'wmhd_frequency': wmhd_frequency,
    }

def load_inter_elm_dwdt(
    shot_id,
    t_begin,
    t_end,
    t_beg_elm=None,
    elm_fraction=(0.75, 0.95),
    filter_order=3,
    fast_wmhd_time = None,
    fast_wmhd_values = None,
):
    """Estimate inter-ELM dWmhd/dt using Davide's Field method.

    This follows ``dWdt_Field``: band-pass Wmhd from 4 Hz to seven times the
    ELM frequency, differentiate it, select the requested inter-ELM phase,
    smooth the selected derivative, and calculate its weighted mean. The
    upper band edge is capped below the Wmhd Nyquist frequency when needed.
    """
    t_begin = float(t_begin)
    t_end = float(t_end)
    phase_begin, phase_end = map(float, elm_fraction)
    if not 0.0 <= phase_begin < phase_end <= 1.0:
        raise ValueError('elm_fraction must satisfy 0 <= start < end <= 1.')
    if t_end <= t_begin:
        raise ValueError('Invalid interval or filtering parameters.')
    
    if t_beg_elm is None:
        t_beg_elm = None
        elm = _open_optional_shotfile(shot_id, 'ELM')
        if elm is None:
            raise RuntimeError('ELM timing is unavailable.')
        t_beg_elm = elm.getobject(
            't_begELM', tbeg=t_begin, tend=t_end)
    if fast_wmhd_time is not None and fast_wmhd_values is not None:
        wmhd_time = fast_wmhd_time
        wmhd = fast_wmhd_values
    else:
        gqh = _open_optional_shotfile(shot_id, 'GQH')
        if gqh is None:
            raise RuntimeError('Fast GQH Wmhd is unavailable.')
        wmhd = _to_array(gqh.getobject(
            'Wmhd', tbeg=t_begin, tend=t_end)).astype(float).ravel()
        wmhd_time = _to_array(gqh.gettimebase(
            'Wmhd', tbeg=t_begin, tend=t_end)).astype(float).ravel()
    # gqh = _open_optional_shotfile(shot_id, 'GQH')
    # if gqh is None:
    #     raise RuntimeError('Fast GQH Wmhd is unavailable.')
    # wmhd = _to_array(gqh.getobject(
    #     'Wmhd', tbeg=t_begin, tend=t_end)).astype(float).ravel()
    # wmhd_time = _to_array(gqh.gettimebase(
    #     'Wmhd', tbeg=t_begin, tend=t_end)).astype(float).ravel()
    elm_time = _to_array(t_beg_elm).astype(float).ravel()
    finite = np.isfinite(wmhd) & np.isfinite(wmhd_time)
    wmhd = wmhd[finite]
    wmhd_time = wmhd_time[finite]
    elm_time = np.unique(np.sort(elm_time[np.isfinite(elm_time)]))
    if wmhd.size < 8 or elm_time.size < 2:
        raise RuntimeError('Insufficient fast Wmhd or ELM timing samples.')

    order = np.argsort(wmhd_time)
    wmhd_time = wmhd_time[order]
    wmhd = wmhd[order]
    time_steps = np.diff(wmhd_time)
    dt_filter = float(np.median(time_steps))
    dt = float(np.mean(time_steps))
    if not np.isfinite(dt_filter) or dt_filter <= 0.0 or not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError('Fast Wmhd timebase is invalid.')
    elm_gaps = np.diff(elm_time)
    valid_gaps = np.isfinite(elm_gaps) & (elm_gaps > 0.0)
    if not np.any(valid_gaps):
        raise RuntimeError('ELM timing does not contain a valid gap.')
    elm_frequency = float(np.nanmean(1.0 / elm_gaps[valid_gaps]))
    sampling_frequency = 1.0 / dt
    nyquist = 0.5 * sampling_frequency
    highcut = min(elm_frequency * 7.0, 5000.0, 0.99 * nyquist)
    if not np.isfinite(highcut) or highcut <= 4.0:
        raise RuntimeError('Wmhd sampling rate is too low for the Field band-pass filter.')
    filtered_wmhd = butter_bandpass_filter(
        wmhd, 4.0, highcut, sampling_frequency, order=filter_order)
    dwdt = np.gradient(filtered_wmhd, wmhd_time, edge_order=2)
    selected = np.zeros(len(dwdt), dtype=bool)
    for cycle_start, cycle_end in zip(elm_time[:-1], elm_time[1:]):
        cycle_length = cycle_end - cycle_start
        if cycle_length <= 0.0:
            continue
        window_start = cycle_start + phase_begin * cycle_length
        window_end = cycle_start + phase_end * cycle_length
        window_mask = (wmhd_time >= window_start) & (wmhd_time <= window_end)
        selected |= window_mask
    selected_inter_elm = dwdt[selected & np.isfinite(dwdt)]
    if selected_inter_elm.size == 0:
        raise RuntimeError('No Wmhd samples fall in the requested inter-ELM phase.')
    _, dwdt_smooth, dwdt_err = smooth(
        wmhd_time[selected & np.isfinite(dwdt)],
        dwdt[selected & np.isfinite(dwdt)],
    )
    weights = 1.0 / np.square(dwdt_err)
    dwdt_stats = DescrStatsW(dwdt_smooth, weights=weights)
    return float(dwdt_stats.mean), float(dwdt_stats.std), wmhd, wmhd_time


def _read_eqh_outer_midplane(equilibrium, time):
    try:
        r_midplane, _ = equilibrium.rhoTheta2rz(
            1.0, 0.0, t_in=time, coord_in='rho_pol')
        return float(np.asarray(r_midplane).ravel()[0])
    except Exception:
        try:
            r_midplane, _ = equilibrium.rhoTheta2rz(
                np.asarray([[1.0]]), 0.0, t_in=[time],
                coord_in='rho_pol', n_line=1)
            return float(np.asarray(r_midplane).ravel()[0])
        except Exception:
            return np.nan


def _read_eqh_outer_midplane_fields(equilibrium, time):
    """Return ``R_out``, ``B_pol``, and ``B_tor`` from the equilibrium field."""
    try:
        r_out, z_out = mapeq.rhoTheta2rz(
            equilibrium, 1.0, 0.0, t_in=time)
        r_out = float(np.asarray(r_out).ravel()[0])
        z_out = float(np.asarray(z_out).ravel()[0])
        br_out, bz_out, btor_out = mapeq.rz2brzt(
            equilibrium, r_out, z_out, t_in=time)
        br_out = float(np.asarray(br_out).ravel()[0])
        bz_out = float(np.asarray(bz_out).ravel()[0])
        btor_out = float(np.asarray(btor_out).ravel()[0])
        bpol_out = float(np.hypot(br_out, bz_out))
        return r_out, bpol_out, btor_out
    except Exception:
        return np.nan, np.nan, np.nan


def load_physics_inputs(
    shot_id,
    interval,
    equilibrium='EQH',
    tth_experiment=None,
    bpt_experiment='DAVIDP',
    dWdt_method='paper',
    elm_fraction=(0.75, 0.95),
    physics_context=None,
):
    """Load interval-averaged power-balance and geometry inputs.

    Values are returned in SI units for the physics-based separatrix model:
    power in watts, radius in meters, and magnetic fields in tesla. Missing
    diagnostics are represented by ``np.nan`` and recorded in ``sources``.
    """
    t_begin, t_end = _interval_bounds(interval)
    time = 0.5 * (t_begin + t_end)
    context = physics_context
    if context is None:
        context = load_physics_context(
            shot_id,
            equilibrium=equilibrium,
            tth_experiment=tth_experiment,
            bpt_experiment=bpt_experiment,
        )
    equ = context['equilibrium']
    tth = context['TTH']
    bpt = context['BPT']
    tot = context['TOT']
    elm = context['ELM']
    gqh = context['GQH']
    fpc = context['FPC']
    mbi = context['MBI']
    idz = context['IDZ']

    P_abs = _read_optional_interval_signal(tth, 'P_NET', t_begin, t_end)
    P_rsep = _read_optional_interval_signal(bpt, 'Pr_sep', t_begin, t_end)
    f_elm = _read_optional_interval_signal(elm, 'f_ELM', t_begin, t_end)
    if dWdt_method not in {'tot', 'inter_elm', 'paper', 'fast_wmhd'}:
        raise ValueError(
            "dWdt_method must be 'tot', 'inter_elm', 'paper', or 'fast_wmhd'."
        )
    if not np.isfinite(f_elm):
        dWdt = 0.0
        dWdt_err = 0.0
        dWdt_source = 'fallback:f_ELM unavailable -> 0'
    elif dWdt_method == 'tot':
        dWdt = _read_optional_interval_signal(tot, 'dWmhd/dt', t_begin, t_end)
        dWdt_err = np.nan
        dWdt_source = 'TOT:dWmhd/dt'
    elif dWdt_method == 'paper':
        dWdt = 0.25 * P_abs if np.isfinite(P_abs) else np.nan
        dWdt_err = 0.065 * P_abs if np.isfinite(P_abs) else np.nan
        dWdt_source = '0.25*TTH:P_NET'
    elif dWdt_method == 'inter_elm':
        if elm is None:
            raise RuntimeError('ELM timing is unavailable for inter-ELM dWdt.')
        t_beg_elm = elm.getobject(
            't_begELM', tbeg=t_begin, tend=t_end)
        dWdt, dWdt_err, _, _ = load_inter_elm_dwdt(
            shot_id, t_begin, t_end, t_beg_elm=t_beg_elm,
            elm_fraction=elm_fraction)
        dWdt_source = 'GQH:Wmhd inter-ELM (75%-95%)'
    else:
        if elm is None:
            raise RuntimeError('ELM timing is unavailable for fast-Wmhd dWdt.')
        t_beg_elm = elm.getobject(
            't_begELM', tbeg=t_begin, tend=t_end )
        dWdt, dWdt_err = _load_fast_wmhd_dwdt(
            shot_id, t_begin, t_end, equ, t_beg_elm, elm_fraction)
        dWdt_source = 'GQH:Wmhd fast Field method'

    R_geo = _read_optional_interval_signal(gqh, 'Rgeo', t_begin, t_end)
    a_geo = _read_optional_interval_signal(gqh, 'ageo', t_begin, t_end)
    if not np.isfinite(R_geo) or not np.isfinite(a_geo):
        r_in = _read_optional_interval_signal(gqh, 'Rin', t_begin, t_end)
        r_out = _read_optional_interval_signal(gqh, 'Raus', t_begin, t_end)
        if np.isfinite(r_in) and np.isfinite(r_out):
            R_geo = 0.5 * (r_in + r_out)
            a_geo = 0.5 * (r_out - r_in)
    kappa = _read_optional_interval_signal(gqh, 'k', t_begin, t_end)
    del_u = _read_optional_interval_signal(gqh, 'delRoben', t_begin, t_end)
    del_l = _read_optional_interval_signal(gqh, 'delRuntn', t_begin, t_end)
    Ip = _read_optional_interval_signal(fpc, 'IpiFP', t_begin, t_end)
    B_tor_mbi = _read_optional_interval_signal(mbi, 'BTFABB', t_begin, t_end)
    Z_eff = _read_optional_interval_signal(idz, 'Zeff', t_begin, t_end)
    Z_eff = 1.5
    R_out, B_pol_eq, B_tor_eq = _read_eqh_outer_midplane_fields(equ, time)
    if not np.isfinite(R_out):
        R_out = _read_eqh_outer_midplane(equ, time)

    del_av = 0.5 * (del_u + del_l)
    kappa_hat = np.sqrt(
        (1.0 + kappa**2 * (1.0 + 2.0 * del_av**2 - 1.2 * del_av**3)) / 2.0)
    B_pol_analytic = 4e-7 * np.pi * Ip / (2.0 * np.pi * a_geo * kappa_hat)
    B_pol = B_pol_eq if np.isfinite(B_pol_eq) else B_pol_analytic
    B_tor = B_tor_eq if np.isfinite(B_tor_eq) else B_tor_mbi
    print('Magnetic Fields:')
    print(f'B_pol_analytic: {B_pol_analytic}, B_pol_out: {B_pol_eq}, B_tor_out: {B_tor_eq}, B_tor_mbi: {B_tor_mbi}')
    return {
        'power_balance': {
            'P_abs': P_abs,
            'P_rsep': P_rsep,
            'dWdt': dWdt,
            'dWdt_err': dWdt_err,
        },
        'geometry': {
            'R_out': R_out,
            'B_pol': B_pol,
            'B_tor': B_tor,
            'Z_eff': Z_eff,
        },
        'metadata': {
            'shot_id': shot_id,
            't_begin': t_begin,
            't_end': t_end,
            'time': time,
            'equilibrium': equilibrium,
            'f_ELM': f_elm,
            'sources': {
                'P_abs': 'TTH:P_NET',
                'P_rsep': 'BPT:Pr_sep',
                'dWdt': dWdt_source,
                'geometry': 'EQH:r/z/B (outer midplane), GQH/FPC/MBI fallback',
            },
        },
    }
