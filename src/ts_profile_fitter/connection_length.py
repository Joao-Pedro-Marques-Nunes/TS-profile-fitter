"""Connection-length acquisition for separatrix calculations."""

from __future__ import annotations

from typing import Callable, Mapping, Optional


def interval_midpoint(interval):
    """Return the midpoint of an interval mapping or two-item sequence."""
    if hasattr(interval, 'get'):
        t_begin = interval.get('t_begin')
        t_end = interval.get('t_end')
    else:
        t_begin, t_end = interval
    if t_begin is None or t_end is None:
        raise ValueError('interval must define t_begin and t_end.')
    return 0.5 * (float(t_begin) + float(t_end))


def get_connection_length(
    shot_id: int,
    interval,
    provider: Callable = None,
    dR: float = 1e-3,
    **provider_kwargs,
):
    """Get the outer-midplane connection length for one shot interval.

    Parameters
    ----------
    shot_id : int
        AUG shot number.
    interval : mapping or two-item sequence
        Interval with ``t_begin`` and ``t_end`` in seconds.
    provider : callable
        Field-line routine, typically the legacy ``get_LcSOL`` function. It
        must accept ``(shot_id, time, dR, **provider_kwargs)`` and return
        either a scalar connection length or a tuple whose first item is the
        connection length.
    dR : float
        Midplane offset passed to the field-line provider.

    A provider is required because the legacy field-line routine depends on
    compiled machine-specific extensions and is not part of this package.
    """
    if provider is None:
        raise ValueError(
            'A connection-length provider is required. Pass the legacy '
            'get_LcSOL function through provider=.'
        )

    time = interval_midpoint(interval)
    result = provider(shot_id, time, dR, **provider_kwargs)
    if isinstance(result, tuple):
        result = result[0]
    return float(result)


__all__ = ['interval_midpoint', 'get_connection_length']
