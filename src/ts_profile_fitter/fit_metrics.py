"""Common fit-quality metrics for profile comparisons."""

import numpy as np


def compute_fit_metrics(
    y_observed,
    y_fitted,
    y_error=None,
    n_parameters=None,
):
    """Compute common residual and uncertainty-weighted fit metrics.

    Residuals use the convention ``y_observed - y_fitted``. Non-finite
    observations, fits, and errors are excluded from the corresponding
    calculation. The unweighted metrics require only observations and fits;
    weighted chi-square metrics are returned when ``y_error`` is provided.
    """
    observed = np.asarray(y_observed, dtype=float).ravel()
    fitted = np.asarray(y_fitted, dtype=float).ravel()
    if observed.size != fitted.size:
        raise ValueError('y_observed and y_fitted must have the same size.')
    if observed.size == 0:
        raise ValueError('y_observed and y_fitted must not be empty.')

    residual = observed - fitted
    valid = np.isfinite(residual)
    residual = residual[valid]
    if residual.size == 0:
        raise ValueError('No finite residuals are available.')

    result = {
        'n_points': int(residual.size),
        'bias': float(np.mean(residual)),
        'mae': float(np.mean(np.abs(residual))),
        'sse': float(np.sum(residual ** 2)),
        'mse': float(np.mean(residual ** 2)),
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'max_abs_error': float(np.max(np.abs(residual))),
        'weighted_sse': np.nan,
        'reduced_chi2': np.nan,
    }

    if y_error is None:
        return result

    errors = np.asarray(y_error, dtype=float).ravel()
    if errors.size != observed.size:
        raise ValueError('y_error must have the same size as y_observed.')
    weighted_valid = valid & np.isfinite(errors) & (errors > 0.0)
    if not np.any(weighted_valid):
        raise ValueError('No finite positive uncertainties are available.')

    standardized_residual = (
        (observed[weighted_valid] - fitted[weighted_valid])
        / errors[weighted_valid]
    )
    weighted_sse = float(np.sum(standardized_residual ** 2))
    result['weighted_sse'] = weighted_sse
    if n_parameters is not None:
        degrees_of_freedom = int(np.sum(weighted_valid)) - int(n_parameters)
        if degrees_of_freedom <= 0:
            raise ValueError('n_parameters must be smaller than the valid point count.')
        result['reduced_chi2'] = weighted_sse / degrees_of_freedom

    return result