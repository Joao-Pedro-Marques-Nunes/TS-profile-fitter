import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ts_profile_fitter.fit_metrics import compute_fit_metrics


def test_compute_fit_metrics_returns_unweighted_metrics():
    result = compute_fit_metrics(
        y_observed=np.array([1.0, 2.0, 4.0]),
        y_fitted=np.array([1.5, 1.0, 5.0]),
    )

    assert result['n_points'] == 3
    assert np.isclose(result['bias'], -0.5 / 3)
    assert np.isclose(result['mae'], 2.5 / 3.0)
    assert np.isclose(result['sse'], 2.25)
    assert np.isclose(result['rmse'], np.sqrt(0.75))
    assert np.isnan(result['weighted_sse'])


def test_compute_fit_metrics_returns_weighted_chi_square():
    result = compute_fit_metrics(
        y_observed=np.array([1.0, 2.0, 4.0]),
        y_fitted=np.array([1.5, 1.0, 5.0]),
        y_error=np.array([0.5, 2.0, 1.0]),
        n_parameters=1,
    )

    expected_chi2 = (-1.0) ** 2 + (0.5) ** 2 + (1.0) ** 2
    assert np.isclose(result['weighted_sse'], expected_chi2)
    assert np.isclose(result['reduced_chi2'], expected_chi2 / 2)


def test_compute_fit_metrics_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match='same size'):
        compute_fit_metrics([1.0, 2.0], [1.0])