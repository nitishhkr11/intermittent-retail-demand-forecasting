"""Tests for the Syntetos-Boylan bias correction to Croston's forecast."""

import numpy as np
import pandas as pd
import pytest

from src.models.croston import CrostonForecaster
from src.models.sba import SBAForecaster


def test_sba_is_the_first_order_bias_correction_of_croston():
    history = pd.Series([0, 4, 0, 0, 2, 0, 0, 0, 3], dtype=float)
    alpha = 0.2
    croston = CrostonForecaster(alpha).fit(history)
    sba = SBAForecaster(alpha).fit(history)

    assert sba.name == "SBA(alpha=0.2)"
    assert sba.predict(5) == pytest.approx(
        np.full(5, (1.0 - alpha / 2.0) * croston.predict(1)[0])
    )
    assert sba.fit_diagnostics()["bias_correction_factor"] == pytest.approx(0.9)


def test_sba_preserves_crostons_undefined_all_zero_case():
    with pytest.raises(ValueError, match="all-zero training series"):
        SBAForecaster().fit(pd.Series([0.0] * 20))


@pytest.mark.parametrize("alpha", [0, -0.1, 1.1, np.nan, np.inf])
def test_sba_rejects_invalid_smoothing_weights(alpha):
    with pytest.raises(ValueError, match="alpha"):
        SBAForecaster(alpha)
