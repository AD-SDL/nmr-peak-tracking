"""Test tools built atop trackpy"""
import numpy as np

from pytest import mark

from nmrtrack.trackpy import PeakMovementPredict


@mark.parametrize('order', [2, 3, 4])
def test_poly_fit(order):
    f = PeakMovementPredict(order=3)

    # Make some sample data
    x = np.arange(-1, -4, -1)
    y = 8.1 + 0.01 * x ** 2 - 0.001 * x ** 3
    poly = f.fit_predictor(x, y)
    assert poly.deriv(1)(0) == 0.


def test_history():
    f = PeakMovementPredict(order=3, min_history=8, max_history=16)

    x = np.arange(-1, -32, -1)
    y = 8.1 + 0.01 * x ** 2 - 0.001 * x ** 3

    # Make sure that if we do less than 4 than we fit a linear model
    poly = f.fit_predictor(x[:7], y[:7])
    assert poly.coef.shape == (2,)

    # Make sure that above we fit a polynomial
    poly = f.fit_predictor(x, y)
    assert poly.coef.shape == (4,)

    # Fill the later entries with noise
    y[:8] = 10000
    poly_2 = f.fit_predictor(x, y)
    assert np.isclose(poly_2.coef, poly.coef).all()
