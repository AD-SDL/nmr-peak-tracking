"""Tests around generating synthetic peaks"""
from scipy.integrate import quad
import numpy as np

from nmrtrack.synthetic import generate_peak, PatternGenerator, MultiplePeakFunctions, MobilePeakFunction, TimeSeriesGenerator


def test_single_peak():
    """Test generating a single set of peaks"""

    # Start easy: single peak
    peak_fn = generate_peak(0, area=1, width=0.01)
    assert peak_fn(0.) == [100.0 / np.pi]
    assert np.isclose(quad(peak_fn, -100, 100)[0], 1.0, atol=1e-3)

    peak_fn = generate_peak(1, area=1, width=0.01)
    assert peak_fn(1.) == [100.0 / np.pi]
    assert np.isclose(quad(peak_fn, -100, 100)[0], 1.0, atol=1e-3)

    # Now for a doublet and triplet
    peak_fn = generate_peak(0, area=1, width=0.01, multiplicity=(2,), coupling_offsets=(0.1,))
    assert np.isclose(quad(peak_fn, -100, 100)[0], 1.0, atol=1e-3)  # Should still have an area of 1

    peak_fn = generate_peak(0, area=1, width=0.01, multiplicity=(3,), coupling_offsets=(1,))  # Far enough apart for no peak overlap
    assert np.isclose(quad(peak_fn, -100, 100)[0], 1.0, atol=1e-3)  # Should still have an area of 1
    assert np.isclose(peak_fn(0) / peak_fn(1), 2., atol=1e-2)

    # Doublet of doublets
    peak_fn = generate_peak(0, area=1, width=0.01, multiplicity=(2, 2), coupling_offsets=(0.1, 0.01))
    assert np.isclose(quad(peak_fn, -100, 100)[0], 1.0, atol=1e-3)  # Should still have an area of 1


def test_combine():
    peak_a = generate_peak(0, area=1, width=0.01)
    peak_b = generate_peak(0.5, area=1, width=0.01)
    pattern = MultiplePeakFunctions([peak_a, peak_b])
    assert np.isclose(peak_a(0) + peak_b(0), pattern(0))

    # Test the integral
    assert np.isclose(quad(pattern, -10, 10)[0], 2., atol=1e-2)


def test_generate_static():
    generator = PatternGenerator()

    # Ensure the pattern has the same shape of the offsets
    #  TODO (wardlt): Come up with more tests
    offsets = generator.offsets
    generator.num_to_generate = 32
    for info, pattern in generator.generate_patterns():
        assert offsets.shape == pattern.shape


def test_shift():
    peak_fn = generate_peak(0, area=1, width=0.01)
    shifted_fn = peak_fn.alter_pattern(1, 1)
    assert peak_fn(0) == shifted_fn(1)

    shifted_fn = peak_fn.alter_pattern(0, scale=0.5)
    assert peak_fn(0) == 2 * shifted_fn(0)

    shifted_fn = peak_fn.alter_pattern(-1, scale=0.5)
    assert peak_fn(0) == 2 * shifted_fn(-1)


def test_mobile_peak():
    # Only movement
    peak_fun = generate_peak(0, area=1, width=0.01)
    mobile_peak_fun = MobilePeakFunction(peak=peak_fun, time_steps=64, offset_knots=[0.7, 1.])
    shifted_peak = mobile_peak_fun.apply_movement(32)
    assert shifted_peak.center == 0.7
    assert shifted_peak.area == peak_fun.area

    # Movement and growth
    mobile_peak_fun = MobilePeakFunction(peak=peak_fun, time_steps=64, offset_knots=[0.7, 1.], growth_factor=1.)
    shifted_peak = mobile_peak_fun.apply_movement(32)
    assert shifted_peak.center == 0.7
    assert shifted_peak.area == peak_fun.area

    mobile_peak_fun = MobilePeakFunction(peak=peak_fun, time_steps=64, offset_knots=[0.7, 1.], growth_factor=2.)
    shifted_peak = mobile_peak_fun.apply_movement(32)
    assert shifted_peak.center == 0.7
    assert np.isclose(shifted_peak.area, np.sqrt(2) * peak_fun.area)

    mobile_peak_fun = MobilePeakFunction(peak=peak_fun, time_steps=64, offset_knots=[0.7, 1.], growth_factor=0.5)
    shifted_peak = mobile_peak_fun.apply_movement(32)
    assert shifted_peak.center == 0.7
    assert np.isclose(shifted_peak.area, peak_fun.area / np.sqrt(2))


def test_generate_time_varying():
    generator = TimeSeriesGenerator()

    offsets = generator.offsets
    times = generator.times
    generator.num_to_generate = 32
    for info, pattern in generator.generate_patterns():
        assert pattern.shape == times.shape + offsets.shape
