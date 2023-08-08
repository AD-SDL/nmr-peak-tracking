"""Tests around generating synthetic peaks"""
from scipy.integrate import quad
import numpy as np

from nmrtrack.synthetic import generate_peak, PatternGenerator, PeakFunction, MultiplePeakFunctions


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


def test_generate():
    generator = PatternGenerator()

    # Ensure the pattern has the same shape of the offsets
    #  TODO (wardlt): Come up with more tests
    offsets = generator.offsets
    generator.num_to_generate = 32
    for info, pattern in generator.generate_patterns():
        assert offsets.shape == pattern.shape
