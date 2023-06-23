"""Utilities for generating synthetic NMR patterns

Our approach is to generate entire sets of NMR peaks
and to simulate them moving along several frames.
This is somewhat between the approach of
`DeepPicker <https://www.nature.com/articles/s41467-021-25496-5>`_,
which generated individual single peaks,
and `Sagmeister et al. <https://pubs.rsc.org/en/content/articlehtml/2022/dd/d2dd00006g>`_
who use entire patterns as the base source.
"""
from functools import partial
from typing import Sequence, Callable

from scipy.special import binom
import numpy as np

VectorFunction = Callable[[int | float | np.ndarray], np.ndarray]


def lorentz(x: np.ndarray, center: float, width: float) -> np.ndarray:
    x_prime = (x - center) / width
    return 1.0 / (np.pi * width * (1.0 + x_prime * x_prime))


def generate_peak(center: float, area: float, width: float, multiplicity: Sequence[int] = (), coupling_offsets: Sequence[float] = ()) -> VectorFunction:
    """Generate a function with produces a synthetic peak centered at :math:`\\delta=0`

    We describe peaks using a Lorentz distribution function.

    Features we should add later on:
        - Heights between peaks being imbalanced rather than perfect binomial
        - pseudo-Voigt for peak shape with asymmetry

    Args:
        center: Location of the center of mass of the peak
        area: Total area of the cluster of peaks
        width: Width of peaks
        multiplicity: A list of the iterative splits to apply. A doublet of doublets would be (2, 2).
        coupling_offsets: Offsets for between peaks at each level of splitting
    """

    # Output is determined as a sum of laplacian functions
    functions: list[tuple[float, VectorFunction]] = []

    # Recurse to make the peak functions
    if len(multiplicity) == 0:
        functions.append((area, partial(lorentz, center=center, width=width)))
    else:
        # Determine the contribution of each peak to the total weight
        my_mult = multiplicity[0]
        weights = np.array([binom(my_mult - 1, i) for i in range(my_mult)])
        weights /= weights.sum()

        # Determine the positions of each peak
        my_offset = coupling_offsets[0]
        offsets = np.arange(my_mult, dtype=float) * my_offset
        offsets -= offsets.mean()

        # Build the underlying peak functions
        if len(multiplicity) == 1:
            peak_functions = [partial(lorentz, center=center + offset, width=width) for offset in offsets]
        else:
            # TODO (wardlt): Have the recursion return a list of functions with weights rather than another wrapped function
            peak_functions = [generate_peak(center + offset, area, width, multiplicity[1:], coupling_offsets[1:]) for offset in offsets]

        # Combine them
        for w, p in zip(weights, peak_functions):
            functions.append((w * area, p))

    # The output function is a sum of peaks
    def _output(x: np.ndarray) -> np.ndarray:
        output = np.zeros_like(x, dtype=float)
        for w, f in functions:
            output += w * f(x)
        return output

    return _output


def combine_peaks(peaks: Sequence[VectorFunction]) -> VectorFunction:
    """Takes several peaks and combines them together

    Will eventually support leaning due to proximity.

    Args:
        peaks: List of peaks
    Returns:
        Function which produces all peaks
    """

    def _output(x: np.ndarray) -> np.ndarray:
        output = np.zeros_like(x, dtype=float)
        for peak in peaks:
            output += peak(x)
        return output
    return _output
