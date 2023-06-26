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
from dataclasses import dataclass
from typing import Sequence, Callable, Iterator, NamedTuple

from scipy.special import binom
import numpy as np

VectorFunction = Callable[[int | float | np.ndarray], np.ndarray]
"""Function which accepts a number or vector and returns a vector"""


class PeakInformation(NamedTuple):
    """The type (e.g., doublet of doublets), center, width, and area of a peak"""
    peak_type: str
    center: float
    width: float
    area: float


def lorentz(x: np.ndarray, center: float, width: float) -> np.ndarray:
    x_prime = (x - center) / width
    return 1.0 / (np.pi * width * (1.0 + x_prime * x_prime))


# TODO (wardlt): Should each split have different widths?
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


@dataclass()
class PatternGenerator:
    """Tool for creating an endless stream of randomly-generated NMR patterns

    Algorithm:
    1. Determine how many peaks to generate
    2. Create the peak functions. For each peak:
        1. Determine the recursion depth for splitting
        2. Determine the multiplicity of each split
        3. Determine the offset between each peak, sort such that the largest offset is for the first split
        4. Determine the position, width, and area of the peak. The center may be no closer than 0.02 ppm from the edge of the range
        5. Generate a function for the peak
    3. Combine peaks into a single function
    4. Generate intensity as a function of offset
    """

    # Control the generation of peaks
    multiplicity_depth_weights: Sequence[float] = (0.2, 0.75, 0.05)
    """Weights used when selecting the depth of splitting. The first weight is for no splitting, the second for a single split (e.g., a triplet)"""
    multiplicity_weights: Sequence[float] = (0.50, 0.45, 0.05)
    """Weights to use when selecting multiplicity for each split. The first weight is for a doublet, second for a triplet, etc."""
    multiplicity_coupling_offset_range: tuple[float, float] = (0.002, 0.02)

    # Control the area and width of the peaks
    peak_width_range: tuple[float, float] = (0.0004, 0.001)
    """Range over which to vary peak widths"""
    peak_area_range: tuple[float, float] = (0.02, 1.)
    """Range over which to vary peak areas"""

    # Control the number of peaks in the pattern
    pattern_peak_count_weights: Sequence[float] = (0.5, 0.4, 0.1)
    """Probably to include a certain number of peaks. The first weight is for a single peak, second is for two peaks, etc"""

    # Features which control rendering the output as a vector
    offset_length: float = 0.2
    """Range over which to generate NMR spectrum. Will generate peaks centered between (0.1, ``offset_length`` - 0.1)"""
    offset_count: int = 256
    """Number of equally-spaced points to generate over the range"""

    @property
    def offsets(self):
        """Position of the offsets for each point in the generated patterns"""
        return np.linspace(0, self.offset_length, self.offset_count)

    def generate_patterns(self) -> Iterator[tuple[set[PeakInformation], np.ndarray]]:
        """Generate random patterns according to the parameters of this class"""

        offsets = self.offsets
        while True:
            peak_info, peak_funcs = self.generate_peak_functions()
            yield peak_info, combine_peaks(peak_funcs)(offsets)

    def generate_peak_functions(self) -> tuple[list[PeakInformation], list[VectorFunction]]:
        """Generate a random pattern

        Returns:
            - Information about the peaks
            - A function which produces each peak
        """

        # Determine the number of peaks to create
        n_peaks = np.random.choice(len(self.pattern_peak_count_weights), p=self.pattern_peak_count_weights) + 1

        # Create them
        peak_info = []
        peak_func = []
        for _ in range(n_peaks):
            # Determine the peak types
            depth = np.random.choice(len(self.multiplicity_depth_weights), p=self.multiplicity_depth_weights) + 1
            multiplicity = np.random.choice(len(self.multiplicity_weights), size=(depth,), p=self.multiplicity_weights) + 2

            # Determine the coupling offsets, sort in descending
            coupling_offsets = np.random.uniform(*self.multiplicity_coupling_offset_range, size=(depth,))
            coupling_offsets.sort()
            coupling_offsets = coupling_offsets[::-1]

            # Determine the center, area, and width of the peak
            center = np.random.uniform(0.02, self.offset_length - 0.02)
            area = np.random.uniform(*self.peak_area_range)
            width = np.random.uniform(*self.peak_width_range)

            # Make the peak define its information
            peak_info.append(PeakInformation(''.join(map(str, multiplicity)), center, width, area))
            peak_func.append(
                generate_peak(center, area, width, multiplicity, coupling_offsets)
            )

        return peak_info, peak_func
