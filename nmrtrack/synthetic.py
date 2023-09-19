"""Utilities for generating synthetic NMR patterns

Our approach is to generate entire sets of NMR peaks
and to simulate them moving along several frames.
This is somewhat between the approach of
`DeepPicker <https://www.nature.com/articles/s41467-021-25496-5>`_,
which generated individual single peaks,
and `Sagmeister et al. <https://pubs.rsc.org/en/content/articlehtml/2022/dd/d2dd00006g>`_
who use entire patterns as the base source.
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Callable, Iterator

from scipy.interpolate import CubicSpline
from scipy.special import binom
import numpy as np


@dataclass(frozen=True)
class PeakFunction(Callable):
    """Generates peaks and stores the centers and maxima of the constituent peaks"""

    peak_type: str | None
    """Type of the peak"""
    center: float
    """Location of the center of the collection of peaks peaks"""
    width: float
    """Width of all peaks"""
    area: float
    """Total area of the peaks"""
    subpeak_centers: tuple[float]
    """Center of each subpeak"""
    subpeak_areas: tuple[float]
    """Area of the subpeaks"""

    def __call__(self, offsets: np.ndarray) -> np.ndarray:
        output = np.zeros_like(offsets, dtype=float)
        for area, center in zip(self.subpeak_areas, self.subpeak_centers):
            output += area * lorentz(offsets, center, self.width)
        return output

    def alter_pattern(self, offset: float, scale: float) -> 'PeakFunction':
        """Shift the pattern and adjust its intensity

        Args:
            offset: How much to translate the peaks
            scale: Multiplicative factor for adjusting the peak height
        Returns:
            New peak
        """
        return PeakFunction(
            peak_type=self.peak_type,
            center=self.center + offset,
            width=self.width,
            area=self.area * scale,
            subpeak_centers=tuple(x + offset for x in self.subpeak_centers),
            subpeak_areas=tuple(a * scale for a in self.subpeak_areas)
        )


@dataclass(frozen=True)
class MultiplePeakFunctions(Callable):
    peaks: Sequence[PeakFunction] = ...

    def __call__(self, offsets: np.ndarray) -> np.ndarray:
        output = np.zeros_like(offsets, dtype=float)
        for peak in self.peaks:
            for area, center in zip(peak.subpeak_areas, peak.subpeak_centers):
                output += area * lorentz(offsets, center, peak.width)
        return output


def lorentz(x: np.ndarray, center: float, width: float) -> np.ndarray:
    x_prime = (x - center) / width
    return 1.0 / (np.pi * width * (1.0 + x_prime * x_prime))


def generate_peak(center: float, area: float, width: float, multiplicity: Sequence[int] = (), coupling_offsets: Sequence[float] = ()) -> PeakFunction:
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

    # Compute the locations of the peaks
    subpeak_area, subpeak_center, subpeak_width = zip(*_generate_peaks(center, area, width, multiplicity, coupling_offsets))
    assert np.isclose(subpeak_width[0], subpeak_width).all(), 'All peak widths should be the same'

    # Make the peak function class
    name = '1' if len(multiplicity) == 0 else ''.join(map(str, multiplicity))
    return PeakFunction(
        peak_type=name,
        center=center,
        area=area,
        width=width,
        subpeak_centers=subpeak_center,
        subpeak_areas=subpeak_area
    )


def _generate_peaks(center: float, area: float, width: float, multiplicity: Sequence[int] = (), coupling_offsets: Sequence[float] = ()) \
        -> list[tuple[float, float, float]]:
    """Private interface to :meth:`generate_peaks`, generates the peak locations but does not package them"""

    # Recurse to make the peak functions
    if len(multiplicity) == 0:
        return [(area, center, width)]
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
            return [(weight * area, center + offset, width) for weight, offset in zip(weights, offsets)]
        else:
            return sum([_generate_peaks(center + offset, weight * area, width, multiplicity[1:], coupling_offsets[1:])
                        for weight, offset in zip(weights, offsets)], [])


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

    # General options
    seed: int | None = None
    """Random number seed"""
    num_to_generate: int | None = None
    """If defined, will generate a fixed number of patterns"""

    # Control the generation of peaks
    multiplicity_depth_weights: Sequence[float] = (0.2, 0.7, 0.1)
    """Weights used when selecting the depth of splitting. The first weight is for no splitting, the second for a single split (e.g., a triplet)"""
    multiplicity_weights: Sequence[float] = (0.50, 0.45, 0.05)
    """Weights to use when selecting multiplicity for each split. The first weight is for a doublet, second for a triplet, etc."""
    multiplicity_coupling_offset_range: tuple[float, float] = (0.002, 0.02)
    """Spacing between peaks within a multiplet"""

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

    def generate_patterns(self) -> Iterator[tuple[list[PeakFunction], np.ndarray]]:
        """Generate random patterns according to the parameters of this class"""

        # Initialize
        offsets = self.offsets
        rng = np.random.RandomState(self.seed)
        num_to_generate = self.num_to_generate

        while True:
            peak_funcs = self.generate_peak_functions(rng)
            yield peak_funcs, MultiplePeakFunctions(peak_funcs)(offsets)
            if num_to_generate is not None and (num_to_generate := num_to_generate - 1) == 0:
                break

    def generate_peak_functions(self, rng: np.random.RandomState | None = None) -> list[PeakFunction]:
        """Generate a random pattern

        Args:
            rng: Random number generator. Will create a new one if none provided

        Returns:
            A list of functions which produce each peak
        """

        # Make an RNG if needed
        if rng is None:
            rng = np.random.RandomState()

        # Determine the number of peaks to create
        n_peaks = rng.choice(len(self.pattern_peak_count_weights), p=self.pattern_peak_count_weights) + 1

        # Create them
        peak_funcs = []
        for _ in range(n_peaks):
            # Determine the peak types
            depth = rng.choice(len(self.multiplicity_depth_weights), p=self.multiplicity_depth_weights)
            multiplicity = rng.choice(len(self.multiplicity_weights), size=(depth,), p=self.multiplicity_weights) + 2

            # Determine the coupling offsets, sort in descending
            coupling_offsets = rng.uniform(*self.multiplicity_coupling_offset_range, size=(depth,))
            coupling_offsets.sort()
            coupling_offsets = coupling_offsets[::-1]

            # Determine the center, area, and width of the peak
            center = rng.uniform(0.02, self.offset_length - 0.02)
            area = rng.uniform(*self.peak_area_range)
            width = rng.uniform(*self.peak_width_range)

            # Make the peak and define its information
            peak_func = generate_peak(center, area, width, multiplicity, coupling_offsets)
            peak_funcs.append(peak_func)

        return peak_funcs


@dataclass(frozen=True)
class MobilePeakFunction:
    """A peak which moves and changes in intensity as a function of time

    Build a moving peak by supplying the peak at t=start, the number of time steps over which it was measured,
    and the offsets and growth factors at longer times.
    The class assumes that the offsets and growth factors are measured at equally-space times,
    and that the rate of change at t=start is zero.

    Peak growth is exponential.
    """

    peak: PeakFunction
    """Peak function at t=start"""
    time_steps: int
    """Number of times this sequence was measured"""

    offset_knots: Sequence[float]
    """The offsets at equally-spaced points in time. The first knot is assumed to the offset at t=start and the last at t=end"""
    growth_factor: float | None = None
    """Factor by which the peak grows between t=start and t=end"""

    @cached_property
    def _offset_spline(self) -> CubicSpline:
        """Tool used to compute the offset over time"""
        times = np.linspace(0, self.time_steps, len(self.offset_knots) + 1, dtype=float)
        offsets = [self.peak.center] + list(self.offset_knots)
        return CubicSpline(times, offsets, bc_type=[(1, 0.0), (2, 0.0)])

    def apply_movement(self, time: float) -> PeakFunction:
        """Generate a pattern at a certain time interval

        Applies a movement according to the knots in :attr:`offset_knots`"""

        # If no changes, just return the underlying peak
        if len(self.offset_knots) == 0 and self.growth_factor is None:
            return self.peak

        # If not, determine how much to change the peak
        shift = self._offset_spline(time) - self.peak.center if len(self.offset_knots) > 0 else 0
        scale = (1. if self.growth_factor is None else self.growth_factor) ** (time / self.time_steps)
        return self.peak.alter_pattern(shift, scale)


@dataclass()
class TimeSeriesGenerator(PatternGenerator):
    """Generate NMR spectra which move over time.

    The generated patterns are observed NMR intensity as a function of offset
    for frames supposed to be taken at exponentially-increasing time intervals.

    Peaks move along a path governed by a spline.
    The positions of the spline knots are defined such that the peak's center will not
    move more than a certain amount.
    The number of knots is also chosen randomly.

    Peak growth follows an exponential growth model and the total fraction
    of growth is drawn from a random distribution.
    """

    # Related to how the peaks move
    movement_probability: float = 0.7
    """Probability that a peak will move rather than stay stationary"""
    movement_knot_count_weight: Sequence[float] = (0.4, 0.4, 0.2)
    """Weight for different numbers of knots in the movement"""
    movement_maximum_offset: float = 0.1
    """Maximum amount of a peak is allowed to move"""

    # Related to how the peaks grow
    growth_probability: float = 0.7
    """Probability that the peak intensity will change over time"""
    growth_rate_distribution: tuple[float, float] = (-4., 4.)
    """Range of a uniform distribution describing growth.
    The growth parameter is :math:`log(I(t=start)) - log(I(t=end))`"""

    # Related to the output shape
    time_count: int = 64
    """Number of points to generate in the time domain"""

    @property
    def times(self) -> np.ndarray:
        """Times at which we generate patterns"""
        return np.arange(self.time_count)

    def generate_patterns(self) -> Iterator[tuple[list[MobilePeakFunction], np.ndarray]]:
        # Start by creating an RNG and offsets, just like the subclass
        offsets = self.offsets
        times = self.times
        rng = np.random.RandomState(self.seed)
        num_to_generate = self.num_to_generate

        while True:
            # Start by generate the mobile peaks
            peak_funcs = self.generate_mobile_peaks(rng)

            # Fill in the peaks at each time
            output = np.zeros((self.time_count, self.offset_count), dtype=float)
            for i, time in enumerate(times):
                adjusted_peaks = [peak.apply_movement(time) for peak in peak_funcs]
                combined_peaks = MultiplePeakFunctions(peaks=adjusted_peaks)
                output[i, :] = combined_peaks(offsets)

            yield peak_funcs, output

            # Break if desired
            if num_to_generate is not None and (num_to_generate := num_to_generate - 1) == 0:
                break

    def generate_mobile_peaks(self, rng: np.random.RandomState | None = None) -> list[MobilePeakFunction]:
        """Generate a series of time-varying peaks

        Args:
            rng: Random number generator. Will create a new one if none provided
        Returns:
            Time-varying versions of these peak functions
        """

        # Make an RNG if needed
        if rng is None:
            rng = np.random.RandomState()

        # Get a starting set of peaks
        peaks = self.generate_peak_functions(rng)

        # Determine the movement for each peak
        #  TODO (wardlt): Give some peaks the same movement/growth
        output = []
        for peak in peaks:
            # Decide if we will move
            offsets = ()
            if rng.random() < self.movement_probability:
                # Pick the number of offset points
                n_points = rng.choice(len(self.movement_knot_count_weight), p=self.movement_knot_count_weight) + 1

                # Start by defining the offset at the end of the period
                # Peaks are allowed to the maximum offset to within 0.1 of the edge of the pattern
                offset_range = [
                    max(0.1, peak.center - self.movement_maximum_offset),
                    min(peak.center + self.movement_maximum_offset, self.offset_length - 0.1)
                ]
                offsets = rng.uniform(*offset_range, size=(n_points,))

            # Decide if we will grow
            growth_factor = None
            if rng.random() < self.growth_probability:
                growth_factor = np.exp(rng.uniform(*self.growth_rate_distribution))

            # Make the peak
            output.append(MobilePeakFunction(offset_knots=offsets, peak=peak, time_steps=self.time_count, growth_factor=growth_factor))
        return output
