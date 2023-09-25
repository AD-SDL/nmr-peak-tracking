"""Datasets associated with training PyTorch peak detection modules"""
from itertools import product
import logging

import numpy as np
from torch.utils.data import IterableDataset

from nmrtrack.synthetic import PatternGenerator, PeakFunction, MobilePeakFunction, TimeSeriesGenerator

logger = logging.getLogger(__name__)


class BaseSyntheticNMRDataset(IterableDataset):
    """
    Args:
        generator: Tool which generates synthetic patterns
        normalize: Normalize the pattern such that the maximum value is 1
    """

    def __init__(self, generator: PatternGenerator | TimeSeriesGenerator, normalize: bool = True):
        super().__init__()
        self.generator = generator
        self.peak_count = len(generator.pattern_peak_count_weights)
        self.normalize = normalize

    def __iter__(self):
        logger.info('Beginning training data generation')
        for info, pattern in self.generator.generate_patterns():
            peak_centers = self.generate_labels(info)
            if self.normalize:
                pattern -= pattern.min()
                pattern /= pattern.max()
            yield np.array(pattern, dtype=np.float32), peak_centers

    def generate_labels(self, info: list[PeakFunction]):
        raise NotImplementedError()


class PeakPositionDataset(BaseSyntheticNMRDataset):
    """Produce a stream of NMR patterns and use a sequence of peak positions as labels"""

    def generate_labels(self, info: list[PeakFunction]):
        peak_centers = np.zeros((self.peak_count,), dtype=float)
        for i, peak in enumerate(info):
            peak_centers[i] = peak.center
        return peak_centers


class PeakClassifierDataset(BaseSyntheticNMRDataset):
    """Labels are whether each point in the offset corresponds to a peak center

    Args:
        generator: Tool which generates synthetic patterns
        label_types: Whether to assign a label based on the peak type
        normalize: Normalize the pattern such that the maximum value is 1
    """

    def __init__(self, generator: PatternGenerator | TimeSeriesGenerator, label_types: bool = False, normalize: bool = True):
        super().__init__(generator, normalize)
        self.label_types = label_types

        # Generate a list of all possible peak types
        peak_types = ['1']  # Will always label a singlet
        multiplet_types = [str(2 + c) for c in range(len(self.generator.multiplicity_weights))]
        peak_types.extend(multiplet_types)
        for depth in range(len(self.generator.multiplicity_depth_weights) - 2):
            depth += 2
            peak_types.extend(map(''.join, product(multiplet_types, repeat=depth)))
        self._peak_types = tuple(peak_types)

        # Store the space between offsets
        self._offset_pitch = self.generator.offset_length / (self.generator.offset_count - 1)

    @property
    def peak_types(self):
        """Types of peaks included in the dataset"""
        return self._peak_types

    def _update_peak_output(self, peak: PeakFunction, output: np.ndarray):
        """Update the output classification map with the information about a new peak

        Args:
            peak: New peak
            output: Output to be updated with the location of the peak
        """
        peak_ind = round(peak.center / self._offset_pitch)
        while output[peak_ind] != 0:  # Edge case: overlapping peaks
            peak_ind += 1
        output[peak_ind] = self._peak_types.index(peak.peak_type) + 1 if self.label_types else 1

    def generate_labels(self, info: list[PeakFunction]):
        # Different behavior depending on whether the peak is mobile or not
        # TODO (wardlt): I dislike having a single function with two mutually exclusive code paths. Need to think on refactoring this...
        is_moving_peak = isinstance(info[0], MobilePeakFunction)
        if is_moving_peak:
            output = np.zeros((self.generator.time_count, self.generator.offset_count), dtype=np.int64)
            # Provide the labels at each timestep
            for i, time in enumerate(self.generator.times):
                for peak in info:
                    time_peak = peak.apply_movement(time)
                    self._update_peak_output(time_peak, output[i, :])
        else:
            output = np.zeros((self.generator.offset_count,), dtype=np.int64)
            for peak in info:
                self._update_peak_output(peak, output)
        return output
