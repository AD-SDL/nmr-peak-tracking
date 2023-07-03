"""Datasets associated with training PyTorch peak detection modules"""
from itertools import product
import logging

import numpy as np
from torch.utils.data import IterableDataset

from nmrtrack.synthetic import PatternGenerator, PeakInformation

logger = logging.getLogger(__name__)


class BaseSyntheticNMRDataset(IterableDataset):
    """
    Args:
        generator: Tool which generates synthetic patterns
        normalize: Normalize the pattern such that the maximum value is 1
    """

    def __init__(self, generator: PatternGenerator, normalize: bool = True):
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

    def generate_labels(self, info: list[PeakInformation]):
        raise NotImplementedError()


class PeakPositionDataset(BaseSyntheticNMRDataset):
    """Produce a stream of NMR patterns and use a sequence of peak positions as labels"""

    def generate_labels(self, info: list[PeakInformation]):
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

    def __init__(self, generator: PatternGenerator, label_types: bool = False, normalize: bool = True):
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

    def generate_labels(self, info: list[PeakInformation]):
        output = np.zeros((self.generator.offset_count,), dtype=np.int64)

        # Get the index corresponding to the center of each peak
        for peak in info:
            peak_ind = round(peak.center / self._offset_pitch)
            while output[peak_ind] != 0:  # Edge case: overlapping peaks
                peak_ind += 1
            output[peak_ind] = self._peak_types.index(peak.peak_type) + 1 if self.label_types else 1

        return output
