"""Datasets and models associated with training PyTorch peak detection modules"""
import numpy as np
import torch
from torch.utils import data
from torch import nn

from nmrtrack.synthetic import PatternGenerator


class PatternDataset(data.IterableDataset):
    """Produce a stream of NMR patterns and labels of the peak positions as a sequence

    Args:
        generator: Tool which generates synthetic patterns
    """

    def __init__(self, generator: PatternGenerator):
        super().__init__()
        self.generator = generator
        self.peak_count = len(generator.pattern_peak_count_weights)

    def __iter__(self):
        for info, pattern in self.generator.generate_patterns():
            peak_centers = np.zeros((self.peak_count,), dtype=float)
            for i, peak in enumerate(info):
                peak_centers[i] = peak.center
            yield pattern, peak_centers


class PeakLocationPredictor(nn.Module):
    """Network which predicts the positions of peaks

    Args:
        num_offsets: Number of offsets at which we measure intensity
        num_features: Number of features used to describe the pattern
        max_peaks: Maximum number of peaks to generate
    """

    def __init__(self, num_offsets: int, num_features: int = 8, max_peaks: int = 4):
        super().__init__()
        self.linear = nn.Linear(num_offsets, num_features)
        self.act = nn.ReLU()
        self.rnn = nn.RNN(num_features, num_features, num_layers=max_peaks)
        self.out_network = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1),
            nn.ReLU()
        )
        self.num_features = num_features
        self.max_peaks = max_peaks

    def forward(self, patterns: torch.Tensor):
        # Get features for the image
        feats = self.linear(patterns)
        feats = self.act(feats)

        # Use them to generate a set of peak locations
        feats = feats.unsqueeze(0).expand((self.max_peaks, -1, -1))
        output, h = self.rnn(feats)

        # Pass the outputs through a linear network
        peak_pos = self.out_network(output)
        return torch.squeeze(peak_pos, dim=-1).t()
