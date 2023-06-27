import torch
from torch.utils.data import DataLoader
from pytest import fixture

from nmrtrack.synthetic import PatternGenerator
from nmrtrack.torch import PeakPositionDataset, PeakLocationPredictor


@fixture()
def generator() -> PatternGenerator:
    return PatternGenerator(offset_count=128)


def test_dataset(generator):
    # Test that the dataset work
    ds = PeakPositionDataset(generator)
    pattern, peaks = next(iter(ds))
    assert pattern.shape == (128,)
    assert peaks.shape == (ds.peak_count,)

    # Ensure it can be used in a loader
    loader = DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 128)
    assert batch_y.shape == (2, ds.peak_count)


def test_cnn_to_seq(generator):
    # Get a sample batch of data
    ds = PeakPositionDataset(generator)
    loader = DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))

    model = PeakLocationPredictor(generator.offset_count)
    batch_y_pred = model(batch_x.to(torch.float32))
    assert batch_y_pred.shape == (2, 4)
    assert torch.all(batch_y_pred >= 0)
