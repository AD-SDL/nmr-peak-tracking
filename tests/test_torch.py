import torch
from torch.utils.data import DataLoader
from pytest import fixture

from nmrtrack.synthetic import PatternGenerator
from nmrtrack.torch import PeakPositionDataset, PeakLocationPredictor, PeakClassifierDataset


@fixture()
def generator() -> PatternGenerator:
    return PatternGenerator(offset_count=128)


def test_dataset(generator):
    # Test that the dataset work
    ds = PeakPositionDataset(generator)
    pattern, peaks = next(iter(ds))
    assert pattern.shape == (128,)
    assert pattern.max() == 1.
    assert pattern.min() == 0.
    assert peaks.shape == (ds.peak_count,)

    # Ensure it can be used in a loader
    loader = DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 128)
    assert batch_y.shape == (2, ds.peak_count)


def test_classifier(generator):
    ds = PeakClassifierDataset(generator, label_types=True)
    assert len(set(ds.peak_types)) == 4 + 9

    # Test that the proper shapes are returned
    pattern, label = next(iter(ds))
    assert pattern.shape == label.shape

    # Make sure the right types, proper count are returned
    for (info, _), _ in zip(generator.generate_patterns(), range(16)):
        labels = ds.generate_labels(info)
        assert set(labels) == {0}.union(ds.peak_types.index(x.peak_type) + 1 for x in info)
        assert sum(labels != 0) == len(info)

    # Ensure overlapping peaks get their own labels
    info, _ = next(generator.generate_patterns())
    info.append(info[0])
    labels = ds.generate_labels(info)
    assert sum(labels != 0) == len(info)


def test_cnn_to_seq(generator):
    # Get a sample batch of data
    ds = PeakPositionDataset(generator)
    loader = DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))

    model = PeakLocationPredictor(generator.offset_count)
    batch_y_pred = model(batch_x.to(torch.float32))
    assert batch_y_pred.shape == (2, 4)
    assert torch.all(batch_y_pred >= 0)
