import torch
from torch.utils.data import DataLoader
from pytest import fixture, mark

from nmrtrack.synthetic import PatternGenerator, TimeSeriesGenerator
from nmrtrack.torch.data import PeakPositionDataset, PeakClassifierDataset
from nmrtrack.torch.models import PeakLocationPredictor, UNetPeakClassifier


@fixture()
def generator() -> PatternGenerator:
    return PatternGenerator(seed=1, offset_count=128)


@fixture()
def timeseries_generator() -> TimeSeriesGenerator:
    return TimeSeriesGenerator(seed=1, offset_count=128, time_count=64)


def test_dataset(generator):
    # Test that the dataset works
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

    # Make sure we repeat the batch through the same order
    repeat_batch_x, repeat_batch_y = next(iter(loader))
    assert torch.isclose(repeat_batch_x, batch_x).all()
    assert torch.isclose(repeat_batch_y, batch_y).all()


def test_classifier_dataset(generator):
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

    # Make sure it works with the Data Loader
    loader = DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == batch_y.shape


def test_cnn_to_seq(generator):
    # Get a sample batch of data
    ds = PeakPositionDataset(generator)
    loader = DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))

    model = PeakLocationPredictor(generator.offset_count)
    batch_y_pred = model(batch_x.to(torch.float32))
    assert batch_y_pred.shape == (2, 4)
    assert torch.all(batch_y_pred >= 0)


@mark.parametrize('label_types', [True, False])
@mark.parametrize('generator_name', ['generator', 'timeseries_generator'])
def test_classifier(generator_name, label_types, request):
    # Create the model and loader
    generator = request.getfixturevalue(generator_name)
    ds = PeakClassifierDataset(generator, label_types=label_types)
    loader = DataLoader(ds, batch_size=2)
    n_classes = len(ds.peak_types) + 1 if label_types else 2
    model = UNetPeakClassifier(output_classes=n_classes, dimensionality=2 if 'time' in generator_name else 1)

    # Ensure it gives the correct batch sizes
    batch_x, batch_y = next(iter(loader))
    pred_y = model(batch_x)
    assert pred_y.shape == (batch_x.shape[0], n_classes, *batch_x.shape[1:])

    # Make sure we can compute losses
    loss = torch.nn.CrossEntropyLoss()
    output = loss(pred_y, batch_y)
    output.backward()  # Ensure we get gradients


def test_timeseries_classifier_dataset(timeseries_generator):
    ds = PeakClassifierDataset(timeseries_generator, label_types=True)

    image, labels = next(iter(ds))

    # Make sure the image is reasonable
    assert image.shape == (64, 128)
    assert image.max() == 1.

    # Check that the labels are present
    assert labels.shape == image.shape
    assert (labels >= 1).any(axis=1).all()  # At least one peak per frame
