"""Train a UNet model and record outputs with MLFlow"""
from argparse import ArgumentParser
from ignite.contrib.handlers import MLflowLogger
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import global_step_from_engine
from ignite.metrics import Loss
from nmrtrack.torch.models import UNetPeakClassifier
from nmrtrack.torch.data import PeakClassifierDataset
from nmrtrack.synthetic import PatternGenerator
from torch.utils.data import DataLoader
from torch import nn
import ignite
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='Training Data', description='Setting which control how the training data are generated')
    group.add_argument('--offset-count', default=64, help='Number of offsets at which to measure NMR signal', type=int)
    group.add_argument('--training-batches', default=16, help='Number of batches to generate per epoch', type=int)
    group.add_argument('--validation-size', default=1024, help='Number of patterns to include in validate set', type=int)
    group.add_argument('--test-size', default=1024, help='Number of patterns to include in validate set', type=int)

    group = parser.add_argument_group(title='Optimization Setting', description='Parameters of the optimization algorithm')
    group.add_argument('--nonpeak-weight', default=1e-2, help='Relative weight of assigning segments as not peaks', type=float)
    group.add_argument('--num-epochs', default=32, help='Number of training epochs', type=int)
    group.add_argument('--batch-size', default=128, help='Number of training epochs', type=int)

    group = parser.add_argument_group(title='Architecture', description='Shape of the machine learning model')
    group.add_argument('--kernel-width', default=5, help='Width of the convolution kernel', type=int)
    group.add_argument('--depth', default=3, help='Number of reduction steps', type=int)
    group.add_argument('--features', default=64, help='Number of features per image', type=int)

    args = parser.parse_args()

    # Create the model and data loaders
    model = UNetPeakClassifier(depth=args.depth, first_features=args.features, downscale_kernel=args.kernel_width).to(device)
    train_loader = DataLoader(PeakClassifierDataset(PatternGenerator(offset_count=args.offset_count, num_to_generate=args.training_batches * args.batch_size)),
                              batch_size=args.batch_size, num_workers=4)
    valid_loader = DataLoader(PeakClassifierDataset(PatternGenerator(offset_count=args.offset_count, num_to_generate=args.validation_size, seed=1)),
                              batch_size=args.batch_size)
    test_loader = DataLoader(PeakClassifierDataset(PatternGenerator(offset_count=args.offset_count, num_to_generate=args.test_size, seed=2)),
                             batch_size=args.batch_size)

    # Define the loss function and optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([args.nonpeak_weight, 1])).to(device)

    # Define the trainer and evaluators
    val_metrics = {
        "loss": Loss(criterion)
    }
    trainer = create_supervised_trainer(model, opt, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    # Run validation at the end of each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(trainer):
        evaluator.run(valid_loader)


    # Prepare the logging parts
    mlflow_logger = MLflowLogger()
    mlflow_logger.log_params({
        **args.__dict__.copy(),
        'batch_size': train_loader.batch_size,
        'model': model.__class__.__name__,
        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda,
        "device name": torch.cuda.get_device_name(0)
    })
    mlflow_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names='all',
        global_step_transform=global_step_from_engine(trainer),
    )

    # Run it
    trainer.run(train_loader, max_epochs=args.num_epochs)

    # Evaluate the performance
    eval_state = evaluator.run(test_loader)
    mlflow_logger.log_metrics(eval_state.metrics)
