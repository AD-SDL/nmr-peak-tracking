"""Model architectures for NMR data"""

from torch import nn
import torch


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


class UNetPeakClassifier(nn.Module):
    """A network which maps an NMR spectrum to a classification of which pixels are at the center of a peak

    The network is `a UNet architecture <https://en.wikipedia.org/wiki/U-Net>`_ that uses zero padding to avoid the need for windowing.

    Args:
        dimensionality: 1 for single spectra and 2 for time series
        depth: How many downsampling steps to perform
        first_features: How many features in the first layer
        downscale_kernel: Width of the kernel for downscaling
        output_classes: Number of output classes
    """

    def __init__(self, dimensionality: int = 1, depth: int = 3, first_features: int = 32, downscale_kernel: int = 3, output_classes: int = 2):
        super().__init__()

        # Determine the correct convolutions for each dimensionality
        self.dimensionality = dimensionality
        print(dimensionality)
        if dimensionality == 1:
            conv_layer = nn.Conv1d
            up_conv_layer = nn.ConvTranspose1d
            self.pool_layer = nn.MaxPool1d(2, stride=2)
        elif dimensionality == 2:
            conv_layer = nn.Conv2d
            up_conv_layer = nn.ConvTranspose2d
            self.pool_layer = nn.MaxPool2d(2, stride=2)
        else:
            raise ValueError(f'Dimensionality should be one or two. Supplied: {dimensionality}')

        # Build the downscaling layers
        input_features = 1
        output_features = first_features
        self.downscale_convs = nn.ModuleList()
        for _ in range(depth):
            # Run a convolution twice
            self.downscale_convs.append(nn.Sequential(
                conv_layer(input_features, output_features, downscale_kernel, padding='same'),
                nn.ReLU(),
                conv_layer(output_features, output_features, downscale_kernel, padding='same'),
                nn.ReLU(),
            ))

            # Increase the features for the next layer
            input_features = output_features
            output_features *= 2

        # Build the upscaling layers
        output_features = input_features // 2
        self.upscale_steps = nn.ModuleList()
        self.upscale_convs = nn.ModuleList()
        for _ in range(depth - 1):
            # Make the upscale convolution, which reduces the feature size
            self.upscale_steps.append(
                up_conv_layer(input_features, input_features // 2, 2, stride=2)
            )
            self.upscale_convs.append(nn.Sequential(
                conv_layer(input_features, output_features, downscale_kernel, padding='same'),
                nn.ReLU(),
                conv_layer(output_features, output_features, downscale_kernel, padding='same'),
                nn.ReLU(),
            ))

            # Decrease the features for the next layer
            input_features = output_features
            output_features = output_features // 2

        # The output convolution
        self.output_conv = conv_layer(input_features, output_classes, downscale_kernel, padding='same')

    def forward(self, inputs):

        # In each layer, we progressively downstep
        shortcut_inputs = []
        input_stack = inputs[:, None, :]  # Insert a "channel" dimension
        for convs in self.downscale_convs[:-1]:
            output_stack = convs(input_stack)
            shortcut_inputs.append(output_stack)
            input_stack = self.pool_layer(output_stack)

        # Run the last downstep without the maxpool
        input_stack = self.downscale_convs[-1](input_stack)

        # Then start upscaling
        for step, convs, shortcut in zip(self.upscale_steps, self.upscale_convs, shortcut_inputs[-1::-1]):
            upscaled = step(input_stack)
            appended = torch.cat((upscaled, shortcut), axis=1)
            input_stack = convs(appended)

        # Now map to output classes
        output = self.output_conv(input_stack)
        return nn.Softmax(dim=1)(output)  # Normalize on dimensions
