from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_cnn_output_dim(
    input_size: int, conv_kernel_size: int, padding_size: int, conv_stride_size: int
) -> int:
    return (input_size + 2 * padding_size - conv_kernel_size) / conv_stride_size + 1


def get_pool_output_dim(
    input_size, pool_kernel_size: int, pool_stride_size: int
) -> int:
    return np.floor((input_size - pool_kernel_size) / pool_stride_size + 1)


def get_cnn_layer_output_dim(
    n_layers: int,
    input_size: int,
    conv_kernel_size: int,
    padding_size: int = 0,
    conv_stride_size: int = 1,
    pool_kernel_size: int = 2,
    pool_stride_size: int = 2,
) -> int:
    if n_layers > 1:
        cnn_output = get_cnn_output_dim(
            input_size, conv_kernel_size, padding_size, conv_stride_size
        )
        pool_output = get_pool_output_dim(
            cnn_output, pool_kernel_size, pool_stride_size
        )
        n_layers -= 1
        return int(
            get_cnn_layer_output_dim(
                n_layers,
                pool_output,
                conv_kernel_size,
                padding_size,
                conv_stride_size,
                pool_kernel_size,
                pool_stride_size,
            )
        )
    else:
        cnn_output = get_cnn_output_dim(
            input_size, conv_kernel_size, padding_size, conv_stride_size
        )
        pool_output = get_pool_output_dim(
            cnn_output, pool_kernel_size, pool_stride_size
        )
        return int(pool_output)


class MultivariateMLP(nn.Module):
    def __init__(
        self,
        input_dimension: Tuple,
        in_channels: int,
        n_outputs: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            FlattenMLP(),
            nn.Linear(input_dimension[0] * input_dimension[1] * 1 * in_channels, 564),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.linear_classifiers = [
            nn.Linear(564, n_outputs) for _ in range(in_channels)
        ]

    def forward(self, x):
        out = self.encoder(x)
        outputs = [
            F.softmax(classifier(out), dim=-1) for classifier in self.linear_classifiers
        ]
        return outputs


class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class FlattenMLP(nn.Module):
    """A custom layer that views an input as 1D."""

    def forward(self, input):
        return input.reshape(-1)


class MultivariateCNN(nn.Module):
    """https://github.com/ArminBaz/UTK-Face/blob/master/src/MultNN.py"""

    def __init__(
        self,
        input_dimension: Tuple,
        in_channels: int,
        n_outputs: int,
        n_cnn_layers: int = 2,
        conv_kernel_size: int = 2,
        pool_kernel_size: int = 2,
        conv_channels_1: int = 256,
        conv_channels_2: int = 512,
        linear_hidden_cells: int = 256,
        linear_dropout: float = 0.5,
    ):
        super(MultivariateCNN, self).__init__()
        self.in_channels = in_channels
        linear_dim1 = get_cnn_layer_output_dim(
            n_layers=n_cnn_layers,
            input_size=input_dimension[0],
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
        )
        linear_dim2 = get_cnn_layer_output_dim(
            n_layers=n_cnn_layers,
            input_size=input_dimension[1],
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels_1,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.Conv2d(
                in_channels=conv_channels_1,
                out_channels=conv_channels_2,
                kernel_size=conv_kernel_size,
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            #               nn.Conv2d(in_channels, out_channels=8, kernel_size=conv_kernel_size),
            #               nn.ReLU(),
            #               nn.MaxPool2d(kernel_size=pool_kernel_size),
            Flatten(),
            nn.Linear(linear_dim1 * linear_dim2 * conv_channels_2, linear_hidden_cells),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
        )
        self.linear_classifiers = [
            nn.Linear(linear_hidden_cells, n_outputs) for _ in range(in_channels)
        ]

    def forward(self, x):
        out = self.encoder(x)
        outputs = [
            F.softmax(classifier(out), dim=-1) for classifier in self.linear_classifiers
        ]
        return outputs
