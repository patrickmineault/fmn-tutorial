"""
Builds a UNet model for spike trains. 

It uses the same architecture as the original UNet paper with residual connections, but in 1D.

The input is a spike train of shape (batch_size, N_channels, Nt) and the output is a spike train of shape (batch_size, N_channels, Nt).
"""

from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class UpsampleMethod(Enum):
    DECONV = "deconv"
    LINEAR = "linear"


class UNet1D(nn.Module):
    def __init__(
        self, nlayers, dim, latent_dim, upsample: UpsampleMethod = UpsampleMethod.LINEAR
    ):
        super(UNet1D, self).__init__()

        self.nlayers = nlayers
        self.dim = dim
        self.latent_dim = latent_dim
        self.upsample = upsample

        self.build()

    def build(self):
        self.embedding = nn.Conv1d(self.dim, self.latent_dim, kernel_size=1)
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.unembedding = nn.Conv1d(self.latent_dim, self.dim, kernel_size=1)

        for i in range(self.nlayers):
            self.downsample_layers.append(
                DownsampleLayer(
                    self.latent_dim * 2**i, self.latent_dim * (2 ** (i + 1))
                )
            )
            self.upsample_layers.append(
                UpsampleLayer(
                    self.latent_dim * (2 ** (self.nlayers - i)),
                    self.latent_dim * 2 ** (self.nlayers - i - 1),
                    upsample=self.upsample,
                )
            )

    def forward(self, X):
        X = X.permute(0, 2, 1)
        # Pad bidirectionally to the nearest (relevant) power of 2
        if self.nlayers > 0:
            X_shape = X.shape
            ideal_size = (
                int(np.ceil((X.shape[2] - 1) / (2**self.nlayers)) * (2**self.nlayers))
                + 1
            )
            left_pad = (ideal_size - X.shape[2]) // 2
            right_pad = ideal_size - X.shape[2] - left_pad
            X = F.pad(X, (left_pad, right_pad))

            assert X_shape[0] == X.shape[0]
            assert X_shape[1] == X.shape[1]
            assert X.shape[2] % (2**self.nlayers) == 1
        else:
            left_pad = 0
            right_pad = 0

        X = self.embedding(X)
        activations = []
        for layer in self.downsample_layers:
            X = layer(X)
            activations.append(X)

        if self.nlayers > 0:
            X = 0
            
        for layer in self.upsample_layers:
            X = layer(X + activations.pop())

        X = self.unembedding(X)
        # Unpad!
        if right_pad > 0:
            X = X[:, :, left_pad:-right_pad]
        X = X.permute(0, 2, 1)
        return X

    def set_baseline_rate(self, baseline_rate):
        """Sets the baseline mean spike rate of the model by modifying the bias
        of the unembedding layer."""
        self.baseline_rate = baseline_rate
        self.unembedding.bias.data = self.unembedding.bias.data * 0.0 + np.log(
            baseline_rate
        )


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super(DownsampleLayer, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=min(out_channels, in_channels),
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            stride=2,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X = self.relu(X)

        return X


class Doubler(nn.Module):
    def __init__(self):
        super(Doubler, self).__init__()

    def forward(self, X):
        assert X.ndim == 3
        X1 = torch.zeros(
            X.shape[0], X.shape[1], X.shape[2] * 2 - 1, device=X.device, dtype=X.dtype
        )
        X1[:, :, ::2] = X
        X1[:, :, 1::2] = X[:, :, :-1] + X[:, :, 1:]
        return X1


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample: UpsampleMethod):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super(UpsampleLayer, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=min(in_channels, out_channels),
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
        if upsample == UpsampleMethod.DECONV:
            self.conv3 = nn.ConvTranspose1d(
                out_channels, out_channels, kernel_size=3, padding=1, stride=2
            )
        elif upsample == UpsampleMethod.LINEAR:
            self.conv3 = nn.Sequential(
                Doubler(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            raise NotImplementedError(f"Invalid upsample method {upsample}")

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X = self.relu(X)

        return X
