import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxDepthPool2d(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        shape = x.shape
        channels = shape[1] // self.pool_size
        new_shape = (shape[0], channels, self.pool_size, *shape[-2:])
        return torch.amax(x.view(new_shape), dim=2)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=8):
        super().__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.feed_forward = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_channels, squeeze_channels),
            nn.Mish(),
            nn.Linear(squeeze_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        calibration = self.feed_forward(x)
        return x * calibration.view(-1, x.shape[1], 1, 1)


class ResidualConnection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        squeeze_active=False,
        squeeze_factor=8,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.squeeze_active = squeeze_active
        self.squeeze_excitation = SqueezeExcitation(out_channels, squeeze_factor)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut_connection = nn.Sequential()
        if not in_channels == out_channels or stride > 1:
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x_residual = self.feed_forward(x)
        x_shortcut = self.shortcut_connection(x)
        residual_output = F.mish(x_residual + x_shortcut)
        if self.squeeze_active:
            return self.squeeze_excitation(residual_output) + x_shortcut
        return residual_output


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        channels1x1,
        channels3x31,
        channels3x32,
        channels5x51,
        channels5x52,
        channels3x3pool,
        squeeze_active=False,
        squeeze_factor=8,
    ):
        super().__init__()
        self.squeeze_active = squeeze_active
        self.squeeze_excitation = SqueezeExcitation(
            channels1x1 + channels3x32 + channels5x52 + channels3x3pool, squeeze_factor
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, channels1x1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=channels1x1),
            nn.Mish(),
        )
        self.branch3x3_dbl = nn.Sequential(
            nn.Conv2d(in_channels, channels3x31, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=channels3x31),
            nn.Mish(),
            nn.Conv2d(channels3x31, channels3x32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels3x32),
            nn.Mish(),
        )
        self.branch5x5_dbl = nn.Sequential(
            nn.Conv2d(in_channels, channels5x51, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=channels5x51),
            nn.Mish(),
            nn.Conv2d(channels5x51, channels5x52, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=channels5x52),
            nn.Mish(),
        )
        self.branch3x3_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, channels3x3pool, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=channels3x3pool),
            nn.Mish(),
        )

    def forward(self, x):
        x_concat = torch.cat(
            (
                self.branch1x1(x),
                self.branch3x3_dbl(x),
                self.branch5x5_dbl(x),
                self.branch3x3_pool(x),
            ),
            dim=1,
        )
        if self.squeeze_active:
            return self.squeeze_excitation(x_concat)
        return x_concat
