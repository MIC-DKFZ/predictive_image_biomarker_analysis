import numpy as np

import torch
from torch.autograd import grad, Function
import torch.nn.functional as F
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        activation="ReLU",
        bias=True,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.act = getattr(nn, activation, "ReLU")()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        activation="ReLU",
        bias=False,
    ):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.act = getattr(nn, activation, "ReLU")()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class MiniResNet(nn.Module):
    # ResNet 14
    def __init__(
        self,
        block=ResidualBlock,
        layers=[2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=3,
        img_size=None,
        use_avgpool=None,
    ):
        super(MiniResNet, self).__init__()
        self.in_channels = in_channels
        self.local_in_channels = 16
        self.activation_type = activation
        self.conv = nn.Conv2d(
            self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 1)
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.local_in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
            )
        )
        self.local_in_channels = out_channels
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, activation=self.activation_type)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avg_pool(x)  # 1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.finalact is not None:
            x = self.finalact(x)
        return x


class MiniResNetMTL(nn.Module):
    def __init__(
        self,
        block=ResidualBlock,
        layers=[2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=3,
        img_size=None,
        use_avgpool=None,
    ):
        super(MiniResNetMTL, self).__init__()
        self.in_channels = in_channels
        self.local_in_channels = 16
        self.activation_type = activation
        self.conv = nn.Conv2d(
            self.in_channels, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc0 = nn.Linear(64, 1)
        self.fc1 = nn.Linear(64, 1)
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.local_in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
            )
        )
        self.local_in_channels = out_channels
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, activation=self.activation_type)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x0 = self.fc0(x)
        x1 = self.fc1(x)
        if self.finalact is not None:
            x0 = self.finalact(x0)
            x1 = self.finalact(x1)
        return x0, x1


class ResNet18(nn.Module):
    # ResNet18
    # reference: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
    def __init__(
        self,
        block=ResidualBlock,
        layers=[2, 2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=3,
        img_size=None,
        use_avgpool=None,
    ):
        super(ResNet18, self).__init__()
        self.in_channels = in_channels
        self.local_in_channels = 64
        self.activation_type = activation
        self.conv = nn.Conv2d(
            self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # self.avg_pool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.local_in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    # padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
                bias=False,
            )
        )
        self.local_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    activation=self.activation_type,
                    bias=False,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.finalact is not None:
            x = self.finalact(x)
        return x


class ResNet18MTL(nn.Module):
    # ResNet18, two-headed
    # reference: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
    def __init__(
        self,
        block=ResidualBlock,
        layers=[2, 2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=3,
        img_size=None,
        use_avgpool=None,
    ):
        super(ResNet18MTL, self).__init__()
        self.in_channels = in_channels
        self.local_in_channels = 64
        self.activation_type = activation
        self.conv = nn.Conv2d(
            self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(512, 1)
        self.fc1 = nn.Linear(512, 1)
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None
        self.use_avgpool = use_avgpool

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.local_in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
                bias=False,
            )
        )
        self.local_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    activation=self.activation_type,
                    bias=False,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x0 = self.fc0(x)
        x1 = self.fc1(x)
        if self.finalact is not None:
            x0 = self.finalact(x0)
            x1 = self.finalact(x1)
        return x0, x1


# ResNet model for 3D data
class ResNet3D18(nn.Module):
    def __init__(
        self,
        block=ResidualBlock3D,
        layers=[2, 2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=1,
        use_batchnorm=True,
    ):
        super(ResNet3D18, self).__init__()
        self.activation_type = activation
        self.in_channels = in_channels
        self.local_in_channels = 64
        self.conv = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm3d(64)
        else:
            self.bn = nn.InstanceNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AveragePool3d((7,7,7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 1)
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            if self.use_batchnorm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(out_channels),
                )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
                bias=False,
            )
        )
        self.local_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    activation=self.activation_type,
                    bias=False,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)  #
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.finalact is not None:
            x = self.finalact(x)

        return x


# One headed ResNet model for 3D data with 4FC layers as final layers
class ResNet3D184FC(nn.Module):
    def __init__(
        self,
        block=ResidualBlock3D,
        layers=[2, 2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=1,
        use_batchnorm=True,
    ):
        super(ResNet3D184FC, self).__init__()
        self.activation_type = activation
        self.in_channels = in_channels
        self.local_in_channels = 64
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if self.use_batchnorm:
            self.bn = nn.BatchNorm3d(64)
        else:
            self.bn = nn.InstanceNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AveragePool3d((7,7,7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),  # First hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(128, 32),  # Second hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(32, 16),  # Third hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(16, 1),  # Output layer
        )

        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            if self.use_batchnorm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(out_channels),
                )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
                bias=False,
            )
        )
        self.local_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    activation=self.activation_type,
                    bias=False,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)  #
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.finalact is not None:
            x = self.finalact(x)

        return x


# Two head ResNet model for 3D data
class ResNet3D18MTL(nn.Module):
    def __init__(
        self,
        block=ResidualBlock3D,
        layers=[2, 2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=1,
        use_batchnorm=True,
    ):
        super(ResNet3D18MTL, self).__init__()
        self.activation_type = activation
        self.in_channels = in_channels
        self.local_in_channels = 64
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if self.use_batchnorm:
            self.bn = nn.BatchNorm3d(64)
        else:
            self.bn = nn.InstanceNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AveragePool3d((7,7,7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc0 = nn.Linear(512, 1)
        self.fc1 = nn.Linear(512, 1)
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            if self.use_batchnorm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(out_channels),
                )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
                bias=False,
            )
        )
        self.local_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    activation=self.activation_type,
                    bias=False,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)  #
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x0 = self.fc0(x)
        x1 = self.fc1(x)

        if self.finalact is not None:
            x0 = self.finalact(x0)
            x1 = self.finalact(x1)

        return x0, x1


# Two head ResNet model for 3D data
class ResNet3D18MTL4FC(nn.Module):
    def __init__(
        self,
        block=ResidualBlock3D,
        layers=[2, 2, 2, 2],
        activation="ReLU",
        final_activation=None,
        in_channels=1,
        use_batchnorm=True,
    ):
        super(ResNet3D18MTL4FC, self).__init__()
        self.activation_type = activation
        self.in_channels = in_channels
        self.local_in_channels = 64
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if self.use_batchnorm:
            self.bn = nn.BatchNorm3d(64)
        else:
            self.bn = nn.InstanceNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AveragePool3d((7,7,7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc0 = nn.Sequential(
            nn.Linear(512, 128),  # First hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(128, 32),  # Second hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(32, 16),  # Third hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(16, 1),  # Output layer
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),  # First hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(128, 32),  # Second hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(32, 16),  # Third hidden layer
            getattr(nn, activation, "ReLU")(),
            nn.Linear(16, 1),  # Output layer
        )
        self.act = getattr(nn, activation, "ReLU")()
        if final_activation is not None:
            self.finalact = getattr(nn, final_activation, "Linear")()
        else:
            self.finalact = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.local_in_channels != out_channels):
            if self.use_batchnorm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.local_in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        # padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm3d(out_channels),
                )
        layers = []
        layers.append(
            block(
                self.local_in_channels,
                out_channels,
                stride,
                downsample,
                activation=self.activation_type,
                bias=False,
            )
        )
        self.local_in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    activation=self.activation_type,
                    bias=False,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)  #
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x0 = self.fc0(x)
        x1 = self.fc1(x)

        if self.finalact is not None:
            x0 = self.finalact(x0)
            x1 = self.finalact(x1)

        return x0, x1


models_dict = {
    "miniresnet": MiniResNet,
    "miniresnetmtl": MiniResNetMTL,
    "resnet18": ResNet18,
    "resnet18mtl": ResNet18MTL,
    "resnet3d18": ResNet3D18,
    "resnet3d184fc": ResNet3D184FC,
    "resnet3d18mtl": ResNet3D18MTL,
    "resnet3d18mtl4fc": ResNet3D18MTL4FC,
}

if __name__ == "__main__":
    model = ResNet18()
    x = torch.zeros((1, 3, 224, 224))
    out = model(x)
    print(model.layer1)
