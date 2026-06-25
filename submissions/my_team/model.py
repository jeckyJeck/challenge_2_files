import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Small ResNet-style block for stable deeper CNN training."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_blocks)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.residual_blocks(x)
        return self.pool(x)


class ModelArchitecture(nn.Module):
    """
    Student model architecture.

    Students should define their model here.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.features = nn.Sequential(
            ConvStage(3, 48, num_blocks=1),
            ConvStage(48, 96, num_blocks=1),
            ConvStage(96, 192, num_blocks=2),
            ConvStage(192, 384, num_blocks=1),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: batch of images

        Returns:
            logits for 20 classes
        """
        x = self.features(x)
        return self.classifier(x)
