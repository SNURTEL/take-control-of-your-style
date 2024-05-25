import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torchvision.models import ResNet18_Weights


class ResidualBlock(nn.Module):
    """A naive residual block. Creates a skip connection between itself and the previous block on the graph."""

    def __init__(self, n_channels: int) -> None:
        """Init the module.

        Args:
            n_channels: Number of channels to pass through.
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(n_channels),
            nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(n_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)  # type: ignore[no-any-return]


class Generator(nn.Module):
    """
    U-net like generator model. Downsamples the input, passes the latent vector through
    subsequent residual blocks and scales it back up.
    """

    def __init__(self) -> None:
        """Init the generator"""
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),  # Normalization layer, equivalent to batch norm but for style transfer
            nn.ReLU(True),
            # Downsampling
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            # Residual blocks - pretty deep network - avoid vanishing gradients
            *[ResidualBlock(256) for _ in range(9)],
            # Upsampling
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)  # type: ignore[no-any-return]


class Discriminator(nn.Module):
    """Discriminator model based on a CNN with a linear head"""

    def __init__(self) -> None:
        """Init the module."""
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)  # type: ignore[no-any-return]


class SemanticRegularization(nn.Module):
    def __init__(self, feature_extractor: nn.Module | None = None, beta_param: float = 1e-2) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor or models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.beta_param = beta_param

    def forward(self, source_batch: Tensor, target_batch: Tensor) -> float:
        src_features, tgt_features = self.feature_extractor(source_batch), self.feature_extractor(target_batch)
        avg_similarity = cosine_similarity(src_features, tgt_features, dim=1).view(-1).mean().item()
        avg_similarity_in_range_0_1 = (avg_similarity + 1) / 2
        return self.beta_param * (1 - avg_similarity_in_range_0_1)
