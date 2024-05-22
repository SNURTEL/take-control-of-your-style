from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class FeatureMaps(NamedTuple):
    content: list[torch.Tensor]
    style: list[torch.Tensor]


class FeatureMapExtractor(nn.Module, ABC):
    """Abstract feature map extractor for Gatys' neural style transfer alghoritm"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> FeatureMaps:
        """Compute the feature maps of the given image tensor

        Args:
            x: Input tensor

        Returns:
            Content and style feature maps
        """
        ...


class BaseVGGFeatureMapExtractor(FeatureMapExtractor, ABC):
    """Base class for feature map extractors from the VGG family"""

    _content_layer_indices: list[int]
    _style_layer_indices: list[int]

    def __init__(
        self,
        vgg: nn.Module,
        content_layer_indices: list[int],
        style_layer_indices: list[int],
    ) -> None:
        """Init the extractor"""
        super().__init__()

        self._content_layer_indices = content_layer_indices
        self._style_layer_indices = style_layer_indices

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.vgg = vgg
        self.vgg = self.vgg.eval()
        self.vgg.requires_grad_(False)

        # misc improvements
        for i, layer in enumerate(self.vgg.children()):
            if isinstance(layer, nn.MaxPool2d):
                self.vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2)
            if isinstance(layer, nn.ReLU):
                self.vgg[i] = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        """Compute the feature maps of the given image tensor

        Args:
            x: Input tensor

        Returns:
            Content and style feature maps
        """
        content_feature_maps = []
        style_feature_maps = []

        x = self.normalize(x).unsqueeze(0)

        for idx, layer in enumerate(self.vgg.children()):
            x = layer(x)
            if idx in self._content_layer_indices:
                content_feature_maps.append(x)
            if idx in self._style_layer_indices:
                style_feature_maps.append(x)

        return FeatureMaps(content=content_feature_maps, style=style_feature_maps)

    def __call__(self, x: torch.Tensor) -> FeatureMaps:
        return super().__call__(x)  # type: ignore[no-any-return]


class VGG11FeatureMapExtractor(BaseVGGFeatureMapExtractor):
    """VGG11-based feature map extractor from the original paper
    Content feature maps are extracted from 2nd conv layer in 4th VGG11 block
    Style feature maps are extracted from the first conv layer in each VGG11 block
    """

    def __init__(self) -> None:
        """Init the extractor"""
        vgg = models.vgg11(weights=models.VGG11_Weights.DEFAULT, progress=True).features
        # Uses:
        # - conv4_2 (2nd conv layer in 4th block) for content representation
        # - conv1, conv2, ..., conv5_1 (first conv in each block) for style representation
        content_layer_indices = [14]
        style_layer_indices = [0, 4, 7, 12, 17]
        super().__init__(vgg=vgg, content_layer_indices=content_layer_indices, style_layer_indices=style_layer_indices)


class VGG19FeatureMapExtractor(BaseVGGFeatureMapExtractor):
    """VGG19-based feature map extractor from the original paper
    Content feature maps are extracted from 2nd conv layer in 4th VGG19 block
    Style feature maps are extracted from the first conv layer in each VGG19 block
    """

    def __init__(self) -> None:
        """Init the extractor"""
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT, progress=True).features
        # As in original paper, use:
        # - conv4_2 (2nd conv layer in 4th block) for content representation
        # - conv1_1, conv2_1, ..., conv5_1 (first conv in each block) for style representation
        content_layer_indices = [21]
        style_layer_indices = [0, 5, 10, 19, 28]
        super().__init__(vgg=vgg, content_layer_indices=content_layer_indices, style_layer_indices=style_layer_indices)
