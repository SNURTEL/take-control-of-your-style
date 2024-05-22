from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as T


class FeatureMaps(NamedTuple):
    content: list[torch.Tensor]
    style: list[torch.Tensor]


class FeatureMapExtractor(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> FeatureMaps:
        ...


class VGG19FeatureMapExtractor(FeatureMapExtractor):
    def __init__(self) -> None:
        super().__init__()

        # As in original paper, use:
        # - conv4_2 (2nd conv layer in 4th block) for content representation
        # - conv1_1, conv2_1, ..., conv5_1 (first conv in each block) for style representation
        self._content_layer_indices = [21]
        self._style_layer_indices = [0, 5, 10, 19, 28]

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.vgg19 = models.vgg19(pretrained=True, progress=True).features
        self.vgg19 = self.vgg19.eval()
        self.vgg19.requires_grad_(False)

        # misc improvements
        for name, layer in self.vgg19.named_children():
            if isinstance(layer, nn.MaxPool2d):
                self.vgg19[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)
            if isinstance(layer, nn.ReLU):
                self.vgg19[int(name)] = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        content_feature_maps = []
        style_feature_maps = []

        x = self.normalize(x).unsqueeze(0)

        for idx, layer in enumerate(self.vgg19.children()):
            x = layer(x)
            if idx in self._content_layer_indices:
                content_feature_maps.append(x)
            if idx in self._style_layer_indices:
                style_feature_maps.append(x)

        return FeatureMaps(content=content_feature_maps, style=style_feature_maps)

    def __call__(self, x: torch.Tensor) -> FeatureMaps:
        return super().__call__(x)  # type: ignore[no-any-return]


class GatysNSTLoss(nn.Module):
    class LossDict(NamedTuple):
        total: torch.Tensor
        total_unweighted: torch.Tensor
        content: torch.Tensor
        style: torch.Tensor

    def __init__(self, targets: FeatureMaps, content_style_weights: tuple[float, float] | None = None):
        super().__init__()

        self._content_weight, self._style_weight = content_style_weights or (1.0, 1.0)

        style_targets_gram = [self._gram_matrix(t) for t in targets.style]

        for i, target in enumerate(targets.content):
            self.register_buffer(f"con_target_{i}", target.detach())

        for i, target in enumerate(style_targets_gram):
            self.register_buffer(f"sty_target_{i}", target.detach())

    @staticmethod
    def _gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t()).div(b * c * h * w * 2)

        return gram_matrix

    def _content_loss(self, preds: torch.Tensor) -> torch.Tensor:
        loss = [F.mse_loss(getattr(self, f"con_target_{i}"), p, reduction="sum") for i, p in enumerate(preds)]
        # TODO torch.mean
        avg_loss = torch.sum(torch.stack(loss)) / len(loss)
        return avg_loss

    def _style_loss(self, preds: torch.Tensor) -> torch.Tensor:
        loss = [
            F.mse_loss(getattr(self, f"sty_target_{i}"), self._gram_matrix(p), reduction="sum")
            for i, p in enumerate(preds)
        ]
        avg_loss = torch.sum(torch.stack(loss)) / len(loss)
        return avg_loss

    def forward(self, content_preds: torch.Tensor, style_preds: torch.Tensor) -> LossDict:
        con_loss = self._content_loss(content_preds)
        sty_loss = self._style_loss(style_preds)
        return self.LossDict(
            content=con_loss,
            style=sty_loss,
            total=con_loss * self._content_weight + sty_loss * self._style_weight,
            total_unweighted=con_loss + sty_loss,
        )

    def __call__(self, content_preds: torch.Tensor, style_preds: torch.Tensor) -> LossDict:
        return super().__call__(content_preds, style_preds)  # type: ignore[no-any-return]


class GatysNST(pl.LightningModule):
    def __init__(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        content_style_weights: tuple[float, float] | None = None,
        extractor: FeatureMapExtractor | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        log_img_every_n_epochs: int | None = None,
    ):
        super().__init__()

        assert log_img_every_n_epochs is None or log_img_every_n_epochs > 0, "log_img_every_n_epochs must be positive"

        self.extractor = extractor or VGG19FeatureMapExtractor()
        content_img_feature_maps = self.extractor(content_img)
        style_img_feature_maps = self.extractor(style_img)


        self._var_img = nn.Parameter(content_img.clone())

        self.loss = GatysNSTLoss(
            targets=FeatureMaps(content=content_img_feature_maps.content, style=style_img_feature_maps.style),
            content_style_weights=content_style_weights,
        )

        self._optimizer_kwargs = optimizer_kwargs

        self._log_img_every_n_epochs = log_img_every_n_epochs

    @property
    def image(self) -> torch.Tensor:
        return self._var_img

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        feature_maps = self.extractor(self._var_img)
        loss_dict = self.loss(content_preds=feature_maps.content, style_preds=feature_maps.style)
        self.log_dict(loss_dict._asdict(), on_epoch=True)

        if (
            self._log_img_every_n_epochs is not None
            and self.logger
            and self.current_epoch % self._log_img_every_n_epochs == 0
        ):
            grid = torchvision.utils.make_grid(self._var_img)
            self.logger.experiment.add_image("image", grid, self.current_epoch)

        return loss_dict.total

    def on_train_epoch_end(self) -> None:
        self._var_img.data.clamp_(0, 1)

    def configure_optimizers(self) -> optim.Optimizer:
        if self._optimizer_kwargs:
            return optim.Adam(params=[self._var_img], **self._optimizer_kwargs)
        else:
            return optim.Adam([self._var_img], lr=2e-2, betas=(0.9, 0.999))
