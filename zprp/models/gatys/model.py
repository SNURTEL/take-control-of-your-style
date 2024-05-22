from typing import Any, NamedTuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from zprp.models.gatys.extractor import FeatureMapExtractor, FeatureMaps, VGG19FeatureMapExtractor


class GatysNSTLoss(nn.Module):
    """Loss function for Gatys' neural style transfer alghoritm.
    """
    class LossDict(NamedTuple):
        total: torch.Tensor
        total_unweighted: torch.Tensor
        content: torch.Tensor
        style: torch.Tensor

    def __init__(self, targets: FeatureMaps, content_style_weights: tuple[float, float] | None = None):
        """Init the loss module.

        Args:
            targets: Content and style feature maps.
            content_style_weights: A tuple of content + style loss weights. If None, loss weights are equal.
        """
        super().__init__()

        self._content_weight, self._style_weight = content_style_weights or (1.0, 1.0)

        style_targets_gram = [self._gram_matrix(t) for t in targets.style]

        for i, target in enumerate(targets.content):
            self.register_buffer(f"con_target_{i}", target.detach())

        for i, target in enumerate(style_targets_gram):
            self.register_buffer(f"sty_target_{i}", target.detach())

    @staticmethod
    def _gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        """Calculate the Gram matrix of a given tensor.

        Args:
            tensor: A tensor of size [1, C, H, W]

        Returns:
            _description_
        """
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t()).div(b * c * h * w * 2)

        return gram_matrix

    def _content_loss(self, pred_feature_map: torch.Tensor) -> torch.Tensor:
        """Compute the content loss.

        Args:
            pred_feature_map: Content feature map of generated image

        Returns:
            Content loss value
        """
        loss = [F.mse_loss(getattr(self, f"con_target_{i}"), p, reduction="sum") for i, p in enumerate(pred_feature_map)]
        avg_loss = torch.mean(torch.stack(loss), dim=0)
        return avg_loss

    def _style_loss(self, pred_feature_map: torch.Tensor) -> torch.Tensor:
        """Cmpute the style loss.

        Args:
            pred_feature_map: Style feature map of generated image

        Returns:
            Style loss value
        """
        loss = [
            F.mse_loss(getattr(self, f"sty_target_{i}"), self._gram_matrix(p), reduction="sum")
            for i, p in enumerate(pred_feature_map)
        ]
        avg_loss = torch.mean(torch.stack(loss), dim=0)
        return avg_loss

    def forward(self, content_preds: torch.Tensor, style_preds: torch.Tensor) -> LossDict:
        """Calculate the total weighted loss

        Args:
            content_preds: Content feature map of generated image
            style_preds: Style feature map of generated image

        Returns:
            Computed loss values
        """
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
    """Implementation of the original neural style transfer alghoritm by Gatys et. al.
    https://arxiv.org/abs/1508.06576
    """
    def __init__(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        content_style_weights: tuple[float, float] | None = None,
        extractor: FeatureMapExtractor | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        log_img_every_n_epochs: int | None = None,
    ):
        """Instantiate the module

        Args:
            content_img: Content image of shape (C, H, W) and value range [0, 1]
            style_img: Style image of shape (C, H, W) and value range [0, 1]
            content_style_weights: A tuple of content + style loss weights. If None, loss weights are equal.
            extractor: Feature map extractor to use. If None, deafults to the original VGG19-based extractor.
            optimizer_kwargs: Additional kwargs to pass to the Adam optimizer. If none, lr=2e-2, betas=(0.9, 0.999) are passed.
            log_img_every_n_epochs: When using TensorBoard logging, log the transformed image every n epochs. If None, image logging is disabled.
        """
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
        """Transformed image

        Returns:
            Image tensor of shape (C, H, W) and value range [0, 1]
        """
        return self._var_img.clone().detach()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Perform on traing step - transofrm the image and log loss values.
        This function should only be called by the pytorch_lighting framework.
        """
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
        """Clamp the image value range to [0, 1].
        This function should only be called by the pytorch_lighting framework.
        """
        self._var_img.data.clamp_(0, 1)

    def configure_optimizers(self) -> optim.Optimizer:
        """Setup the Adam optimizer using either provided or default args.
        This function should only be called by the pytorch_lighting framework.
        """
        if self._optimizer_kwargs:
            return optim.Adam(params=[self._var_img], **self._optimizer_kwargs)
        else:
            return optim.Adam([self._var_img], lr=2e-2, betas=(0.9, 0.999))
