from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor, optim


class LSGANLoss(nn.Module):
    @staticmethod
    def forward(preds: Tensor, targets: Tensor) -> Tensor:
        return torch.mean((preds - targets) ** 2)


class CycleConsistencyLoss(nn.Module):
    """L1 norm"""

    @staticmethod
    def forward(real: Tensor, reconstructed: Tensor) -> Tensor:
        return torch.mean(torch.abs(real - reconstructed))


# lr=lr, betas=(0.0002, 0.999)
class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        GeneratorClass: type[nn.Module],
        DiscriminatorClass: type[nn.Module],
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.g = GeneratorClass()
        self.f = GeneratorClass()
        self.dx = DiscriminatorClass()
        self.dy = DiscriminatorClass()

        self._optimizer_kwargs = optimizer_kwargs or {}

    def configure_optimizers(
        self,
    ) -> tuple[Sequence[torch.optim.Optimizer], Sequence[Any]]:
        g_optimizer = optim.Adam(self.g.parameters(), **self._optimizer_kwargs)
        f_optimizer = optim.Adam(self.f.parameters(), **self._optimizer_kwargs)
        dx_optimizer = optim.Adam(self.dx.parameters(), **self._optimizer_kwargs)
        dy_optimizer = optim.Adam(self.dy.parameters(), **self._optimizer_kwargs)
        return [g_optimizer, f_optimizer, dx_optimizer, dy_optimizer], []
