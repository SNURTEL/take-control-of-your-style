from pathlib import Path
from typing import Any, Callable

import cv2
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class GatysDataset(Dataset):
    def __init__(
        self,
        content_path: Path | str,
        style_path: Path | str,
        transforms: Callable[[Any], Any] | None = None,
    ):
        super().__init__()

        transforms = transforms or (lambda x: x)
        self.content_img = transforms(cv2.imread(str(content_path)))
        self.style_img = transforms(cv2.imread(str(style_path)))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.content_img, self.style_img


class GatysDataModule(pl.LightningDataModule):
    def __init__(
        self,
        content_path: Path | str,
        style_path: Path | str,
        img_size: int | None = None,
    ) -> None:
        super().__init__()

        self.transforms = T.Compose(
            [
                T.ToTensor(),
                (T.Resize(img_size) if img_size else T.Lambda(lambda t: t)),
                T.Lambda(lambda t: torch.clip(t, min=0.0, max=1.0)),
            ]
        )

        self.train = GatysDataset(transforms=self.transforms, content_path=content_path, style_path=style_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
