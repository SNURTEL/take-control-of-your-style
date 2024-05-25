import random
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image


class CycleGanDataset(Dataset):
    def __init__(
        self,
        content_path: Path | str,
        style_path: Path | str,
        img_size: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Dataset for training style-transfer CycleGANs.
        Images from content and style domains are shuffled and returned in pairs on __getitem__.

        Args:
            content_path: Path to content images (domain X)
            style_path: Path to style images (domain Y)
            img_size: Size to rescale images to. If None, do not rescale images.
            random_seed: Random seed for shuffling the dataset. If None, do not seed the generator.
        """
        super().__init__()
        content_path = Path(content_path)
        style_path = Path(style_path)
        assert content_path.is_dir(), "Content path must be a directory"
        assert style_path.is_dir(), "Style path must be a directory"
        extensions = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
        self.content_images = [x for x in sorted(content_path.iterdir()) if x.suffix.lower() in extensions]
        self.style_images = [x for x in sorted(style_path.iterdir()) if x.suffix.lower() in extensions]

        assert len(
            self.content_images
        ), f"No images found in content dir ({content_path}). Supported extensions are: {', '.join(extensions)}."
        assert len(
            self.style_images
        ), f"No images found in style dir ({content_path}). Supported extensions are: {', '.join(extensions)}."

        self.random = random.Random(random_seed)
        self.random.shuffle(self.content_images)
        self.random.shuffle(self.style_images)

        self.transform = T.Compose(
            [
                T.ConvertImageDtype(dtype=torch.float),
                (T.Resize(img_size) if img_size else T.Lambda(lambda t: t)),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self.transform(read_image(str(self.content_images[index % len(self.content_images)]))),
            self.transform(read_image(str(self.style_images[index % len(self.style_images)]))),
        )

    def __len__(self) -> int:
        # return 100
        return min(len(self.content_images), len(self.style_images))  # TODO try with max


class CycleGanDataModule(LightningDataModule):
    """A LightningDataModule for training style-transfer CycleGANs. Uses CycleGanDataset under the hood.
    """
    def __init__(
        self,
        content_path: Path | str,
        style_path: Path | str,
        img_size: int | None = None,
        n_val_images: int = 0,
        batch_size: int = 1,  # TODO test higher values
        random_seed: int | None = None,
    ) -> None:
        """Init the data module.

        Args:
            content_path: Path to content images (domain X)
            style_path: Path to style images (domain Y)
            img_size: Size to rescale images to. If None, do not rescale images.
            n_val_images: Number of images to transfer in the validation step. All images are fed to the network in one batch. Defaults to 0 (skip transfer in validation).
            batch_size: Batch size for training. Defaults to 1.
            random_seed: Random seed for shuffling the dataset. If None, do not seed the generator.
        """
        super().__init__()
        dataset = CycleGanDataset(content_path=content_path, style_path=style_path, img_size=img_size, random_seed=random_seed)

        generator = Generator()
        if random_seed:
            generator.manual_seed(random_seed)
        self.train, self.val = random_split(dataset, [len(dataset) - n_val_images, n_val_images], generator=generator)

        self.batch_size = batch_size
        self.n_val_images = n_val_images

    def train_dataloader(self) -> Any:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self) -> Any:
        return DataLoader(self.val, batch_size=self.n_val_images, shuffle=False, num_workers=4, pin_memory=True)
