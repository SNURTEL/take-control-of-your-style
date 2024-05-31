import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

from zprp.models.gatys.data import GatysDataModule
from zprp.models.gatys.extractor import VGG11FeatureMapExtractor, \
    VGG19FeatureMapExtractor
from zprp.models.gatys.model import GatysNST

IMG_SIZE = 256
DATA_PATH = Path("data/monet2photo")

torch.set_float32_matmul_precision("high")


def show_img(img: torch.Tensor, title: str = "") -> None:
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(img, axes=(1, 2, 0))[:, :, ::-1])
    plt.title(title)
    plt.show()


def train(
        content_image_name: str,
        style_image_name: str,
        weights: tuple[float, float] = (1e-5, 1e4),
        epochs: int = 100
):
    dm = GatysDataModule(
        content_path=DATA_PATH / f"trainB/{content_image_name}",
        style_path=DATA_PATH / f"trainA/{style_image_name}",
        img_size=IMG_SIZE
    )
    dm.setup("fit")

    content_img, style_img = dm.train[0]

    model = GatysNST(
        content_img=content_img,
        style_img=style_img,
        extractor=VGG19FeatureMapExtractor(),
        content_style_weights=weights,
        log_img_every_n_epochs=10
    )

    trainer = pl.Trainer(
        max_epochs=epochs, enable_progress_bar=False,
        enable_checkpointing=False, log_every_n_steps=1
    )
    trainer.fit(model, dm)

    return model


if __name__ == "__main__":
    model = train("2013-11-08 16_45_24.jpg", "00001.jpg")
    show_img(model.image.cpu().numpy(), title="Final Image (VGG19)")
