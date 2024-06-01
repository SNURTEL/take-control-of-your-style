from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from PIL import Image

from zprp.models.gatys.data import GatysDataModule
from zprp.models.gatys.extractor import VGG19FeatureMapExtractor
from zprp.models.gatys.model import GatysNST

IMG_SIZE = 256

torch.set_float32_matmul_precision("high")


def save_image(image, path: Path) -> None:
    if len(image.shape) == 4:
        image = image[0]

    image = np.transpose(image, axes=(1, 2, 0))[:, :, ::-1]
    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)
    pil_image.save(path)


def show_img(img: torch.Tensor, title: str = "") -> None:
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(img, axes=(1, 2, 0))[:, :, ::-1])
    plt.title(title)
    plt.show()


def train(
        content_image_path: str,
        style_image_path: str,
        weights: tuple[float, float] = (1e-5, 1e4),
        epochs: int = 100
) -> GatysNST:
    dm = GatysDataModule(
        content_path=content_image_path,
        style_path=style_image_path,
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


def main() -> None:
    parser = ArgumentParser(
        usage="%(prog)s [options]",
        description="Train CycleGAN model"
    )
    parser.add_argument(
        "--content-img",
        type=Path,
        help="Path to content image",
        required=True
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        help="Content loss weight",
        default=1e-5
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        help="Style loss weight",
        default=1e4
    )
    parser.add_argument(
        "--style-img",
        type=Path,
        help="Path to style image",
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
        default=100
    )
    parser.add_argument(
        '--display-image',
        action='store_true',
        help='Flag to display images'
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        help="Path to save image",
        required=False,
        default=None
    )

    args = parser.parse_args()
    model = train(
        args.content_img,
        args.style_img,
        (args.content_weight, args.style_weight),
        args.epochs
    )

    if args.display_image:
        show_img(model.image.cpu().numpy(), title="Final Image (VGG19)")

    if args.save_image is not None:
        save_image(model.image.cpu().numpy(), args.save_image)


if __name__ == "__main__":
    main()
