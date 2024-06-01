from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from zprp.models.cycle_gan.data import CycleGanDataModule
from zprp.models.cycle_gan.model import CycleGAN

IMG_SIZE = 256
DATA_PATH = Path("data/monet2photo")

torch.set_float32_matmul_precision("high")


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5) + 0.5


def show_images(model, title, dm):
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    model.eval()
    model.g.eval()
    model.f.eval()

    images_x, images_y = next(iter(dm.val_dataloader()))
    images_x = images_x[:16].to(device)
    images_y = images_y[:16].to(device)

    with torch.no_grad():
        fake_y = model.x_to_y(images_x)
        cycle_x = model.y_to_x(fake_y)
        fake_x = model.y_to_x(images_y)
        cycle_y = model.x_to_y(fake_x)

    fig, axes = plt.subplots(2, 3)

    axes[0][0].imshow(make_grid(denormalize(images_x), nrow=4).permute(1, 2, 0).cpu())
    axes[0][0].set_title("Real photo")
    axes[0][1].imshow(make_grid(denormalize(fake_y), nrow=4).permute(1, 2, 0).cpu())
    axes[0][1].set_title("Fake painting")
    axes[0][2].imshow(make_grid(denormalize(cycle_x), nrow=4).permute(1, 2, 0).cpu())
    axes[0][2].set_title("Recreated photo")

    axes[1][0].imshow(make_grid(denormalize(images_y), nrow=4).permute(1, 2, 0).cpu())
    axes[1][0].set_title("Real painting")
    axes[1][1].imshow(make_grid(denormalize(fake_x), nrow=4).permute(1, 2, 0).cpu())
    axes[1][1].set_title("Fake photo")
    axes[1][2].imshow(make_grid(denormalize(cycle_y), nrow=4).permute(1, 2, 0).cpu())
    axes[1][2].set_title("Recreated painting")

    for ax in axes.reshape(-1):
        ax.axis("off")

    if title:
        fig.suptitle(title, size=22)

    plt.show()


def train(lambda_param: float = 0.5, epochs: int = 10):
    dm = CycleGanDataModule(
        content_path=DATA_PATH / "trainB/",
        style_path=DATA_PATH / "trainA/",
        n_val_images=16,
        batch_size=4,
        img_size=IMG_SIZE,
    )
    dm.setup("fit")

    model = CycleGAN(
        lambda_param=lambda_param,
        optimizer_kwargs={"lr": 0.0002, "betas": (0.5, 0.999)},
    )

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, dm)

    return model, trainer, dm


def main() -> None:
    parser = ArgumentParser(usage="%(prog)s [options]", description="Train CycleGAN model")
    parser.add_argument("--save", type=Path, help="Path to save the trained model", required=True)
    parser.add_argument("--lambda-param", type=float, help="Lambda parameter for model", required=False, default=0.5)
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=10)
    parser.add_argument("--display-images", action="store_true", help="Flag to display images")

    args = parser.parse_args()
    model, trainer, dm = train(args.lambda_param, args.epochs)

    if args.display_images:
        show_images(model, "Final images", dm)

    trainer.save_checkpoint(args.save)


if __name__ == "__main__":
    main()
