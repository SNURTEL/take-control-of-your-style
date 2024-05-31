from pathlib import Path

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from zprp.models.cycle_gan.components import SemanticRegularization
from zprp.models.cycle_gan.data import CycleGanDataModule
from zprp.models.cycle_gan.model import CycleGAN

IMG_SIZE = 256
DATA_PATH = Path("data/monet2photo")

torch.set_float32_matmul_precision("high")


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5) + 0.5


def show_images(model, title, dm):
    device = torch.device("cuda") if torch.cuda.is_available else torch.device(
        "cpu")

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

    axes[0][0].imshow(
        make_grid(unnormalize(images_x), nrow=4).permute(1, 2, 0).cpu())
    axes[0][0].set_title("Real photo")
    axes[0][1].imshow(
        make_grid(unnormalize(fake_y), nrow=4).permute(1, 2, 0).cpu())
    axes[0][1].set_title("Fake painting")
    axes[0][2].imshow(
        make_grid(unnormalize(cycle_x), nrow=4).permute(1, 2, 0).cpu())
    axes[0][2].set_title("Recreated photo")

    axes[1][0].imshow(
        make_grid(unnormalize(images_y), nrow=4).permute(1, 2, 0).cpu())
    axes[1][0].set_title("Real painting")
    axes[1][1].imshow(
        make_grid(unnormalize(fake_x), nrow=4).permute(1, 2, 0).cpu())
    axes[1][1].set_title("Fake photo")
    axes[1][2].imshow(
        make_grid(unnormalize(cycle_y), nrow=4).permute(1, 2, 0).cpu())
    axes[1][2].set_title("Recreated painting")

    for ax in axes.reshape(-1):
        ax.axis("off")

    if title:
        fig.suptitle(title, size=22)

    plt.show()


def train(
        lambda_param: float = 0.5,
        epochs: int = 10
):
    dm = CycleGanDataModule(
        content_path=DATA_PATH / f"trainB/",
        style_path=DATA_PATH / f"trainA/",
        n_val_images=16,
        batch_size=4,
        img_size=IMG_SIZE
    )
    dm.setup("fit")

    model = CycleGAN(
        lambda_param=lambda_param,
        optimizer_kwargs={"lr": 0.0002, "betas": (0.5, 0.999)},
    )

    trainer = pl.Trainer(
        max_epochs=epochs
    )
    trainer.fit(model, dm)

    return model, dm


if __name__ == "__main__":
    model, dm = train()
    show_images(model, f"CycleGAN", dm)
