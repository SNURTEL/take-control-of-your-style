from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms

from zprp.models.cycle_gan.components import SemanticRegularization
from zprp.models.cycle_gan.model import CycleGAN

beta_param = 0


class TestSemanticRegularization(SemanticRegularization):
    def __init__(self) -> None:
        super().__init__(beta_param=beta_param)


def load_image(path: Path) -> Tensor:
    image = Image.open(path)
    transform = transforms.ToTensor()
    image = transform(image)
    return image


def save_image(image, path: Path) -> None:
    if len(image.shape) == 4:
        image = image[0]

    if image.shape[0] in [1, 3]:  # Assuming grayscale or RGB image
        image = np.transpose(image, (1, 2, 0))

    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    pil_image.save(path)


def denormalize(x: Tensor) -> Tensor:
    return (x * 0.5) + 0.5


def inference(model: CycleGAN, image: Tensor) -> Tensor:
    output_image = model.x_to_y(image)
    output_image = denormalize(output_image)
    return output_image.detach().numpy()


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        help="Path to the trained model.",
        required=True
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to the input image.",
        required=True
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        help="Path for the output image.",
        required=True
    )

    args = parser.parse_args()
    model = CycleGAN.load_from_checkpoint(checkpoint_path=args.model).to(
        torch.device('cpu')
    )

    model.eval()
    model.g.eval()
    model.f.eval()

    image = load_image(args.image)
    output_image = inference(model, image)
    save_image(output_image, args.output_image)


if __name__ == "__main__":
    main()
