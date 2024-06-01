from PIL import Image


def images_to_gif(images_paths: list[str], output_image: str, duration: int) -> None:
    images = [Image.open(path) for path in images_paths]

    image = images[0]
    image.save(output_image, format="GIF", append_images=images[1:], save_all=True, duration=duration, loop=0)

    for image in images:
        image.close()


def main() -> None:
    images_to_gif([f"temp/comparison/gatys{idx}.png" for idx in range(10)], "temp/comparison/gatys_comparison.gif", 300)
    images_to_gif(
        [f"temp/comparison/cycle{idx}.png" for idx in range(4)], "temp/comparison/cyclegan_comparison.gif", 750
    )


if __name__ == "__main__":
    main()
