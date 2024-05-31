import os
from argparse import ArgumentParser
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(dataset: str, datasets_dir: Path) -> None:
    api = KaggleApi()
    api.authenticate()

    name = dataset.split("/")[-1]
    dataset_dir = datasets_dir / name

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print(f"Downloading dataset {name} to {dataset_dir}")
    api.dataset_download_files(dataset, path=dataset_dir, unzip=True)


def main() -> None:
    cwd = Path(os.getcwd())
    repo_dir = cwd if cwd.name != "scripts" else cwd.parent
    dataset_dir = repo_dir / "data"

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset to fetch from Kaggle",
        default="balraj98/monet2photo"
    )
    args = parser.parse_args()
    dataset = args.dataset

    download_dataset(dataset, dataset_dir)

    print("All done!")


if __name__ == '__main__':
    main()
