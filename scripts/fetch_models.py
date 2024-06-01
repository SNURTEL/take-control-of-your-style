#!/usr/bin/env python3

import argparse
import os
import shutil
from enum import Enum
from pathlib import Path

import requests
from tqdm import tqdm


class Experiment(str, Enum):
    lambdas = "lambdas"
    l2 = "l2"
    regularization = "regularization"


_checkpoints_url_mapping = {
    Experiment.lambdas: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Ee6eOwEAMWtDnic7e0Vfu1cBK2OCIqurUJg2RgZpzm0hUw?e=lXMJam&download=1",
    Experiment.l2: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Eaa8KXroYI9MsbnImxvpP6MBWcUYl622HZPhGgi3_m-rFg?e=aOdte7&download=1",
    Experiment.regularization: [
        "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/ERHUIutJRQ5Lu3hfXzHIbHQBhNJ2RRkfCbPaSCSrLEMhzw?e=jMrOnn&download=1",
        "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Ee6eOwEAMWtDnic7e0Vfu1cBK2OCIqurUJg2RgZpzm0hUw?e=lXMJam&download=1",
    ],
}


def download_experiment_weights(experiment: Experiment, target_path: Path) -> Path | None:
    """Download pre-trained models form given experiment as PL checkpoints.

    Args:
        experiment: Experiment name
        target_path: Target directory to extract checkpoints

    Returns:
        Target directory if downloaded successfully, None if stopped.
    """
    os.makedirs("temp", exist_ok=True)

    mapped = _checkpoints_url_mapping[experiment]
    urls = mapped if isinstance(mapped, list) else [mapped]  # type: ignore[list-item]
    out_files = [Path(f"temp/{experiment.value}_{i}.zip") for i in range(len(urls))]

    try:
        for url, out_file in zip(urls, out_files):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=f'Download "{experiment.value}" to {out_file}'
            ) as progress_bar:
                with open(out_file, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)

            print(f"Extract {out_file} to {target_path}")
            shutil.unpack_archive(out_file, target_path)
            out_file.unlink()

    except KeyboardInterrupt:
        print("Stopping...")
        out_file.unlink()
        return None

    return target_path


def main() -> None:
    cwd = Path(os.getcwd())
    repo_dir = cwd if cwd.name != "scripts" else cwd.parent
    models_dir = repo_dir / "models"

    experiments = [str(e.value) for e in Experiment]
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", choices=experiments + ["all"])
    args = parser.parse_args()
    to_download = set(args.experiments) if "all" not in args.experiments else experiments

    for experiment in [Experiment(v) for v in to_download]:
        target_path = models_dir / experiment.value
        if target_path.exists() and target_path.is_dir() and len(list(target_path.iterdir())) > 0:
            print(f'"{experiment.value}" already downloaded to {target_path.absolute()}, skipping')
            continue
        download_experiment_weights(experiment, target_path)  # type: ignore[assignment]

    print("All done!")


if __name__ == "__main__":
    main()
