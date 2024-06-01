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
    delete_me = "delete_me"


_checkpoints_url_mapping = {
    Experiment.lambdas: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Ee6eOwEAMWtDnic7e0Vfu1cBK2OCIqurUJg2RgZpzm0hUw?e=lXMJam&download=1",
    Experiment.l2: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Eaa8KXroYI9MsbnImxvpP6MBWcUYl622HZPhGgi3_m-rFg?e=aOdte7&download=1",
    Experiment.regularization: [
        "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/ERHUIutJRQ5Lu3hfXzHIbHQBhNJ2RRkfCbPaSCSrLEMhzw?e=jMrOnn&download=1",
        "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Ee6eOwEAMWtDnic7e0Vfu1cBK2OCIqurUJg2RgZpzm0hUw?e=lXMJam&download=1",
    ],
}


def download_experiment_zip(experiment: Experiment, out_file: Path) -> Path | None:
    os.makedirs("temp", exist_ok=True)

    mapped = _checkpoints_url_mapping[experiment]
    urls = mapped if isinstance(mapped, list) else [mapped]  # type: ignore[list-item]

    try:
        for url in urls:
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

                if total_size != 0 and progress_bar.n != total_size:
                    raise RuntimeError(f"Could not download {experiment.value}")
    except KeyboardInterrupt:
        print("Stopping...")
        out_file.unlink()
        os.rmdir("temp")
        return None

    return out_file


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
        out_file = Path(f"temp/{experiment.value}.zip")
        target_path = models_dir / out_file.with_suffix("").name
        if target_path.exists():
            print(f"\"{experiment.value}\" already downloaded to {target_path.absolute()}, skipping")
            continue
        out_file = download_experiment_zip(experiment, out_file)  # type: ignore[assignment]
        if not out_file:
            break
        print(f"Extract {out_file} to {target_path}")
        shutil.unpack_archive(out_file, target_path)
        out_file.unlink()

        print("All done!")


if __name__ == "__main__":
    main()
