import argparse
import os
import shutil
from enum import Enum
from pathlib import Path

import requests
from tqdm import tqdm

# TODO: zip checkpoints and put them in onedrive
# Can be downloaded with wget by appending "&download=1" to sharing URL
# Does not work with directories!


class Experiment(str, Enum):
    gatys = "gatys"
    lambdas = "lambdas"
    l2 = "l2"
    regularization = "regularization"
    delete_me = "delete_me"


_checkpoints_url_mapping = {
    Experiment.gatys: "https://fill.this.up",
    Experiment.lambdas: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/Ee6eOwEAMWtDnic7e0Vfu1cBEQCPV6FXiTm1gGW6r0Zjkg?e=xF8rf2&download=1",
    Experiment.l2: "https://fill.this.up",
    Experiment.regularization: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169263_pw_edu_pl/ERHUIutJRQ5Lu3hfXzHIbHQBHSODw3dqGxEdpSwdqUIutQ?e=dPJ8y7&download=1",
    # example
    Experiment.delete_me: "https://wutwaw-my.sharepoint.com/:u:/g/personal/01169218_pw_edu_pl/EbuMmQBfrrxAriH5mpVkZyAB76Q5Jth7m8HtoZW0jCcqAQ?e=8dduXm&download=1",
}


def download_experiment_weights(experiment: Experiment) -> Path | None:
    os.makedirs("temp", exist_ok=True)
    out_file = Path(f"temp/{experiment.value}.zip")

    response = requests.get(_checkpoints_url_mapping[experiment], stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc=f'Download "{experiment.value}" to {out_file}'
    ) as progress_bar:
        try:
            with open(out_file, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            if total_size != 0 and progress_bar.n != total_size:
                raise RuntimeError("Could not download file")
        except KeyboardInterrupt:
            tqdm.write("Stopping...")
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
        temp_path = download_experiment_weights(experiment)
        if not temp_path:
            break
        target_path = models_dir / temp_path.with_suffix("").name
        print(f"Extract {temp_path} to {target_path}")
        shutil.unpack_archive(temp_path, target_path)
        temp_path.unlink()

        print("All done!")


if __name__ == "__main__":
    main()
