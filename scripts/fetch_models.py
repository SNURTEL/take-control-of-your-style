import os
import argparse
from enum import StrEnum, auto

# TODO: zip checkpoints and put them in onedrive
# Can be downloaded with wget by appending "&download=1" to sharing URL
# Does not work with directories!


class Experiment(StrEnum):
    gatys = auto()
    lambdas = auto()
    l2 = auto()
    regularization = auto()


_checkpoints_url_mapping = {
    Experiment.gatys: "https://fill.this.up",
    Experiment.lambdas: "https://fill.this.up",
    Experiment.l2: "https://fill.this.up",
    Experiment.regularization: "https://fill.this.up"
}

def download_experiment_weights(experiment: Experiment) -> None:
    # TODO implement
    print(f"Fetch \"{experiment.value}\"")


def main() -> None:
    experiments = [str(e) for e in Experiment]
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", choices=experiments + ["all"])
    args = parser.parse_args()
    to_download = set(args.experiments) if "all" not in args.experiments else experiments
    
    for experiment in [Experiment(v) for v in to_download]:
        download_experiment_weights(experiment)



if __name__ == "__main__":
    main()
