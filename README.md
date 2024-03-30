# 24L-ZPRP

Description goes here.

## Setup

#### Prerequisites  

- Python >=3.11
- `conda`

#### Install

```shell
conda env create --name zprp --file=environment.yml
poetry install [--no-dev]
```

This will re-create the conda environment (mostly `pytorch` related dependencies) and install other project deps plus some extra tools - `ruff`, `mypy`, `pytest`, etc. (if `--no-dev` was not passed).

#### Run tests

```shell
poetry run pytest -v
```

## Contributing

**NOTE** - when adding dependencies to the project, try to maximize the use of `poetry` - we don't want to rely on `conda` in anything that is not strictly `pytorch` or CUDA related.


Before submitting a PR:

- 1. Reformat the code

```shell
poetry run ruff format
```

- 2. Lint with `mypy` and `ruff`

```shell
poetry run mypy
```

```shell
poetry run ruff check [--fix]
```

- 3. Run tests as described above.