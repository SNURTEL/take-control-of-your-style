# 24L-ZPRP

Description goes here.

## Setup

#### Prerequisites  

- Python >=3.11
- `poetry`

#### Install

```shell
poetry install [--no-dev]
```

This will install project dependencies plus (if `--no-dev` was not passed) some extra tools - `ruff`, `mypy`, `pytest`, etc.

#### Run tests

```shell
poetry run pytest -v
```

## Contributing

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