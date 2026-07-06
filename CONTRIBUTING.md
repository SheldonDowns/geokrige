# Contributing to geokrige

Thanks for considering a contribution!

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/yourusername/geokrige
cd geokrige
uv sync --all-extras
```

## Running tests

```bash
uv run pytest
```

## Linting

```bash
uv run ruff check src tests
```

## Building docs locally

```bash
uv run mkdocs serve
```

## Submitting changes

1. Fork the repo and create a feature branch.
2. Add tests for any new behavior.
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. Open a pull request describing the change and motivation.

By contributing, you agree to abide by the project's `CONDUCT.md`.
