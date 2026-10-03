"""Command-line entry point for ``python -m pc_cmps_rwgan``."""

from .train import parse_args, run_pipeline


if __name__ == "__main__":
    run_pipeline(parse_args())
