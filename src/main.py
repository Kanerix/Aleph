"""Aleph: find the faces in a photo, then work out whose they are.

The pipeline is a chain of stages, one subcommand each. Every stage reads the
folder the previous one wrote, including its manifest, so the stages can be
run separately and re-run on their own:

    uv run python -m src.main extract data/ --out faces
    uv run python -m src.main profile faces/ --out profiles
    uv run python -m src.main lookup profiles/ known/ --out identities

Run `python -m src.main <stage> --help` for the flags of one stage.
"""

import argparse

from .extract import cli as extract_cli
from .lookup import cli as lookup_cli
from .profiles import cli as profiles_cli

STAGES = (extract_cli, profiles_cli, lookup_cli)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser, with one subcommand per pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="aleph",
        description="Facial recognition and identity discovery pipeline.",
    )
    stages = parser.add_subparsers(dest="stage", required=True, metavar="stage")
    for stage in STAGES:
        stage.add_parser(stages)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one pipeline stage. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
