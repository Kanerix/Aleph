"""Command line wiring for the lookup stage.

Reads the profiles stage 2 wrote and a gallery of named reference photos, and
writes one entry per profile saying who it matched, if anyone.
"""

import argparse
import sys
from pathlib import Path

from .. import manifests
from ..device import pick_device
from ..faces import FaceReader
from . import gallery, pipeline

HELP = "match each profile against a gallery of people you already know"
SHOWN = 10


def add_parser(subparsers) -> None:
    """Register the `lookup` subcommand."""
    parser = subparsers.add_parser(
        "lookup",
        help=HELP,
        description=(
            "Match each profile from `profile` against a folder of reference "
            "photos of named people. Only people in that folder can be "
            "matched: nothing is searched for online."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(run=run)
    parser.add_argument(
        "profiles", type=Path, help="folder written by `profile` (holds profiles.json)"
    )
    parser.add_argument(
        "gallery",
        type=Path,
        help="reference photos, named per file or grouped in a folder per person",
    )
    parser.add_argument(
        "-o", "--out", default="identities", type=Path, help="output folder"
    )
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | mps | 0 (cuda index)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="cosine similarity a profile needs before it is named",
    )


def _summarise(identities) -> None:
    named = sorted(
        (entry for entry in identities if entry["match"]),
        key=lambda entry: -entry["score"],
    )
    print(f"\n{len(named)}/{len(identities)} profiles matched a face in the gallery")
    for entry in named[:SHOWN]:
        runner_up = entry["runner_up"]
        second = (
            f", then {runner_up['name']} at {runner_up['score']}" if runner_up else ""
        )
        print(f"  {entry['id']} -> {entry['match']} ({entry['score']}{second})")


def run(args) -> int:
    """Match every profile against the gallery."""
    profiles = manifests.read(args.profiles / manifests.PROFILES)
    if not profiles:
        print(f"No profiles listed in {args.profiles / manifests.PROFILES}")
        return 1
    if not args.gallery.is_dir():
        print(f"No such gallery folder: {args.gallery}", file=sys.stderr)
        return 1
    vectors = manifests.read_vectors(args.profiles / manifests.VECTORS, len(profiles))

    device = pick_device(args.device)
    print(f"Loading the face model on {device} ...")
    reader = FaceReader(device)

    print(f"\nReading the gallery in {args.gallery}/ ...")
    names, references = gallery.read(args.gallery, reader)
    if not names:
        print(f"No usable reference faces in {args.gallery}/", file=sys.stderr)
        return 1

    identities = pipeline.identify(profiles, vectors, names, references, args.threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    manifests.write(args.out / manifests.IDENTITIES, identities)

    _summarise(identities)
    print(f"\nDone: {len(identities)} profiles checked against {len(names)} people")
    return 0
