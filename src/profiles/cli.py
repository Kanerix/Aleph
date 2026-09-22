"""Command line wiring for the profile stage.

Reads the crops and manifest written by `extract`, and writes one folder and
one manifest entry per person.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from .. import manifests
from ..device import pick_device, torch_device
from ..faces import FaceReader
from . import pipeline
from .encoder import MODEL, WEIGHTS, Encoder

HELP = "group crops of the same person and tag them with keywords"


def add_parser(subparsers) -> None:
    """Register the `profile` subcommand."""
    parser = subparsers.add_parser(
        "profile",
        help=HELP,
        description=(
            "Group the crops from `extract` into one profile per person and "
            "describe each profile with keywords."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(run=run)
    parser.add_argument(
        "crops", type=Path, help="folder written by `extract` (holds manifest.json)"
    )
    parser.add_argument(
        "-o", "--out", default="profiles", type=Path, help="output folder"
    )
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | mps | 0 (cuda index)"
    )

    group = parser.add_argument_group("identity")
    group.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="cosine similarity two faces need to count as the same person",
    )
    group.add_argument(
        "--batch", type=int, default=32, help="crops encoded per forward pass"
    )

    group = parser.add_argument_group("keywords")
    group.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="drop keywords the model is less sure of than this",
    )
    group.add_argument("--model", default=MODEL, help="open_clip model name")
    group.add_argument("--weights", default=WEIGHTS, help="open_clip pretrained tag")


def _summarise(profiles) -> None:
    repeated = [p for p in profiles if p["appearances"] > 1]
    print(f"\n{len(profiles)} profiles, {len(repeated)} seen in more than one photo")
    if repeated:
        best = max(repeated, key=lambda p: p["appearances"])
        print(f"  most seen: {best['id']} in {best['appearances']} crops")
    tally = Counter(word for profile in profiles for word in profile["keywords"])
    common = ", ".join(f"{word} ({n})" for word, n in tally.most_common(8))
    print(f"  common keywords: {common}")


def run(args) -> int:
    """Group the crops into profiles and describe them."""
    entries = manifests.read(args.crops / manifests.CROPS)
    if not entries:
        print(f"No crops listed in {args.crops / manifests.CROPS}")
        return 1

    saved = args.crops / manifests.VECTORS
    cached = manifests.read_vectors(saved, len(entries)) if saved.exists() else None

    device = torch_device(pick_device(args.device))
    print(f"Loading {args.model} on {device} ...")
    encoder = Encoder(device, args.model, args.weights)

    if cached is None:
        print("Reading the faces, as --no-verify left no vectors to reuse ...")
        faces = pipeline.read_faces(entries, args.crops, FaceReader(device))
    else:
        faces = pipeline.saved_faces(entries, cached)

    vectors, described = pipeline.describe(
        entries, args.crops, encoder, faces, args.batch
    )
    if not described:
        print("No faces recognised in any crop", file=sys.stderr)
        return 1
    print(f"\n{len(described)}/{len(entries)} crops held a recognisable face")

    profiles, identities = pipeline.build(described, vectors, args)
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline.write_folders(profiles, args.crops, args.out)
    manifests.write(args.out / manifests.PROFILES, profiles)
    manifests.write_vectors(args.out / manifests.VECTORS, identities)

    _summarise(profiles)
    print(f"\nDone: {len(profiles)} profiles in {args.out}/")
    return 0
