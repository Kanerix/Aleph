"""Command line wiring for the face extraction stage.

Reads photos, writes `<out>/<photo>_001.jpg` and the manifest that stage 2
reads. Model weights download themselves on first run.
"""

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

from .. import manifests
from ..device import pick_device
from ..faces import FaceReader
from .cropping import parse_aspect
from .images import RAW_EXTS, iter_images, unique_stems
from .inference import Detector
from .pipeline import process_image

HELP = "find people in photos and crop one portrait per person"


def add_parser(subparsers) -> None:
    """Register the `extract` subcommand."""
    parser = subparsers.add_parser(
        "extract",
        help=HELP,
        description=(
            "Extract faces (with body) of everyone in a photo. Accepts "
            "JPEG/PNG/TIFF/HEIC and camera RAW (.ARW, .CR2, .NEF, .DNG, ...)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(run=run)
    parser.add_argument("inputs", nargs="+", help="image file(s), directory, or glob")
    parser.add_argument("-o", "--out", default="faces", type=Path, help="output folder")
    parser.add_argument(
        "-m",
        "--model",
        default="yolo11m-pose.pt",
        help="Ultralytics pose weights; yolo11x-pose.pt finds more, "
        "yolo11n-pose.pt is fastest",
    )
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | mps | 0 (cuda index)"
    )

    group = parser.add_argument_group("detection")
    group.add_argument(
        "--conf", type=float, default=0.25, help="person confidence threshold"
    )
    group.add_argument(
        "--imgsz", type=int, default=1280, help="inference resolution per tile"
    )
    group.add_argument(
        "--tile", type=int, default=768, help="tile size in source pixels"
    )
    group.add_argument(
        "--tile-overlap", type=float, default=0.25, help="fraction of tile overlap"
    )
    group.add_argument(
        "--no-tile", action="store_true", help="single full-image pass only"
    )
    group.add_argument("--batch", type=int, default=8, help="tiles per inference batch")
    group.add_argument("--nms", type=float, default=0.55, help="within-tile NMS IoU")
    group.add_argument(
        "--merge-iou", type=float, default=0.55, help="cross-tile merge IoU"
    )
    group.add_argument(
        "--max-det", type=int, default=300, help="max detections per tile"
    )

    group = parser.add_argument_group("face filter")
    group.add_argument(
        "--kp-conf", type=float, default=0.35, help="facial keypoint confidence"
    )
    group.add_argument(
        "--min-kps", type=int, default=2, help="visible facial keypoints required"
    )
    group.add_argument(
        "--min-face",
        type=float,
        default=16.0,
        help="min head width in px - the main quality filter",
    )
    group.add_argument(
        "--no-require-face",
        dest="require_face",
        action="store_false",
        help="also keep people whose face is hidden or turned away",
    )
    group.add_argument(
        "--no-facing-check",
        dest="facing_check",
        action="store_false",
        help="skip the left/right keypoint test that rejects backs of heads",
    )
    group.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="keep crops that a face recognition model finds no face in "
        "(hands, banners, fence panels)",
    )

    group = parser.add_argument_group("crop")
    group.add_argument(
        "--pad", type=float, default=0.08, help="padding as a fraction of the box"
    )
    group.add_argument(
        "--body",
        type=float,
        default=2.5,
        help="how much torso to keep, in head-heights below the chin",
    )
    group.add_argument(
        "--frame-width",
        type=float,
        default=3.2,
        help="max crop width, in head-widths (stops arms/flags widening the crop)",
    )
    group.add_argument(
        "--aspect",
        type=parse_aspect,
        default=None,
        help="force an aspect ratio, e.g. 3:4, 1:1",
    )
    group.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="skip crops whose short side is under N source px (0 = keep all; "
        "--min-face is the better quality knob)",
    )
    group.add_argument(
        "--min-out", type=int, default=0, help="upscale so the short side is >= N px"
    )
    group.add_argument("--format", default="jpg", choices=("jpg", "png", "webp"))
    group.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality")
    group.add_argument(
        "--annotate", action="store_true", help="also save a preview with boxes drawn"
    )

    group = parser.add_argument_group("raw files (.arw, .cr2/.cr3, .nef, .dng, ...)")
    group.add_argument(
        "--raw-source",
        choices=("auto", "preview", "develop"),
        default="auto",
        help="auto: use the embedded JPEG when it is full size, else develop the RAW",
    )
    group.add_argument(
        "--raw-half",
        action="store_true",
        help="develop RAW at half resolution (much faster, finds fewer distant faces)",
    )
    group.add_argument(
        "--raw-auto-bright",
        action="store_true",
        help="let LibRaw auto-level each RAW instead of keeping the exposure as shot",
    )
    group.add_argument(
        "--raw-bright", type=float, default=1.0, help="RAW brightness multiplier"
    )


def _save_vectors(path: Path, vectors, verify: bool) -> None:
    """Save the identity vectors, or clear the ones an earlier run left.

    `--no-verify` reads no faces, so there is nothing to save and a file from a
    previous run would be matched against the crops this one just wrote.
    """
    if verify:
        manifests.write_vectors(path, vectors)
    else:
        path.unlink(missing_ok=True)


def run(args) -> int:
    """Crop every usable face out of the given photos."""
    images = iter_images(args.inputs, exclude=args.out)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    if any(p.suffix.lower() in RAW_EXTS for p in images) and not find_spec("rawpy"):
        print(
            "RAW files were given but rawpy is not installed.\n  uv sync",
            file=sys.stderr,
        )
        return 1

    device = pick_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    detector = Detector(args.model, device, args)
    if not detector.pose and args.require_face:
        print(
            f"! {args.model} is not a pose model, so faces cannot be found.\n"
            "  Use a *-pose.pt model, or pass --no-require-face to crop every person.",
            file=sys.stderr,
        )
        return 1

    reader = None
    if args.verify:
        print("Loading the face model ...")
        reader = FaceReader(device)

    args.out.mkdir(parents=True, exist_ok=True)
    stems = unique_stems(images)
    manifest: list[dict] = []
    vectors = []
    for i, path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {path}")
        try:
            entries, read = process_image(path, stems[path], detector, reader, args)
        except Exception as exc:  # keep going through a batch of photos
            print(f"  ! failed: {exc}", file=sys.stderr)
            continue
        manifest += entries
        vectors += read

    manifests.write(args.out / manifests.CROPS, manifest)
    _save_vectors(args.out / manifests.VECTORS, vectors, args.verify)
    print(f"\nDone: {len(manifest)} crops in {args.out}/")
    return 0
