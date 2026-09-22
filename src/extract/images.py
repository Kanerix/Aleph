"""Find the photos to read, decode them, and write the crops back out.

Camera RAW and HEIC need decoders that not every environment has, so `rawpy`
and `pillow_heif` are imported where they are used rather than at module level.
"""

import glob as globlib
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

# Camera RAW formats, decoded through rawpy/LibRaw rather than Pillow.
RAW_EXTS = {
    ".arw",
    ".srf",
    ".sr2",  # Sony
    ".cr2",
    ".cr3",
    ".crw",  # Canon
    ".nef",
    ".nrw",  # Nikon
    ".raf",  # Fujifilm
    ".orf",  # Olympus / OM System
    ".rw2",  # Panasonic
    ".pef",
    ".dng",  # Pentax / Adobe & phones
    ".srw",  # Samsung
    ".erf",
    ".kdc",
    ".dcr",  # Epson / Kodak
    ".mrw",  # Minolta
    ".3fr",
    ".fff",  # Hasselblad
    ".iiq",
    ".mos",  # Phase One / Leaf
    ".rwl",  # Leica
    ".x3f",  # Sigma
    ".raw",
}
READABLE_EXTS = IMAGE_EXTS | RAW_EXTS


def iter_images(inputs: list[str], exclude: Path | None = None) -> list[Path]:
    """Expand files, folders and globs into a de-duplicated list of images."""
    skip_root = exclude.resolve() if exclude else None

    def excluded(path: Path) -> bool:
        # Stops a second run from re-cropping the crops when --out lives
        # inside the folder being scanned.
        return skip_root is not None and skip_root in path.resolve().parents

    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found += sorted(
                p for p in path.rglob("*") if p.suffix.lower() in READABLE_EXTS
            )
        elif path.is_file():
            found.append(path)
        else:
            matches = sorted(Path(p) for p in globlib.glob(raw))
            if not matches:
                print(f"! no such image: {raw}", file=sys.stderr)
            found += [p for p in matches if p.suffix.lower() in READABLE_EXTS]
    # de-duplicate, keep order
    seen, unique = set(), []
    for path in found:
        key = path.resolve()
        if key not in seen and not excluded(path):
            seen.add(key)
            unique.append(path)
    return unique


def unique_stems(paths: list[Path]) -> dict[Path, str]:
    """Give every photo an output stem that is unique across the whole run.

    Two memory cards both hold an IMG_0001, and crops from both would land on
    the same filenames in a single output folder. The first photo with a given
    stem keeps it and later ones get a numeric suffix, so the names stay stable
    when the same input is processed again.
    """
    stems: dict[Path, str] = {}
    used: set[str] = set()
    for path in paths:
        stem = path.stem
        candidate, n = stem, 2
        while candidate in used:
            candidate = f"{stem}-{n}"
            n += 1
        used.add(candidate)
        stems[path] = candidate
    return stems


def _embedded_preview(raw):
    """Return the camera's own JPEG preview, if the file has one."""
    import rawpy

    try:
        thumb = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None

    if thumb.format != rawpy.ThumbFormat.JPEG:
        return None

    preview = Image.open(io.BytesIO(thumb.data))
    preview = ImageOps.exif_transpose(preview)
    return preview.convert("RGB")


def _trim_black_borders(image, threshold: int = 8):
    """Drop unlit sensor margins left by LibRaw.

    For camera models LibRaw doesn't know yet it returns the whole sensor
    frame instead of the image area, which shows up as black bands down the
    right/bottom edge. Only obvious margins are trimmed - if the "border"
    would eat a quarter of the frame it is more likely a genuinely dark photo,
    so the frame is left alone.
    """
    mask = image.convert("L").point(lambda v: 255 if v >= threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return image

    x0, y0, x1, y1 = bbox
    W, H = image.size
    if (x1 - x0) < W * 0.75 or (y1 - y0) < H * 0.75:
        return image
    if (x0, y0, x1, y1) == (0, 0, W, H):
        return image
    return image.crop(bbox)


def load_raw(path: Path, args):
    """Develop a camera RAW (ARW, CR2/CR3, NEF, DNG, ...) into RGB.

    RAW files embed the JPEG the camera itself produced, and on most modern
    bodies that preview is full resolution. Using it is several times faster
    than demosaicing, and for a camera released after your LibRaw build it is
    also *more* correct: LibRaw falls back to a generic profile and hands back
    the raw sensor frame, black margins and all. So a full-size preview wins
    by default; `--raw-source develop` forces real RAW processing.

    When developing, LibRaw is given the camera's own white balance so crops
    look like the out-of-camera JPEG rather than green-ish sensor data. Auto
    brightening stays off because it rescales every frame independently, which
    makes a burst of stand photos come out inconsistently exposed.
    """
    try:
        import rawpy
    except ModuleNotFoundError:
        raise RuntimeError(
            "RAW files need rawpy - install the dependencies with `uv sync`"
        ) from None

    with rawpy.imread(str(path)) as raw:
        if args.raw_source != "develop":
            preview = _embedded_preview(raw)
            # The preview has been rotated upright but raw_width has not, so
            # compare long side with long side or portrait shots never match.
            full_size = (
                preview is not None and max(preview.size) >= raw.sizes.raw_width * 0.9
            )
            if preview is not None and (args.raw_source == "preview" or full_size):
                return preview
            if args.raw_source == "preview":
                raise RuntimeError(
                    f"{path.name} has no embedded JPEG preview - "
                    "use --raw-source develop"
                )

        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=not args.raw_auto_bright,
            bright=args.raw_bright,
            half_size=args.raw_half,
            output_bps=8,
        )

    # LibRaw has already applied the camera's orientation flag.
    return _trim_black_borders(Image.fromarray(rgb))


def load_image(path: Path, args):
    """Load any supported image (RAW, HEIC or Pillow format) as upright RGB."""
    if path.suffix.lower() in RAW_EXTS:
        return load_raw(path, args)

    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ModuleNotFoundError:
            raise RuntimeError(
                "HEIC/HEIF files need pillow-heif - install the "
                "dependencies with `uv sync`"
            ) from None

    image = Image.open(path)
    image = ImageOps.exif_transpose(image)  # honour phone/camera rotation
    return image.convert("RGB")


def save_crop(crop, dest: Path, args) -> None:
    """Write one crop to `dest`, upscaling it first if `--min-out` asks for it."""
    if args.min_out:
        short = min(crop.size)
        if 0 < short < args.min_out:
            scale = args.min_out / short
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.Resampling.LANCZOS,
            )
    suffix = dest.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        crop.save(dest, quality=args.quality, subsampling=0, optimize=True)
    elif suffix == ".webp":
        crop.save(dest, quality=args.quality)
    else:
        crop.save(dest)


def annotate(image, entries, dest: Path) -> None:
    """Save a full-size preview with crop boxes (green) and heads (red) drawn."""
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    width = max(2, round(min(preview.size) / 600))
    for entry in entries:
        draw.rectangle(entry["crop"], outline=(0, 220, 120), width=width)
        if entry.get("head"):
            draw.rectangle(
                [round(v) for v in entry["head"]], outline=(255, 90, 60), width=width
            )
        draw.text(
            (entry["crop"][0] + 2, entry["crop"][1] + 2),
            str(entry["index"]),
            fill=(255, 255, 0),
        )
    preview.save(dest, quality=88)
