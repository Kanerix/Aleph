"""Run one photo through detection, filtering and cropping."""

from pathlib import Path

from .cropping import crop_region
from .detections import merge_detections
from .images import annotate, load_image, save_crop
from .inference import Detector, detect_people


def _manifest_entry(index: int, dest: Path, path: Path, box, person) -> dict:
    face = person.face
    return {
        "index": index,
        "file": dest.name,
        "source": str(path),
        "crop": list(box),
        "confidence": round(person.conf, 4),
        "head": [float(v) for v in face["box"]] if face else None,
        "face_keypoints": int(face.get("n_kps", 0)) if face else 0,
        "facing_score": round(float(face.get("facing", 0.0)), 2) if face else 0.0,
    }


def _wanted(person, args) -> bool:
    if not args.require_face:
        return True
    face = person.face
    if not face or face["n_kps"] < args.min_kps:
        return False
    if face["width"] < args.min_face:
        return False
    if args.facing_check and not (face["votes"] and face["facing"] > 0):
        return False
    return True


def _shows_a_face(crop, person, box, reader) -> bool:
    """Whether a face model can find a face in the crop the pose model asked for.

    The pose model happily predicts a person on a raised hand, a banner or a
    stretch of fence, and no amount of keypoint filtering catches all of it.
    A model that only knows about faces settles the question.
    """
    if reader is None:
        return True
    head = None
    if person.face:
        x0, y0, x1, y1 = person.face["box"]
        head = (x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1])
    return reader.read(crop, head) is not None


def process_image(path: Path, stem: str, detector: Detector, reader, args):
    """Detect, filter, crop and save every usable face in one photo.

    Crops are written to `args.out` as `<stem>_001.<format>`. Returns one
    manifest entry per written crop.
    """
    image = load_image(path, args)
    W, H = image.size

    people = merge_detections(detect_people(detector, image, args), args.merge_iou)
    kept = [person for person in people if _wanted(person, args)]

    # top-to-bottom, left-to-right so the numbering follows the crowd
    kept.sort(key=lambda p: (round(p.box[1] / max(1.0, H * 0.02)), p.box[0]))

    entries: list[dict] = []
    faceless = 0
    for person in kept:
        box = crop_region(person, W, H, args)
        if (box[2] - box[0]) < args.min_size or (box[3] - box[1]) < args.min_size:
            continue
        crop = image.crop(box)
        if not _shows_a_face(crop, person, box, reader):
            faceless += 1
            continue
        # Numbered after the filters so the sequence has no gaps.
        index = len(entries) + 1
        dest = args.out / f"{stem}_{index:03d}.{args.format}"
        save_crop(crop, dest, args)
        entries.append(_manifest_entry(index, dest, path, box, person))

    if args.annotate and entries:
        annotate(image, entries, args.out / f"{stem}_preview.jpg")

    dropped = f", {faceless} with no face" if faceless else ""
    print(f"  {len(entries)} crops from {len(people)} detections ({W}x{H}){dropped}")
    return entries
