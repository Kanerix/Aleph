"""Turn a detected person into the box that gets cropped out of the photo."""

import argparse

from .detections import Person


def parse_aspect(value: str | None) -> float | None:
    """Parse an `--aspect` argument such as "3:4" or "0.75" into a ratio."""
    if not value:
        return None
    try:
        if ":" in value:
            w, h = value.split(":", 1)
            ratio = float(w) / float(h)
        else:
            ratio = float(value)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(
            f"invalid aspect ratio: {value!r} (use e.g. 3:4)"
        ) from None
    if ratio <= 0:
        raise argparse.ArgumentTypeError("aspect ratio must be positive")
    return ratio


def apply_aspect(
    box,
    ratio: float,
    size: tuple[int, int],
    anchor_x: float | None = None,
    grow_down: float = 0.8,
):
    """Grow a box to the requested aspect ratio.

    Extra width is added around the head (`anchor_x`) and extra height mostly
    below it, so the subject keeps portrait-style framing instead of drifting
    towards whoever is standing next to them.
    """
    W, H = size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return box

    if w / h < ratio:
        w = h * ratio
    else:
        h = w / ratio

    # Never ask for more pixels than the image has.
    if w > W:
        w, h = float(W), W / ratio
    if h > H:
        h, w = float(H), H * ratio

    cx = anchor_x if anchor_x is not None else (x0 + x1) / 2
    new_x0 = cx - w / 2
    new_y0 = y0 - (h - (y1 - y0)) * (1.0 - grow_down)

    new_x0 = min(max(new_x0, 0.0), W - w)
    new_y0 = min(max(new_y0, 0.0), H - h)
    return (new_x0, new_y0, new_x0 + w, new_y0 + h)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def crop_region(person: Person, W: int, H: int, args):
    """Frame the person: head anchored near the top, torso below.

    The detector's box is a poor framing guide in a stand - it clips hair and
    stops at the chin when the torso is hidden behind the row in front, and
    it balloons across half a row when someone raises their arms or a flag.
    So the box is only allowed to move the crop within limits measured in
    head sizes, which keeps every output consistently framed.
    """
    x0, y0, x1, y1 = person.box
    face = person.face
    anchor_x = None

    if face:
        cx = face["center"][0]
        anchor_x = cx
        hw, hh = face["width"], face["height"]
        hx0, hy0, hx1, hy1 = face["box"]

        half = max(args.frame_width, 1.2) * hw / 2
        x0 = _clamp(min(x0, hx0), cx - half, cx - 0.6 * hw)
        x1 = _clamp(max(x1, hx1), cx + 0.6 * hw, cx + half)
        y0 = _clamp(min(y0, hy0), hy0 - 0.5 * hh, hy0)
        y1 = _clamp(max(y1, hy1), hy1 + 0.6 * hh, hy1 + max(args.body, 0.6) * hh)

    pad_x = (x1 - x0) * args.pad
    pad_y = (y1 - y0) * args.pad
    box = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)

    if args.aspect:
        box = apply_aspect(box, args.aspect, (W, H), anchor_x)

    x0, y0, x1, y1 = box
    return (
        int(round(max(0.0, x0))),
        int(round(max(0.0, y0))),
        int(round(min(float(W), x1))),
        int(round(min(float(H), y1))),
    )
