"""Map the boxes stage 1 recorded onto the crop file it saved.

Stage 1 records the crop box and the head box in source-photo pixels, and it
may have upscaled the saved crop with `--min-out`. A box therefore has to be
shifted and rescaled before it means anything in the saved image.
"""

BODY_FALLBACK_TOP = 0.4


def head_box(image, entry):
    """The head in the saved crop's own pixels, or None if stage 1 found none."""
    if not entry.get("head"):
        return None
    cx0, cy0, cx1, _ = entry["crop"]
    scale = image.width / max(1.0, cx1 - cx0)
    x0, y0, x1, y1 = entry["head"]
    return (
        (x0 - cx0) * scale,
        (y0 - cy0) * scale,
        (x1 - cx0) * scale,
        (y1 - cy0) * scale,
    )


def body(image, entry):
    """Everything below the head, which is where the clothing is."""
    W, H = image.size
    box = head_box(image, entry)
    top = box[3] if box else H * BODY_FALLBACK_TOP
    top = min(max(round(top), 0), H - 1)
    return image.crop((0, top, W, H))
