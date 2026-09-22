"""Name the dominant colours of a region without using a model.

Hue, saturation and value are far easier to name than raw RGB: anything with
little saturation is black, grey or white depending on brightness, and
everything else falls into a hue band.
"""

import colorsys

GREY_SATURATION = 0.20
BLACK_VALUE = 0.22
GREY_VALUE = 0.72
DARK_VALUE = 0.38
PALE_SATURATION = 0.45
SAMPLE_SIZE = 48

# Upper bound of each hue band, in the 0..1 hue circle.
HUE_BANDS = (
    (0.045, "red"),
    (0.11, "orange"),
    (0.19, "yellow"),
    (0.45, "green"),
    (0.52, "turquoise"),
    (0.72, "blue"),
    (0.83, "purple"),
    (0.95, "pink"),
    (1.01, "red"),
)


def name_colour(rgb: tuple[int, int, int]) -> str:
    """Name one RGB triple, for example "dark blue" or "white"."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    if saturation < GREY_SATURATION:
        if value < BLACK_VALUE:
            return "black"
        return "grey" if value < GREY_VALUE else "white"

    # The last band ends above 1.0, so there is always a match.
    name = next(name for upper, name in HUE_BANDS if hue < upper)
    if value < DARK_VALUE:
        return f"dark {name}"
    if saturation < PALE_SATURATION:
        return f"pale {name}"
    return name


def dominant_colours(image, count: int = 2, min_share: float = 0.15) -> list[str]:
    """The colours that cover at least `min_share` of the region, most first."""
    sample = image.convert("RGB").resize((SAMPLE_SIZE, SAMPLE_SIZE))
    total = SAMPLE_SIZE * SAMPLE_SIZE

    tally: dict[str, int] = {}
    # An image cannot hold more distinct colours than it has pixels, so asking
    # for one entry per pixel means getcolors never gives up and returns None.
    for pixels, rgb in sample.getcolors(total):
        name = name_colour(rgb)
        tally[name] = tally.get(name, 0) + pixels

    ranked = sorted(tally.items(), key=lambda item: item[1], reverse=True)
    return [name for name, n in ranked[:count] if n / total >= min_share]
