"""Tests for `src.profiles.regions`."""

from PIL import Image

from src.profiles import regions


def entry():
    """A manifest entry whose crop is 200x200 in the source photo."""
    return {
        "file": "photo_001.jpg",
        "source": "data/photo.jpg",
        "crop": [100, 200, 300, 400],
        "head": [150, 210, 190, 250],
    }


def image(size: int):
    """A square crop file of `size` pixels a side."""
    return Image.new("RGB", (size, size))


class TestHeadBox:
    """Putting the recorded head box into the saved crop's own pixels."""

    def test_shifts_the_box_by_the_crop_origin(self):
        """The head sits at (50, 10) once the crop origin is subtracted."""
        assert regions.head_box(image(200), entry()) == (50, 10, 90, 50)

    def test_rescales_when_the_crop_was_upscaled(self):
        """`--min-out` doubled the saved file, so the box doubles with it."""
        assert regions.head_box(image(400), entry()) == (100, 20, 180, 100)

    def test_no_head_recorded(self):
        """Stage 1 saw no facial keypoints, so there is no box to map."""
        assert regions.head_box(image(200), entry() | {"head": None}) is None


class TestBody:
    """Cutting the clothing out of a saved crop."""

    def test_starts_at_the_bottom_of_the_head(self):
        """Everything under the chin counts as body."""
        assert regions.body(image(200), entry()).size == (200, 150)

    def test_rescales_when_the_crop_was_upscaled(self):
        """The head line moves with the saved file's scale."""
        assert regions.body(image(400), entry()).size == (400, 300)

    def test_falls_back_to_a_fixed_line_without_a_head(self):
        """With no head box the lower 60% of the crop is used instead."""
        assert regions.body(image(200), entry() | {"head": None}).size == (200, 120)

    def test_a_head_below_the_image_still_leaves_a_body(self):
        """A bad head box must not produce an empty region."""
        body = regions.body(image(200), entry() | {"head": [0, 0, 400, 9999]})
        assert body.size == (200, 1)
