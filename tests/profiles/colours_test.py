"""Tests for `src.profiles.colours`."""

from PIL import Image

from src.profiles.colours import dominant_colours, name_colour


class TestNameColour:
    """Naming a single RGB triple."""

    def test_unsaturated_colours_are_named_by_brightness(self):
        """Without saturation there is no hue to report, only lightness."""
        assert name_colour((250, 250, 250)) == "white"
        assert name_colour((128, 128, 128)) == "grey"
        assert name_colour((12, 12, 12)) == "black"

    def test_hue_bands(self):
        """Saturated colours are named after the hue band they fall in."""
        assert name_colour((220, 20, 20)) == "red"
        assert name_colour((40, 90, 230)) == "blue"
        assert name_colour((240, 220, 40)) == "yellow"

    def test_dark_shades_keep_the_hue(self):
        """A navy jacket is reported as dark blue, not as black."""
        assert name_colour((10, 20, 90)) == "dark blue"


class TestDominantColours:
    """Naming the colours of a region."""

    def test_reports_the_largest_area_first(self):
        """The colour covering most of the region leads the list."""
        image = Image.new("RGB", (100, 100), (250, 250, 250))
        image.paste(Image.new("RGB", (100, 40), (10, 20, 90)), (0, 60))

        assert dominant_colours(image) == ["white", "dark blue"]

    def test_ignores_slivers(self):
        """A colour covering a few pixels is not part of the description."""
        image = Image.new("RGB", (100, 100), (220, 20, 20))
        image.paste(Image.new("RGB", (100, 4), (240, 220, 40)), (0, 0))

        assert dominant_colours(image) == ["red"]
