"""Tests for `src.images`."""

from pathlib import Path

from PIL import Image

from src.extract.images import _trim_black_borders, iter_images, save_crop, unique_stems


class TestIterImages:
    """Input discovery."""

    def test_collects_supported_files_only(self, tmp_path):
        """Folders are walked recursively and non-images are ignored."""
        (tmp_path / "a.jpg").touch()
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "b.ARW").touch()
        (tmp_path / "notes.txt").touch()

        found = iter_images([str(tmp_path)])

        assert [p.name for p in found] == ["a.jpg", "b.ARW"]

    def test_skips_the_output_folder(self, tmp_path):
        """A second run does not re-crop the crops of the first one."""
        (tmp_path / "a.jpg").touch()
        out = tmp_path / "faces"
        out.mkdir()
        (out / "a_001.jpg").touch()

        found = iter_images([str(tmp_path)], exclude=out)

        assert [p.name for p in found] == ["a.jpg"]

    def test_missing_input_is_not_fatal(self, tmp_path):
        """A path that matches nothing is reported, not raised."""
        assert iter_images([str(tmp_path / "nope.jpg")]) == []


class TestUniqueStems:
    """Output names for photos that share a filename."""

    def test_distinct_names_are_left_alone(self):
        """Nothing is renamed when the filenames already differ."""
        paths = [Path("a/one.jpg"), Path("b/two.jpg")]

        assert list(unique_stems(paths).values()) == ["one", "two"]

    def test_same_name_on_two_cards(self):
        """The second IMG_0001 gets its own name instead of overwriting."""
        paths = [Path("100/IMG_0001.ARW"), Path("101/IMG_0001.ARW")]

        stems = unique_stems(paths)

        assert stems[paths[0]] == "IMG_0001"
        assert stems[paths[1]] == "IMG_0001-2"

    def test_names_are_stable_across_runs(self):
        """The same input list always produces the same names."""
        paths = [Path("a/x.jpg"), Path("b/x.jpg"), Path("c/x.jpg")]

        assert unique_stems(paths) == unique_stems(paths)


class TestSaveCrop:
    """Writing one crop to disk."""

    def test_min_out_upscales_small_crops(self, cli_args, tmp_path):
        """`--min-out` guarantees a minimum short side on every crop."""
        args = cli_args("--min-out", "256")
        crop = Image.new("RGB", (100, 150), "white")
        dest = tmp_path / "crop.jpg"

        save_crop(crop, dest, args)

        assert min(Image.open(dest).size) >= 256

    def test_quality_applies_to_webp(self, cli_args, tmp_path):
        """`--quality` is not silently dropped for WebP output."""
        crop = Image.radial_gradient("L").convert("RGB")
        low = tmp_path / "low.webp"
        high = tmp_path / "high.webp"

        save_crop(crop, low, cli_args("--quality", "10"))
        save_crop(crop, high, cli_args("--quality", "95"))

        assert low.stat().st_size < high.stat().st_size


class TestTrimBlackBorders:
    """Unlit sensor margins left by LibRaw."""

    def test_trims_unlit_margins(self):
        """A black band down one edge is cropped away."""
        image = Image.new("RGB", (100, 100), "white")
        image.paste(Image.new("RGB", (10, 100), "black"), (90, 0))

        assert _trim_black_borders(image).size == (90, 100)

    def test_keeps_genuinely_dark_photos(self):
        """A mostly black frame is a night shot, not a border to trim."""
        image = Image.new("RGB", (100, 100), "black")
        image.paste(Image.new("RGB", (20, 20), "white"), (0, 0))

        assert _trim_black_borders(image).size == (100, 100)
