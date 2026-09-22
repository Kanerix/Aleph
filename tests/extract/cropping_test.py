"""Tests for `src.cropping`."""

import argparse

import pytest

from src.extract.cropping import apply_aspect, crop_region, parse_aspect
from src.extract.detections import Person, face_geometry


class TestParseAspect:
    """`--aspect` parsing."""

    def test_ratio_notation(self):
        """Colon notation is read as width over height."""
        assert parse_aspect("3:4") == pytest.approx(0.75)

    def test_decimal_notation(self):
        """A bare number is used as the ratio directly."""
        assert parse_aspect("1.5") == pytest.approx(1.5)

    def test_empty_means_no_ratio(self):
        """No value leaves the crop aspect untouched."""
        assert parse_aspect(None) is None
        assert parse_aspect("") is None

    @pytest.mark.parametrize("value", ["3:0", "wide", "-1", "0"])
    def test_rejects_nonsense(self, value: str):
        """Unusable ratios are reported as argparse errors."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_aspect(value)


class TestApplyAspect:
    """Growing a box to a fixed ratio."""

    def test_reaches_the_requested_ratio(self):
        """A square box grows into the requested portrait ratio."""
        x0, y0, x1, y1 = apply_aspect((0, 0, 100, 100), 0.75, (1000, 1000))

        assert (x1 - x0) / (y1 - y0) == pytest.approx(0.75)

    def test_stays_inside_the_image(self):
        """The box never grows past the edges of the source image."""
        x0, y0, x1, y1 = apply_aspect((900, 900, 1000, 1000), 0.75, (1000, 1000))

        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= 1000
        assert y1 <= 1000


class TestCropRegion:
    """Framing of the final crop."""

    def test_head_is_near_the_top_of_the_crop(self, args, keypoints):
        """The crop keeps the head high and the torso below it."""
        face = face_geometry(keypoints(), kp_conf=args.kp_conf)
        person = Person(box=(60, 60, 140, 400), conf=0.9, face=face)

        _, y0, _, y1 = crop_region(person, 1000, 1000, args)
        head_y = face["center"][1]

        assert y0 < head_y < y1
        assert (head_y - y0) < (y1 - head_y)

    def test_wide_body_box_cannot_widen_the_crop(self, args, keypoints):
        """A raised flag or arm must not drag the crop across the row."""
        face = face_geometry(keypoints(), kp_conf=args.kp_conf)
        narrow = Person(box=(60, 60, 140, 400), conf=0.9, face=face)
        wide = Person(box=(0, 60, 900, 400), conf=0.9, face=face)

        narrow_box = crop_region(narrow, 1000, 1000, args)
        wide_box = crop_region(wide, 1000, 1000, args)

        limit = args.frame_width * face["width"] * (1 + 2 * args.pad)
        assert (wide_box[2] - wide_box[0]) <= limit
        assert (wide_box[2] - wide_box[0]) >= (narrow_box[2] - narrow_box[0])

    def test_crop_is_clamped_to_the_image(self, args):
        """Crops of people at the edge of the frame stay inside the photo."""
        person = Person(box=(-20, -20, 80, 300), conf=0.9)

        x0, y0, x1, y1 = crop_region(person, 200, 400, args)

        assert (x0, y0) == (0, 0)
        assert x1 <= 200
        assert y1 <= 400

    def test_aspect_flag_is_honoured(self, cli_args, keypoints):
        """`--aspect` shapes the crop without leaving the image."""
        args = cli_args("--aspect", "3:4")
        face = face_geometry(keypoints(), kp_conf=args.kp_conf)
        person = Person(box=(60, 60, 140, 400), conf=0.9, face=face)

        x0, y0, x1, y1 = crop_region(person, 1000, 1000, args)

        assert (x1 - x0) / (y1 - y0) == pytest.approx(0.75, abs=0.01)
