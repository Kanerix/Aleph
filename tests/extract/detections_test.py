"""Tests for `src.detections`."""

import pytest

from src.extract.detections import (
    Person,
    _iou_pair,
    face_geometry,
    merge_detections,
    orientation_score,
)


class TestOrientation:
    """Front/back detection from left/right keypoint order."""

    def test_facing_camera_scores_positive(self, keypoints):
        """Eyes, ears and shoulders all vote that the face is visible."""
        score, votes = orientation_score(keypoints(), kp_conf=0.35)

        assert score > 0
        assert votes == 3

    def test_turned_away_scores_negative(self, keypoints):
        """A mirrored pose is the back of a head, not a face."""
        score, votes = orientation_score(keypoints(facing_camera=False), kp_conf=0.35)

        assert score < 0
        assert votes == 3


class TestFaceGeometry:
    """Head position and size derived from facial landmarks."""

    def test_no_keypoints(self):
        """A model without keypoints yields no face at all."""
        assert face_geometry(None, kp_conf=0.35) == {}

    def test_low_confidence_keypoints(self, keypoints):
        """Landmarks below the threshold do not count as a face."""
        kps = keypoints()
        kps[:, 2] = 0.1

        assert face_geometry(kps, kp_conf=0.35) == {}

    def test_head_box_wraps_the_landmarks(self, keypoints):
        """The head box is centred on the landmarks and taller than it is wide."""
        face = face_geometry(keypoints(), kp_conf=0.35)

        x0, y0, x1, y1 = face["box"]
        cx, cy = face["center"]

        assert cx == pytest.approx(100)
        assert x0 < cx < x1
        assert y0 < cy < y1
        assert face["width"] == pytest.approx(x1 - x0)
        assert face["height"] > face["width"]
        assert face["n_kps"] == 5
        assert face["facing"] > 0


class TestMergeDetections:
    """De-duplication of the full-frame and tiled passes."""

    @staticmethod
    def person(box, conf, center=None, width=40.0):
        """Build a `Person`, optionally with a located head."""
        face = {}
        if center is not None:
            face = {"center": center, "width": width, "height": width * 1.5}
        return Person(box=box, conf=conf, face=face)

    def test_same_head_seen_twice_is_one_person(self):
        """Two tiles that found the same head collapse into one detection."""
        people = [
            self.person((0, 0, 100, 200), 0.9, center=(50, 30)),
            self.person((4, 2, 104, 202), 0.8, center=(52, 31)),
        ]

        kept = merge_detections(people, iou_thr=0.55)

        assert len(kept) == 1
        assert kept[0].conf == pytest.approx(0.9)

    def test_overlapping_neighbours_are_kept(self):
        """Fans packed together overlap heavily but have separate heads."""
        people = [
            self.person((0, 0, 100, 200), 0.9, center=(50, 30)),
            self.person((40, 0, 140, 200), 0.8, center=(140, 30)),
        ]

        assert len(merge_detections(people, iou_thr=0.55)) == 2

    def test_faceless_duplicates_fall_back_to_iou(self):
        """Without heads to compare, box overlap decides."""
        people = [
            self.person((0, 0, 100, 200), 0.9),
            self.person((2, 2, 102, 202), 0.7),
        ]

        assert len(merge_detections(people, iou_thr=0.55)) == 1

    def test_distant_person_inside_a_foreground_box(self):
        """A small box nested in a large one is someone further away."""
        people = [
            self.person((0, 0, 400, 800), 0.9),
            self.person((10, 10, 60, 110), 0.6),
        ]

        assert len(merge_detections(people, iou_thr=0.55)) == 2


class TestIouPair:
    """Box overlap maths."""

    def test_identical_boxes(self):
        """A box overlaps itself completely."""
        assert _iou_pair((0, 0, 10, 10), (0, 0, 10, 10)) == (1.0, 1.0)

    def test_disjoint_boxes(self):
        """Boxes that do not touch have no overlap."""
        assert _iou_pair((0, 0, 10, 10), (20, 20, 30, 30)) == (0.0, 0.0)

    def test_nested_box_fills_the_smaller_area(self):
        """Intersection-over-smaller is 1.0 when one box contains the other."""
        iou, iom = _iou_pair((0, 0, 100, 100), (10, 10, 20, 20))

        assert iom == pytest.approx(1.0)
        assert iou < 0.1
