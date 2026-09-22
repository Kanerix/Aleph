"""Tests for `src.profiles.clustering`."""

import numpy as np

from src.profiles.clustering import cluster


def vectors(*angles: float) -> np.ndarray:
    """Unit vectors on a circle, so cosine similarity is the cosine of the gap."""
    return np.array([[np.cos(a), np.sin(a)] for a in angles])


class TestCluster:
    """Single-linkage grouping with the one-person-per-photo constraint."""

    def test_no_crops(self):
        """An empty run produces no profiles rather than failing."""
        assert cluster(np.zeros((0, 2)), [], 0.9) == []

    def test_keeps_distant_crops_apart(self):
        """Two faces that look nothing alike stay separate."""
        assert cluster(vectors(0.0, 1.5), ["a.jpg", "b.jpg"], 0.9) == [[0], [1]]

    def test_merges_close_crops_from_different_photos(self):
        """The same person photographed twice ends up in one profile."""
        assert cluster(vectors(0.0, 0.05), ["a.jpg", "b.jpg"], 0.9) == [[0, 1]]

    def test_never_merges_two_crops_from_one_photo(self):
        """Stage 1 already de-duplicated, so one photo means two people."""
        assert cluster(vectors(0.0, 0.0), ["a.jpg", "a.jpg"], 0.9) == [[0], [1]]

    def test_threshold_decides_the_merge(self):
        """The same pair merges or not depending on the threshold."""
        sources = ["a.jpg", "b.jpg"]
        assert cluster(vectors(0.0, 0.5), sources, 0.99) == [[0], [1]]
        assert cluster(vectors(0.0, 0.5), sources, 0.8) == [[0, 1]]

    def test_a_shared_photo_blocks_a_whole_chain(self):
        """Rows 0 and 2 share a photo, so they cannot both join row 1."""
        groups = cluster(vectors(0.0, 0.02, 0.04), ["a.jpg", "b.jpg", "a.jpg"], 0.9)
        assert sorted(len(members) for members in groups) == [1, 2]
        assert sorted(i for members in groups for i in members) == [0, 1, 2]

    def test_orders_the_largest_group_first(self):
        """The most seen person is reported before the one-offs."""
        groups = cluster(vectors(0.0, 0.02, 3.0), ["a.jpg", "b.jpg", "c.jpg"], 0.9)
        assert groups == [[0, 1], [2]]

    def test_every_crop_lands_in_exactly_one_group(self):
        """No crop is dropped and none is counted twice."""
        sources = [f"photo{i}.jpg" for i in range(6)]
        groups = cluster(vectors(0.0, 0.01, 1.0, 1.01, 2.0, 3.0), sources, 0.95)
        assert sorted(i for group in groups for i in group) == list(range(6))
