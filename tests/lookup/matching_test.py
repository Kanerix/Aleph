"""Tests for `src.lookup.matching`."""

import numpy as np

from src.lookup.matching import best


def vectors(*angles: float) -> np.ndarray:
    """Unit vectors on a circle, so cosine similarity is the cosine of the gap."""
    return np.array([[np.cos(a), np.sin(a)] for a in angles])


class TestBest:
    """Ranking the gallery against one identity vector."""

    def test_finds_the_closest_reference(self):
        """The point of the stage: the nearest face wins."""
        row, score = best(vectors(1.0)[0], vectors(0.0, 0.99, 2.0))[0]
        assert row == 1
        assert score > 0.99

    def test_reports_the_runner_up_as_well(self):
        """An ambiguous match is only visible next to the second best."""
        ranked = best(vectors(0.0)[0], vectors(1.0, 0.0, 2.0))
        assert [row for row, _ in ranked] == [1, 0]

    def test_a_gallery_of_one_has_no_runner_up(self):
        """Nothing is invented to fill the second place."""
        assert len(best(vectors(0.0)[0], vectors(0.5))) == 1

    def test_scores_are_plain_floats(self):
        """The scores go straight into JSON, so numpy types would not do."""
        _, score = best(vectors(0.0)[0], vectors(0.0))[0]
        assert isinstance(score, float)
