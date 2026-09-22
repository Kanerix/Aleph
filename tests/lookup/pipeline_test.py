"""Tests for `src.lookup.pipeline`."""

import numpy as np
import pytest

from src.lookup import pipeline


def profile(number: int) -> dict:
    """One profile as stage 2 would have written it."""
    return {
        "id": f"p{number:03d}",
        "appearances": 2,
        "photos": ["data/stand.jpg"],
        "crops": ["a.jpg", "b.jpg"],
    }


def vectors(*angles: float) -> np.ndarray:
    """Unit vectors on a circle, so cosine similarity is the cosine of the gap."""
    return np.array([[np.cos(a), np.sin(a)] for a in angles])


class TestIdentify:
    """Naming a profile after the closest face in the gallery."""

    def test_names_a_profile_that_clears_the_threshold(self):
        """The whole point of the stage."""
        [entry] = pipeline.identify(
            [profile(1)], vectors(0.0), ["anna", "bo"], vectors(0.05, 2.0), 0.9
        )
        assert entry["match"] == "anna"
        assert entry["score"] > 0.9

    def test_leaves_a_distant_profile_unnamed(self):
        """A stranger must come back as nobody rather than as the nearest face."""
        [entry] = pipeline.identify(
            [profile(1)], vectors(0.0), ["anna"], vectors(1.5), 0.9
        )
        assert entry["match"] is None

    def test_records_the_score_even_when_it_falls_short(self):
        """A threshold set slightly too high should be visible in the output."""
        [entry] = pipeline.identify(
            [profile(1)], vectors(0.0), ["anna"], vectors(0.3), 0.99
        )
        assert entry["match"] is None
        assert entry["score"] == pytest.approx(0.955, abs=0.001)

    def test_reports_the_runner_up(self):
        """Two close references mean the gallery cannot tell them apart."""
        [entry] = pipeline.identify(
            [profile(1)], vectors(0.0), ["anna", "bo"], vectors(0.1, 0.2), 0.9
        )
        assert entry["runner_up"]["name"] == "bo"

    def test_a_gallery_of_one_has_no_runner_up(self):
        """Nothing is invented to fill the second place."""
        [entry] = pipeline.identify(
            [profile(1)], vectors(0.0), ["anna"], vectors(0.0), 0.9
        )
        assert entry["runner_up"] is None

    def test_carries_the_profile_across(self):
        """The entry has to be readable without profiles.json beside it."""
        [entry] = pipeline.identify(
            [profile(2)], vectors(0.0), ["anna"], vectors(0.0), 0.9
        )
        assert entry["id"] == "p002"
        assert entry["appearances"] == 2
        assert entry["photos"] == ["data/stand.jpg"]

    def test_every_profile_gets_an_entry(self):
        """The manifest is the join to the next stage, so nothing is dropped."""
        given = [profile(1), profile(2)]
        entries = pipeline.identify(
            given, vectors(0.0, 3.0), ["anna"], vectors(0.0), 0.9
        )
        assert [entry["id"] for entry in entries] == ["p001", "p002"]
