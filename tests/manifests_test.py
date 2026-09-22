"""Tests for `src.manifests`, the artefacts that join the stages together."""

import numpy as np
import pytest

from src import manifests


class TestJson:
    """Reading and writing a stage's manifest."""

    def test_round_trips_a_manifest(self, tmp_path):
        """What one stage writes is what the next one reads."""
        path = tmp_path / manifests.CROPS
        manifests.write(path, [{"file": "a_001.jpg", "head": None}])
        assert manifests.read(path) == [{"file": "a_001.jpg", "head": None}]

    def test_a_missing_manifest_says_what_to_do(self, tmp_path):
        """Running a stage out of order is the likeliest way to get here."""
        with pytest.raises(FileNotFoundError, match="run the stage"):
            manifests.read(tmp_path / manifests.CROPS)


class TestVectors:
    """The identity vectors that travel beside a manifest."""

    def test_round_trips_the_rows_in_order(self, tmp_path):
        """Row order is the only thing tying a vector to its crop."""
        path = tmp_path / manifests.VECTORS
        manifests.write_vectors(path, [[1.0, 0.0], [0.0, 1.0]])
        assert np.array_equal(manifests.read_vectors(path, 2), [[1.0, 0.0], [0.0, 1.0]])

    def test_stores_float32(self, tmp_path):
        """Half the size of float64, and the model produces float32 anyway."""
        path = tmp_path / manifests.VECTORS
        manifests.write_vectors(path, np.ones((2, 4), dtype=np.float64))
        assert manifests.read_vectors(path, 2).dtype == np.float32

    def test_an_empty_run_is_not_a_failure(self, tmp_path):
        """A folder of photos with nobody in it still writes both artefacts."""
        path = tmp_path / manifests.VECTORS
        manifests.write_vectors(path, [])
        assert len(manifests.read_vectors(path, 0)) == 0

    def test_a_count_that_disagrees_is_an_error(self, tmp_path):
        """Reading on would silently attach these rows to the wrong people."""
        path = tmp_path / manifests.VECTORS
        manifests.write_vectors(path, [[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="re-run the stage"):
            manifests.read_vectors(path, 3)

    def test_missing_vectors_say_what_to_do(self, tmp_path):
        """Stage 3 cannot work without them, so it has to ask for a re-run."""
        with pytest.raises(FileNotFoundError, match="run the stage"):
            manifests.read_vectors(tmp_path / manifests.VECTORS, 2)
