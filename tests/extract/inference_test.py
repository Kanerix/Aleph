"""Tests for `src.inference`."""

from src.device import pick_device
from src.extract.inference import plan_tiles, tile_origins


class TestPickDevice:
    """Device selection."""

    def test_explicit_preference_is_returned_as_is(self):
        """An explicit `--device` is never second-guessed."""
        assert pick_device("cpu") == "cpu"
        assert pick_device("0") == "0"


class TestTiling:
    """Sliced inference windows."""

    def test_small_image_is_a_single_tile(self):
        """An image smaller than one tile is not sliced."""
        assert tile_origins(500, 768, 576) == [0]

    def test_last_tile_is_flush_with_the_edge(self):
        """The final origin is pulled back so the edge is still covered."""
        origins = tile_origins(1000, 768, 576)

        assert origins[0] == 0
        assert origins[-1] == 1000 - 768

    def test_tiles_cover_the_whole_image(self):
        """Every pixel of the image falls inside at least one tile."""
        W, H = 2000, 1500
        tiles = plan_tiles(W, H, tile=768, overlap=0.25)

        assert tiles
        assert max(t[2] for t in tiles) == W
        assert max(t[3] for t in tiles) == H
        assert all(0 <= t[0] < t[2] <= W and 0 <= t[1] < t[3] <= H for t in tiles)

    def test_tiles_overlap(self):
        """Neighbouring tiles share a margin so faces on a seam survive."""
        tiles = plan_tiles(2000, 2000, tile=768, overlap=0.25)
        xs = sorted({t[0] for t in tiles})

        assert xs[1] < 768
