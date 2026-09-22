"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from src.main import build_parser


@pytest.fixture
def cli_args():
    """Return a factory for `extract` defaults with extra flags applied."""

    def build(*extra: str):
        """Parse `extra` on top of the defaults for a single photo."""
        return build_parser().parse_args(["extract", "photo.jpg", *extra])

    return build


@pytest.fixture
def args(cli_args):
    """`extract` defaults, as the stage would see them for a single photo."""
    return cli_args()


@pytest.fixture
def profile_args():
    """Return a factory for `profile` defaults with extra flags applied."""

    def build(*extra: str):
        """Parse `extra` on top of the defaults for a crops folder."""
        return build_parser().parse_args(["profile", "faces", *extra])

    return build


@pytest.fixture
def keypoints():
    """Return a factory for a COCO-17 keypoint array of one person."""

    def build(facing_camera: bool = True) -> np.ndarray:
        """Place nose, eyes, ears and shoulders around x=100, y=100."""
        kps = np.zeros((17, 3), dtype=float)
        points = {
            0: (100, 100),  # nose
            1: (110, 95),  # left eye
            2: (90, 95),  # right eye
            3: (120, 98),  # left ear
            4: (80, 98),  # right ear
            5: (130, 140),  # left shoulder
            6: (70, 140),  # right shoulder
        }
        for index, (x, y) in points.items():
            # Anatomical left sits on the image right only when facing us.
            kps[index] = (x if facing_camera else 200 - x, y, 0.9)
        return kps

    return build
