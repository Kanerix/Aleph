"""Identity vectors from a face recognition model.

Both stages need this. Stage 1 uses it to throw away the hands, banners and
stretches of fence that the pose model mistook for people, and stage 2 uses
the vectors to decide which crops show the same person.

CLIP describes what a photo looks like, so it puts two supporters in black caps
behind the same fence next to each other whether or not they are the same man.
ArcFace is trained on the question stage 2 actually asks, so identity comes
from here and only the keywords come from CLIP.

`insightface` and `onnxruntime` are imported inside the class so that `--help`
stays instant.
"""

import math
from typing import NamedTuple

import numpy as np

MODEL = "buffalo_l"
DETECTION_SIZE = 640

PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "CoreMLExecutionProvider")
GENDER = {"M": "man", "F": "woman"}


class Face(NamedTuple):
    """One recognised face."""

    vector: np.ndarray
    gender: str | None
    score: float


def providers(available, device: str) -> list[str]:
    """Accelerated providers first, with CPU always there to fall back on.

    onnxruntime has its own idea of accelerators, so `--device` only gets a say
    in whether one is used at all.
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    chosen = [name for name in PREFERRED_PROVIDERS if name in available]
    return [*chosen, "CPUExecutionProvider"]


def _centre_distance(box, other) -> float:
    ax = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bx = (other[0] + other[2]) / 2, (other[1] + other[3]) / 2
    return math.hypot(ax[0] - bx[0], ax[1] - bx[1])


def nearest(found, head_box):
    """The detected face closest to where stage 1 said the head was.

    A crowd crop often catches a neighbour's face at the edge, so the manifest
    box decides which of them the crop is actually of. Without a box, which
    happens when stage 1 saw no facial keypoints, the surest detection wins.
    """
    if not found:
        return None
    if head_box is None:
        return max(found, key=lambda face: face.det_score)
    return min(found, key=lambda face: _centre_distance(face.bbox, head_box))


class FaceReader:
    """Finds the face in a crop and turns it into an identity vector."""

    def __init__(
        self, device: str, model: str = MODEL, detection_size: int = DETECTION_SIZE
    ):
        """Load `model`, downloading it to ~/.insightface on first use."""
        import onnxruntime
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=model,
            providers=providers(onnxruntime.get_available_providers(), device),
        )
        # ctx_id only picks between the providers above and forcing CPU.
        self.app.prepare(ctx_id=0, det_size=(detection_size, detection_size))

    def read(self, image, head_box) -> Face | None:
        """Read the face of the person this crop is of, if there is one."""
        found = nearest(self.app.get(np.asarray(image)[:, :, ::-1]), head_box)
        if found is None:
            return None
        return Face(
            vector=found.normed_embedding,
            gender=GENDER.get(found.sex),
            score=float(found.det_score),
        )
