"""Compare identity vectors with the gallery's reference faces.

ArcFace vectors are already L2-normalised, so a dot product is the cosine
similarity and the comparison is one matrix multiply. Nothing here touches a
model or the disk, which is what makes the thresholds testable.
"""

import numpy as np

NEIGHBOURS = 2


def best(vector: np.ndarray, references: np.ndarray) -> list[tuple[int, float]]:
    """The closest references to `vector`, best first, as (row, score) pairs.

    The runner-up comes back with the winner because a high score means little
    on its own: two references that both sit at 0.5 say the gallery cannot tell
    those people apart, which is worth seeing before believing the match.
    """
    scores = references @ vector
    order = np.argsort(-scores)[:NEIGHBOURS]
    return [(int(row), float(scores[row])) for row in order]
