"""The JSON artefacts that join one pipeline stage to the next.

Each stage writes a manifest into its output folder, and the stage after it
reads that manifest rather than guessing from filenames.

The identity vectors travel beside the manifest in a `.npy` file instead of in
it. A 512-d vector is 14 KB and 532 lines of indented JSON, which would bury
the fields a person actually reads in the manifest and make it useless to
`jaq`. The rows line up with the manifest's entries, so the two files are read
and written together.
"""

import json
from pathlib import Path

import numpy as np

CROPS = "manifest.json"
PROFILES = "profiles.json"
IDENTITIES = "identities.json"
VECTORS = "vectors.npy"


def write(path: Path, data) -> None:
    """Write `data` as indented JSON."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read(path: Path):
    """Read a manifest written by an earlier stage."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run the stage that produces it first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_vectors(path: Path, vectors) -> None:
    """Write one identity vector per manifest entry, in the same order."""
    np.save(path, np.asarray(vectors, dtype=np.float32))


def read_vectors(path: Path, expected: int) -> np.ndarray:
    """Read the identity vectors saved beside a manifest.

    Nothing inside the file says which crop a row belongs to, so a count that
    disagrees with the manifest is an error rather than something to work
    around: the rows would otherwise be read as the wrong people.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run the stage that produces it first"
        )
    vectors = np.load(path)
    if len(vectors) != expected:
        raise ValueError(
            f"{path} holds {len(vectors)} vectors but its manifest lists "
            f"{expected} entries - re-run the stage that wrote them"
        )
    return vectors
