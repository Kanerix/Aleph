"""The JSON artefacts that join one pipeline stage to the next.

Each stage writes a manifest into its output folder, and the stage after it
reads that manifest rather than guessing from filenames.
"""

import json
from pathlib import Path

CROPS = "manifest.json"
PROFILES = "profiles.json"


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
