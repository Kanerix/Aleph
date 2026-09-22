"""Read the reference faces that profiles are matched against.

The gallery is a folder you put together by hand: one photo per person, named
after them, or a folder per person holding several photos of them. Several
photos of one person are averaged into a single vector.

Nothing is downloaded, scraped or searched for. A profile can only ever be
matched to someone whose photo was deliberately placed in this folder, which is
the whole difference between naming people you already know and identifying
strangers.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..extract.images import IMAGE_EXTS
from ..faces import average


def _open(path: Path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def by_person(folder: Path) -> dict[str, list[Path]]:
    """Group the gallery's photos under the name of the person they show.

    A photo sitting directly in the gallery is named by its filename, and a
    photo in a subfolder takes the subfolder's name, so both ways of laying the
    folder out work without a flag to say which is in use.
    """
    people: dict[str, list[Path]] = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.stem if path.parent == folder else path.parent.name
        people.setdefault(name, []).append(path)
    return people


def read(folder: Path, reader) -> tuple[list[str], np.ndarray]:
    """Turn the gallery into one identity vector per person.

    A reference photo is taken to be of the person it is filed under, so the
    surest face in it wins and anyone else in the frame is ignored. People
    whose photos hold no face at all are reported and left out of the gallery
    rather than matched against with nothing.
    """
    names, vectors = [], []
    for name, paths in by_person(folder).items():
        faces = []
        for path in paths:
            try:
                face = reader.read(_open(path), None)
            except OSError as err:
                print(f"  ! unreadable: {path} ({err})", file=sys.stderr)
                continue
            if face is not None:
                faces.append(face.vector)

        if not faces:
            print(f"  ! no face in the reference photos of {name}", file=sys.stderr)
            continue
        names.append(name)
        vectors.append(average(faces))
        print(f"  {name}: {len(faces)}/{len(paths)} photos")

    return names, np.array(vectors)
