"""Describe every crop, then group the crops into profiles."""

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..faces import Face, average
from . import attributes, colours, regions
from .clustering import cluster

MAX_COLOURS = 3


def _open(path: Path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _describe(entry, image, face, similarity) -> dict:
    labels = attributes.read(similarity)
    if face.gender:
        # The face model gives no probability of its own, so the detection
        # score stands in as the weight this vote carries.
        labels["gender"] = (face.gender, face.score)
    return {
        "file": entry["file"],
        "source": entry["source"],
        "labels": labels,
        "colours": colours.dominant_colours(regions.body(image, entry)),
    }


def read_faces(entries, crops_dir: Path, reader) -> list:
    """Read the face in every crop, for a folder that has no saved vectors.

    `--no-verify` leaves stage 1 with no face model, so the reading it would
    otherwise have done falls to this stage. The crops are opened again in
    `describe` for CLIP, which costs a second JPEG decode on the rare path and
    keeps both ways into `describe` the same shape.
    """
    faces = []
    for entry in entries:
        image = _open(crops_dir / entry["file"])
        faces.append(reader.read(image, regions.head_box(image, entry)))
    return faces


def saved_faces(entries, vectors) -> list:
    """Rebuild the faces stage 1 read from its manifest and its vectors.

    Stage 1 read all of these faces already when it verified the crops, and its
    vectors come from better pixels than a second pass would: it reads the crop
    before `--min-out` upscales it and before it is saved as JPEG.
    """
    return [
        Face(vector=vector, gender=entry["gender"], score=entry["face_score"])
        for entry, vector in zip(entries, vectors, strict=True)
    ]


def describe(entries, crops_dir: Path, encoder, faces, batch: int):
    """Tag every crop that holds a face.

    Crops with no face are dropped, which is how the hands, banners and fence
    panels that stage 1 mistook for people leave the pipeline. Returns the
    identity vectors and one description per surviving crop, in the order the
    entries were given.
    """
    prompt_vectors = encoder.texts(attributes.prompts())
    vectors, described = [], []

    for start in range(0, len(entries), batch):
        chunk = entries[start : start + batch]
        chunk_faces = faces[start : start + batch]
        images = [_open(crops_dir / entry["file"]) for entry in chunk]

        recognised = [i for i, face in enumerate(chunk_faces) if face is not None]
        if recognised:
            similarity = encoder.images([images[i] for i in recognised])
            similarity = similarity @ prompt_vectors.T
            for row, i in enumerate(recognised):
                vectors.append(chunk_faces[i].vector)
                described.append(
                    _describe(chunk[i], images[i], chunk_faces[i], similarity[row])
                )

        read = min(start + batch, len(entries))
        print(f"  read {read}/{len(entries)} crops, {len(described)} with a face")

    return np.array(vectors), described


def _vote(members) -> dict[str, dict]:
    scores: dict[str, Counter] = {}
    for crop in members:
        for group, (label, score) in crop["labels"].items():
            if label is not None:
                scores.setdefault(group, Counter())[label] += score

    voted = {}
    for group, tally in scores.items():
        label, total = tally.most_common(1)[0]
        voted[group] = {"label": label, "confidence": round(total / len(members), 2)}
    return voted


def _profile(number: int, members, min_confidence: float) -> dict:
    voted = _vote(members)
    worn = [colour for crop in members for colour in crop["colours"]]
    common = [colour for colour, _ in Counter(worn).most_common(MAX_COLOURS)]
    keywords = [
        group["label"]
        for group in voted.values()
        if group["confidence"] >= min_confidence
    ]
    return {
        "id": f"p{number:03d}",
        "appearances": len(members),
        "photos": sorted({crop["source"] for crop in members}),
        "crops": [crop["file"] for crop in members],
        "keywords": keywords + common,
        "attributes": voted,
        "colours": common,
    }


def build(described, vectors, args) -> tuple[list[dict], np.ndarray]:
    """Cluster the described crops and turn each cluster into a profile.

    Also returns one averaged identity vector per profile, which is what stage
    3 matches against its gallery.
    """
    sources = [crop["source"] for crop in described]
    groups = cluster(vectors, sources, args.threshold)
    profiles = [
        _profile(number, [described[i] for i in members], args.min_confidence)
        for number, members in enumerate(groups, start=1)
    ]
    return profiles, np.array([average(vectors[members]) for members in groups])


def write_folders(profiles, crops_dir: Path, out_dir: Path) -> None:
    """Copy each profile's crops into a folder of their own."""
    for profile in profiles:
        folder = out_dir / profile["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for name in profile["crops"]:
            shutil.copy2(crops_dir / name, folder / name)
