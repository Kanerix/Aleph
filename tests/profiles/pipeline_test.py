"""Tests for `src.profiles.pipeline`, with both models replaced by stubs."""

import numpy as np
import pytest
from PIL import Image

from src.faces import Face
from src.profiles import attributes, pipeline

WIDTH = 4


class _FakeEncoder:
    """Returns fixed vectors, so the tests need no weights and no device."""

    def __init__(self):
        self.batches = []

    def texts(self, texts):
        """One all-zero row per prompt, which scores every label as a tie."""
        return np.zeros((len(texts), WIDTH))

    def images(self, images):
        """One identical row per image, and a note of how many came at once."""
        self.batches.append(len(images))
        return np.ones((len(images), WIDTH))


class _FakeReader:
    """Recognises a face in every crop except the ones named in `blind_to`."""

    def __init__(self, gender: str | None = "man", blind_to=()):
        self.gender = gender
        self.blind_to = set(blind_to)
        self.head_boxes = []

    def read(self, image, head_box):
        """Hand back a fixed vector, unless this crop is one to fail on."""
        self.head_boxes.append(head_box)
        if len(self.head_boxes) in self.blind_to:
            return None
        return Face(vector=np.ones(WIDTH), gender=self.gender, score=0.8)


@pytest.fixture
def crops(tmp_path):
    """Two blue crop files and the manifest entries that describe them."""
    entries = []
    for index, source in enumerate(["a.jpg", "b.jpg"], start=1):
        name = f"crop_{index:03d}.jpg"
        Image.new("RGB", (200, 200), (10, 10, 200)).save(tmp_path / name)
        entries.append(
            {
                "file": name,
                "source": f"data/{source}",
                "crop": [0, 0, 200, 200],
                "head": [60, 10, 140, 90],
            }
        )
    return tmp_path, entries


def described(source: str, **labels):
    """One described crop, with every group not named left unlabelled."""
    full = {group: (None, 0.5) for group in attributes.EXCLUSIVE}
    full |= {group: (label, 0.9) for group, label in labels.items()}
    return {"file": f"{source}.jpg", "source": source, "labels": full, "colours": []}


class TestDescribe:
    """Recognising and tagging every crop."""

    def test_one_row_and_one_description_per_crop(self, crops):
        """The output keeps the order the manifest entries came in."""
        crops_dir, entries = crops
        vectors, out = pipeline.describe(
            entries, crops_dir, _FakeEncoder(), _FakeReader(), batch=8
        )
        assert vectors.shape == (2, WIDTH)
        assert [crop["file"] for crop in out] == [e["file"] for e in entries]

    def test_drops_crops_with_no_face(self, crops):
        """Hands, banners and fence panels leave the pipeline here."""
        crops_dir, entries = crops
        vectors, out = pipeline.describe(
            entries, crops_dir, _FakeEncoder(), _FakeReader(blind_to=[1]), batch=8
        )
        assert vectors.shape == (1, WIDTH)
        assert [crop["file"] for crop in out] == ["crop_002.jpg"]

    def test_survives_a_batch_with_no_faces_at_all(self, crops):
        """An empty batch must not break the run or the stacking."""
        crops_dir, entries = crops
        vectors, out = pipeline.describe(
            entries, crops_dir, _FakeEncoder(), _FakeReader(blind_to=[1, 2]), batch=8
        )
        assert out == []
        assert len(vectors) == 0

    def test_tells_the_reader_where_the_head_is(self, crops):
        """The manifest box is what keeps a neighbour's face out."""
        crops_dir, entries = crops
        reader = _FakeReader()
        pipeline.describe(entries, crops_dir, _FakeEncoder(), reader, batch=8)
        assert reader.head_boxes == [(60, 10, 140, 90), (60, 10, 140, 90)]

    def test_takes_gender_from_the_face_model(self, crops):
        """CLIP has no gender prompts, so this is the only source of it."""
        crops_dir, entries = crops
        _, out = pipeline.describe(
            entries, crops_dir, _FakeEncoder(), _FakeReader("woman"), batch=8
        )
        assert out[0]["labels"]["gender"] == ("woman", 0.8)

    def test_names_the_colours_of_the_body(self, crops):
        """Colours come from the region under the head, with no model."""
        crops_dir, entries = crops
        _, out = pipeline.describe(
            entries, crops_dir, _FakeEncoder(), _FakeReader(), batch=8
        )
        assert out[0]["colours"] == ["blue"]

    def test_splits_the_work_into_batches(self, crops):
        """CLIP sees one batch per chunk, so the memory use stays bounded."""
        crops_dir, entries = crops
        encoder = _FakeEncoder()
        pipeline.describe(entries, crops_dir, encoder, _FakeReader(), batch=1)
        assert encoder.batches == [1, 1]


class TestBuild:
    """Turning clustered crops into profiles."""

    def test_numbers_the_profiles_from_one(self, profile_args):
        """Identifiers are stable within a run and zero padded."""
        crops = [described("a"), described("b")]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        profiles = pipeline.build(crops, vectors, profile_args())
        assert [p["id"] for p in profiles] == ["p001", "p002"]

    def test_groups_matching_crops_into_one_profile(self, profile_args):
        """The whole point of the stage: one person, several photos."""
        crops = [described("a"), described("b")]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
        profiles = pipeline.build(crops, vectors, profile_args())
        assert len(profiles) == 1
        assert profiles[0]["appearances"] == 2
        assert profiles[0]["photos"] == ["a", "b"]

    def test_keeps_a_confident_label_as_a_keyword(self, profile_args):
        """Keywords are the attributes that cleared the confidence floor."""
        profiles = pipeline.build(
            [described("a", hair="bald")], np.ones((1, 2)), profile_args()
        )
        assert "bald" in profiles[0]["keywords"]
        assert profiles[0]["attributes"]["hair"]["label"] == "bald"

    def test_drops_a_label_below_the_confidence_floor(self, profile_args):
        """A guess the model is unsure of is recorded but not advertised."""
        args = profile_args("--min-confidence", "0.95")
        profiles = pipeline.build([described("a", hair="bald")], np.ones((1, 2)), args)
        assert profiles[0]["keywords"] == []
        assert profiles[0]["attributes"]["hair"]["label"] == "bald"

    def test_the_majority_label_wins_a_disagreement(self, profile_args):
        """Voting across appearances is what makes several crops worth having."""
        crops = [
            described("a", headwear="cap"),
            described("b", headwear="cap"),
            described("c", headwear="hood"),
        ]
        profiles = pipeline.build(crops, np.ones((3, 2)) / np.sqrt(2), profile_args())
        assert profiles[0]["attributes"]["headwear"]["label"] == "cap"


class TestWriteFolders:
    """Copying each profile's crops into a folder of their own."""

    def test_copies_every_crop(self, tmp_path, crops):
        """The folder is how a person is reviewed by eye."""
        crops_dir, entries = crops
        out = tmp_path / "profiles"
        profile = {"id": "p001", "crops": [e["file"] for e in entries]}
        pipeline.write_folders([profile], crops_dir, out)
        assert sorted(p.name for p in (out / "p001").iterdir()) == [
            "crop_001.jpg",
            "crop_002.jpg",
        ]
