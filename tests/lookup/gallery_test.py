"""Tests for `src.lookup.gallery`, with the face model replaced by a stub."""

import numpy as np
import pytest
from PIL import Image

from src.faces import Face
from src.lookup import gallery

WIDTH = 4


class _FakeReader:
    """Recognises a face in every photo except the ones named in `blind_to`."""

    def __init__(self, blind_to=()):
        self.blind_to = set(blind_to)
        self.seen = []

    def read(self, image, head_box):
        """Hand back a fixed vector, unless this photo is one to fail on."""
        self.seen.append((image.size, head_box))
        if len(self.seen) in self.blind_to:
            return None
        return Face(vector=np.ones(WIDTH), gender="man", score=0.8)


def photo(path, size=(60, 60)) -> None:
    """Write a readable image file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


@pytest.fixture
def folder(tmp_path):
    """A gallery with one loose photo and one person's folder of two."""
    photo(tmp_path / "anna.jpg")
    photo(tmp_path / "bo" / "first.jpg")
    photo(tmp_path / "bo" / "second.jpg")
    return tmp_path


class TestByPerson:
    """Working out who each reference photo is of."""

    def test_a_loose_photo_is_named_by_its_filename(self, folder):
        """The quick way to build a gallery: one photo per person."""
        assert folder / "anna.jpg" in gallery.by_person(folder)["anna"]

    def test_a_subfolder_groups_several_photos_of_one_person(self, folder):
        """More angles of one face make a better reference."""
        assert len(gallery.by_person(folder)["bo"]) == 2

    def test_ignores_files_that_are_not_images(self, folder):
        """A gallery folder collects notes and thumbnails databases too."""
        (folder / "notes.txt").write_text("not a face", encoding="utf-8")
        assert set(gallery.by_person(folder)) == {"anna", "bo"}

    def test_an_empty_folder_has_nobody_in_it(self, tmp_path):
        """No references is a state the stage has to report, not crash on."""
        assert gallery.by_person(tmp_path) == {}


class TestRead:
    """Turning the folder into one vector per person."""

    def test_one_vector_per_person(self, folder):
        """Two people in the folder means a gallery of two rows."""
        names, vectors = gallery.read(folder, _FakeReader())
        assert names == ["anna", "bo"]
        assert vectors.shape == (2, WIDTH)

    def test_the_surest_face_in_a_reference_photo_is_used(self, folder):
        """There is no stage 1 head box here, so the reader gets None."""
        reader = _FakeReader()
        gallery.read(folder, reader)
        assert [head_box for _, head_box in reader.seen] == [None, None, None]

    def test_keeps_a_person_whose_other_photo_had_no_face(self, folder):
        """One unusable photo must not lose the person it was of."""
        names, vectors = gallery.read(folder, _FakeReader(blind_to=[2]))
        assert names == ["anna", "bo"]
        assert vectors.shape == (2, WIDTH)

    def test_drops_a_person_with_no_face_at_all(self, folder):
        """Matching against a landscape photo would name people at random."""
        names, _ = gallery.read(folder, _FakeReader(blind_to=[1]))
        assert names == ["bo"]

    def test_skips_an_unreadable_file(self, folder):
        """A truncated or mislabelled file must not end the run."""
        (folder / "broken.jpg").write_bytes(b"not a jpeg")
        names, _ = gallery.read(folder, _FakeReader())
        assert names == ["anna", "bo"]
