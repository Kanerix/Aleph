"""Tests for `src.pipeline`, driven by a stand-in detector."""

import numpy as np
from PIL import Image

from src.extract.inference import Detector
from src.extract.pipeline import process_image


class _Tensor:
    """Stand-in for the torch tensors an Ultralytics result exposes."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)

    def __len__(self):
        return len(self.xyxy.numpy())


class _Keypoints:
    def __init__(self, array):
        self.data = _Tensor(array)


class _Result:
    def __init__(self, boxes, confs, kps=None):
        self.boxes = _Boxes(np.array(boxes, dtype=float), np.array(confs, dtype=float))
        self.keypoints = None if kps is None else _Keypoints(np.array(kps, dtype=float))


class _FakeDetector(Detector):
    """Reports the same boxes for every window it is given.

    `Detector.__init__` is deliberately not called, so no weights are loaded.
    """

    pose = True

    def __init__(self, boxes, confs, kps=None):
        self.boxes = boxes
        self.confs = confs
        self.kps = kps

    def run(self, images):
        return [_Result(self.boxes, self.confs, self.kps) for _ in images]


def _photo(path, size=(800, 600)):
    Image.new("RGB", size, "white").save(path)
    return path


class _FakeReader:
    """Recognises a face in every crop except the ones counted in `blind_to`."""

    def __init__(self, blind_to=()):
        self.blind_to = set(blind_to)
        self.seen = []

    def read(self, crop, head_box):
        """Fail on the nth crop when n is in `blind_to`, otherwise succeed."""
        self.seen.append((crop.size, head_box))
        return None if len(self.seen) in self.blind_to else object()


def test_numbering_has_no_gaps(cli_args, tmp_path):
    """A crop dropped by `--min-size` does not leave a hole in the sequence."""
    args = cli_args("--no-require-face", "--no-tile", "--min-size", "60")
    args.out = tmp_path
    photo = _photo(tmp_path / "shot.jpg")
    detector = _FakeDetector(
        [(0, 0, 20, 20), (100, 200, 200, 500), (300, 200, 400, 500)],
        [0.9, 0.8, 0.7],
    )

    entries = process_image(photo, "shot", detector, None, args)

    assert [entry["index"] for entry in entries] == [1, 2]
    assert sorted(p.name for p in tmp_path.glob("shot_*.jpg")) == [
        "shot_001.jpg",
        "shot_002.jpg",
    ]


def test_crops_are_named_after_the_given_stem(cli_args, tmp_path):
    """Two photos with the same filename write to different crops."""
    args = cli_args("--no-require-face", "--no-tile")
    args.out = tmp_path
    photo = _photo(tmp_path / "IMG_0001.jpg")
    detector = _FakeDetector([(100, 200, 200, 500)], [0.9])

    entries = process_image(photo, "IMG_0001-2", detector, None, args)

    assert entries[0]["file"] == "IMG_0001-2_001.jpg"
    assert entries[0]["source"] == str(photo)
    assert (tmp_path / "IMG_0001-2_001.jpg").exists()


def test_manifest_entry_describes_the_crop(cli_args, tmp_path):
    """Each entry carries the crop box and the detection confidence."""
    args = cli_args("--no-require-face", "--no-tile")
    args.out = tmp_path
    photo = _photo(tmp_path / "shot.jpg")
    detector = _FakeDetector([(100, 200, 200, 500)], [0.9])

    entry = process_image(photo, "shot", detector, None, args)[0]

    x0, y0, x1, y1 = entry["crop"]
    assert 0 <= x0 < x1 <= 800
    assert 0 <= y0 < y1 <= 600
    assert entry["confidence"] == 0.9
    assert entry["head"] is None
    assert entry["face_keypoints"] == 0


def test_a_crop_with_no_face_is_never_written(cli_args, tmp_path):
    """The hands and banners the pose model finds do not reach the disk."""
    args = cli_args("--no-require-face", "--no-tile")
    args.out = tmp_path
    photo = _photo(tmp_path / "shot.jpg")
    detector = _FakeDetector([(100, 200, 200, 500), (300, 200, 400, 500)], [0.9, 0.8])

    entries = process_image(photo, "shot", detector, _FakeReader(blind_to=[1]), args)

    assert len(entries) == 1
    assert list(tmp_path.glob("shot_*.jpg")) == [tmp_path / "shot_001.jpg"]


def test_verified_numbering_has_no_gaps(cli_args, tmp_path):
    """Dropping the first crop must not leave the sequence starting at two."""
    args = cli_args("--no-require-face", "--no-tile")
    args.out = tmp_path
    photo = _photo(tmp_path / "shot.jpg")
    detector = _FakeDetector([(100, 200, 200, 500), (300, 200, 400, 500)], [0.9, 0.8])

    entries = process_image(photo, "shot", detector, _FakeReader(blind_to=[1]), args)

    assert [entry["index"] for entry in entries] == [1]
    assert entries[0]["file"] == "shot_001.jpg"


def test_the_reader_is_told_where_the_head_is(cli_args, tmp_path, keypoints):
    """The head box is offset into the crop, not left in source pixels."""
    args = cli_args("--no-tile")
    args.out = tmp_path
    photo = _photo(tmp_path / "shot.jpg")
    detector = _FakeDetector([(60, 60, 160, 300)], [0.9], [keypoints(True)])
    reader = _FakeReader()

    entries = process_image(photo, "shot", detector, reader, args)

    crop_x0, crop_y0 = entries[0]["crop"][:2]
    head = entries[0]["head"]
    assert reader.seen[0][1] == (
        head[0] - crop_x0,
        head[1] - crop_y0,
        head[2] - crop_x0,
        head[3] - crop_y0,
    )


def test_no_reader_keeps_every_crop(cli_args, tmp_path):
    """`--no-verify` leaves the pose model's word as the last one."""
    args = cli_args("--no-require-face", "--no-tile")
    args.out = tmp_path
    photo = _photo(tmp_path / "shot.jpg")
    detector = _FakeDetector([(100, 200, 200, 500), (300, 200, 400, 500)], [0.9, 0.8])

    assert len(process_image(photo, "shot", detector, None, args)) == 2
