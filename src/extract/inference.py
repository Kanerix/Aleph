"""Run the pose model over a photo, in overlapping tiles when it is large.

A stand photo is mostly distant faces. Running the model once over a 7000px
panorama scales those faces down to nothing, so the image is also cut into
tiles that are each run at full resolution. Duplicates from the overlap are
merged afterwards by `detections.merge_detections`.
"""

import sys
from pathlib import Path

from .detections import Person, parse_result


class Detector:
    """Ultralytics pose model wrapper that falls back to CPU on backend errors."""

    def __init__(self, weights: str, device: str, args):
        """Load `weights` onto `device` using the CLI `args` as predict options."""
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.args = args
        self.pose = "pose" in Path(weights).stem.lower()

    def _predict(self, images):
        return self.model.predict(
            images,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.nms,
            device=self.device,
            max_det=self.args.max_det,
            classes=[0],
            verbose=False,
        )

    def run(self, images):
        """Predict on a batch of images, retrying on CPU if the device fails."""
        try:
            return self._predict(images)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "cpu":
                raise
            print(
                f"  ! {self.device} backend failed ({exc}); falling back to CPU",
                file=sys.stderr,
            )
            self.device = "cpu"
            return self._predict(images)


def tile_origins(total: int, tile: int, step: int) -> list[int]:
    """Tile start offsets along one axis, always ending flush with the edge."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, step))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def plan_tiles(
    W: int, H: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Lay out overlapping tile windows covering a W x H image."""
    step = max(32, int(tile * (1.0 - overlap)))
    return [
        (x, y, min(x + tile, W), min(y + tile, H))
        for y in tile_origins(H, tile, step)
        for x in tile_origins(W, tile, step)
    ]


def detect_people(detector: Detector, image, args) -> list[Person]:
    """Run the full-frame pass plus, for large images, a tiled pass."""
    W, H = image.size
    windows = [(0, 0, W, H)]  # full-frame pass catches the big foreground fans
    if not args.no_tile and max(W, H) > args.tile * 1.3:
        windows += plan_tiles(W, H, args.tile, args.tile_overlap)

    batch = max(1, args.batch)
    people: list[Person] = []
    for i in range(0, len(windows), batch):
        chunk = windows[i : i + batch]
        crops = [image.crop(w) for w in chunk]
        for window, result in zip(chunk, detector.run(crops), strict=True):
            people.extend(parse_result(result, window[0], window[1], args.kp_conf))
    return people
