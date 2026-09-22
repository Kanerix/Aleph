"""People found by the pose model, and the geometry that describes them.

A pose model is used rather than a face detector because a crowd photo defeats
both obvious alternatives: a face detector returns floating heads with no body,
and a plain person detector cannot tell a face from the back of a head. The
keypoints give a full body box and the facial landmarks at the same time.
"""

import math
from dataclasses import dataclass, field

# COCO-17 keypoint indices used by the YOLO pose models.
FACE_KPS = (0, 1, 2, 3, 4)  # nose, eyes, ears
SHOULDERS = (5, 6)

# Left/right keypoint pairs used for the front/back test, with a weight each.
ORIENTATION_PAIRS = (((1, 2), 1.0), ((3, 4), 0.8), (SHOULDERS, 1.0))

# A detection that sits almost entirely inside another one is only a duplicate
# when the two are of a similar size - see merge_detections().
NESTED_OVERLAP = 0.90
SIMILAR_AREA = 0.6


@dataclass
class Person:
    """One detected person, in full-image pixel coordinates."""

    box: tuple[float, float, float, float]
    conf: float
    kps: object | None = None  # (17, 3) numpy array -> x, y, confidence
    face: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        """Area of the detection box in square pixels."""
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def orientation_score(kps, kp_conf: float) -> tuple[float, int]:
    """Decide whether a person is facing the camera or turned away.

    COCO keypoints are *anatomical*: someone facing you has their left eye,
    ear and shoulder on the right-hand side of the image, so the sign of
    (left.x - right.x) flips when they turn around. This matters enormously
    in a stand, where the majority of people are filmed from behind - the
    pose model happily predicts a nose and two ears on the back of a head,
    so landmark confidence alone is not enough to prove you can see a face.

    Returns (score, votes); score > 0 means facing the camera.
    """
    score, votes = 0.0, 0
    # Only relative position matters here, so accept lower-confidence points.
    thr = kp_conf * 0.6
    for (left, right), weight in ORIENTATION_PAIRS:
        if kps[left][2] >= thr and kps[right][2] >= thr:
            dx = float(kps[left][0] - kps[right][0])
            if abs(dx) < 1.0:  # edge-on, no usable evidence
                continue
            score += weight if dx > 0 else -weight
            votes += 1
    return score, votes


def face_geometry(kps, kp_conf: float) -> dict:
    """Estimate head position/size and camera-facing score from keypoints.

    Returns {} when no facial landmarks are visible at all.
    """
    if kps is None:
        return {}

    visible = [(kps[i][0], kps[i][1]) for i in FACE_KPS if kps[i][2] >= kp_conf]
    if not visible:
        return {}

    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)

    # Shoulder width is the most stable scale reference when the head is
    # turned and the landmarks bunch up.
    shoulder_w = 0.0
    ls, rs = kps[SHOULDERS[0]], kps[SHOULDERS[1]]
    if ls[2] >= kp_conf and rs[2] >= kp_conf:
        shoulder_w = math.hypot(ls[0] - rs[0], ls[1] - rs[1])

    head_w = max(span_x * 1.9, span_y * 2.1, shoulder_w * 0.62, 8.0)
    # The landmark centroid sits around eye/nose level, so the head extends
    # further up (forehead, hair, cap) than down (chin).
    top = cy - head_w * 0.80
    bottom = cy + head_w * 0.70

    facing, votes = orientation_score(kps, kp_conf)

    return {
        "center": (cx, cy),
        "width": head_w,
        "height": bottom - top,
        "box": (cx - head_w * 0.5, top, cx + head_w * 0.5, bottom),
        "n_kps": len(visible),
        "facing": facing,
        "votes": votes,
    }


def parse_result(result, off_x: float, off_y: float, kp_conf: float) -> list[Person]:
    """Convert one YOLO result into `Person`s in full-image coordinates."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    kpts = None
    if (
        getattr(result, "keypoints", None) is not None
        and result.keypoints.data is not None
    ):
        kpts = result.keypoints.data.cpu().numpy()

    people = []
    for i in range(len(xyxy)):
        x0, y0, x1, y1 = xyxy[i]
        kps = None
        if kpts is not None and i < len(kpts):
            kps = kpts[i].copy()
            kps[:, 0] += off_x
            kps[:, 1] += off_y
        person = Person(
            box=(x0 + off_x, y0 + off_y, x1 + off_x, y1 + off_y),
            conf=float(confs[i]),
            kps=kps,
        )
        person.face = face_geometry(kps, kp_conf)
        people.append(person)
    return people


def _iou_pair(a, b) -> tuple[float, float]:
    """Return (IoU, intersection-over-smaller-area)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union else 0.0, inter / smaller if smaller else 0.0)


def merge_detections(people: list[Person], iou_thr: float) -> list[Person]:
    """Greedy de-duplication of the full-frame + tile passes.

    Box overlap alone is a blunt instrument here: in a packed terrace two
    different fans genuinely overlap a lot, and a distant person often sits
    entirely inside the box of someone in the foreground. Whenever both
    detections have a located head we therefore compare head positions -
    that is the actual identity of a person - and only fall back to IoU when
    one of them has no face at all.
    """
    kept: list[Person] = []
    for person in sorted(people, key=lambda p: p.conf, reverse=True):
        duplicate = False
        for other in kept:
            if person.face and other.face:
                dist = math.hypot(
                    person.face["center"][0] - other.face["center"][0],
                    person.face["center"][1] - other.face["center"][1],
                )
                if dist < 0.45 * min(person.face["width"], other.face["width"]):
                    duplicate = True
                    break
                continue

            iou, iom = _iou_pair(person.box, other.box)
            if iou >= iou_thr:
                duplicate = True
                break
            # A box nested inside a much larger one is usually a different,
            # more distant person - only treat similar-sized pairs as dupes.
            if (
                iom >= NESTED_OVERLAP
                and min(person.area, other.area) / max(person.area, other.area, 1.0)
                > SIMILAR_AREA
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(person)
    return kept
