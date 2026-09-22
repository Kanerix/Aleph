"""Zero-shot keywords for one crop, scored against CLIP text prompts.

The vocabulary lives here as plain dictionaries. Adding a keyword means adding
a prompt, not training anything, so this is the file to edit when the tags do
not fit the photos you are running. Gender is not in here because the face
model in `src/faces.py` reports it from the face itself.
"""

import numpy as np

# CLIP is trained with this scale on the cosine similarities, so reusing it
# turns the raw similarities into probabilities that spread out sensibly.
LOGIT_SCALE = 100.0

# One label per group wins, the one whose prompt fits best.
EXCLUSIVE = {
    "headwear": {
        "cap": "a person wearing a baseball cap",
        "beanie": "a person wearing a woolly hat",
        "hood": "a person wearing a hood over their head",
        "bare head": "a person with nothing on their head",
    },
    "clothing": {
        "football shirt": "a football supporter wearing a football shirt",
        "jacket": "a person wearing a jacket or coat",
        "hoodie": "a person wearing a hooded sweatshirt",
        "t-shirt": "a person wearing a plain t-shirt",
        "bare chested": "a shirtless person with a bare chest",
    },
    "hair": {
        "short hair": "a person with short hair",
        "long hair": "a person with long hair",
        "bald": "a bald person",
    },
}

# Each of these is decided on its own, against its own negative prompt.
INDEPENDENT = {
    "scarf": (
        "a football supporter wearing a scarf",
        "a football supporter with no scarf",
    ),
    "glasses": (
        "a person wearing glasses",
        "a person not wearing glasses",
    ),
    "beard": (
        "a man with a beard",
        "a man with no beard, clean shaven",
    ),
    "flag": (
        "a supporter holding up a flag or banner",
        "a supporter holding nothing",
    ),
    "printed shirt": (
        "clothing with a large printed logo or lettering",
        "plain clothing with no print",
    ),
}


def prompts() -> list[str]:
    """Every prompt, in the order `read` expects the similarities in."""
    ordered = []
    for options in EXCLUSIVE.values():
        ordered += list(options.values())
    for yes, no in INDEPENDENT.values():
        ordered += [yes, no]
    return ordered


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max())
    return shifted / shifted.sum()


def read(similarities: np.ndarray) -> dict[str, tuple[str | None, float]]:
    """Turn one row of image-to-prompt similarities into labelled scores.

    Returns a label and its probability per group. For the independent groups
    the label is the group name itself, or None when the negative prompt wins.
    """
    scores = similarities * LOGIT_SCALE
    labels: dict[str, tuple[str | None, float]] = {}
    at = 0

    for group, options in EXCLUSIVE.items():
        window = _softmax(scores[at : at + len(options)])
        best = int(window.argmax())
        labels[group] = (list(options)[best], float(window[best]))
        at += len(options)

    for group in INDEPENDENT:
        yes, no = _softmax(scores[at : at + 2])
        labels[group] = (group if yes > no else None, float(max(yes, no)))
        at += 2

    return labels
