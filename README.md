# Aleph

Facial recognition & identity discovery.

Aleph is a data pipeline: it finds every face in a photo and works out whose
profile belongs to it.

| Stage | Status | Code |
| --- | --- | --- |
| 1. `extract`: one cropped portrait per person in a photo | implemented | [`src/extract/`](src/extract/) |
| 2. `profile`: group the crops of one person and tag them | implemented | [`src/profiles/`](src/profiles/) |
| 3. `lookup`: put a name on a profile from a gallery you supply | implemented | [`src/lookup/`](src/lookup/) |

Each stage is a subcommand that reads the folder the stage before it wrote,
including its manifest, so stages can be re-run on their own.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python -m src.main extract stands.jpg      # first run downloads the weights
uv run python -m src.main profile faces/
uv run python -m src.main lookup profiles/ known/
```

`extract` writes one crop per person as `faces/<photo>_001.jpg`, framed with the
head near the top and the torso below, plus `faces/manifest.json`. `profile`
reads that folder and writes `profiles/p001/` per person, plus
`profiles/profiles.json`. `lookup` compares those profiles with a folder of
reference photos of people you already know and writes
`identities/identities.json`.

```sh
uv run python -m src.main extract DCIM/ --aspect 3:4 --min-out 512
uv run python -m src.main extract shoot/*.ARW --raw-half   # quick pass over a card
uv run python -m src.main extract stands.jpg --annotate    # preview what was found
uv run python -m src.main profile faces/ --threshold 0.5
uv run python -m src.main lookup profiles/ squad/ --threshold 0.5
uv run python -m src.main --help
```

## Stage 1: extract

1. **Detection**: an Ultralytics YOLO *pose* model finds people and their
   COCO keypoints in a single pass. A pose model beats the obvious
   alternatives: a face detector returns floating heads with no body, and a
   plain person detector cannot tell a face from the back of a head.
2. **Tiled inference**: large photos are cut into overlapping 768px tiles,
   each run at `--imgsz 1280`. Without this, a person 40px wide in a 7000px
   crowd photo is invisible to the model.
3. **Facing check**: YOLO confidently predicts a nose and two ears on
   the *back* of a head, so landmark confidence alone lets rows of backs
   through. COCO keypoints are *anatomical*, so someone facing the camera has
   their left eye, ear and shoulder on the image's **right**: the sign of
   `left.x - right.x` flips when they turn around. Eyes, ears and shoulders
   each get a weighted vote.
4. **Framing**: the crop is anchored to the head and may only grow within
   limits measured in head sizes. The raw detector box is a bad guide: it
   clips hair, stops at the chin when the torso is hidden by the row in front,
   and balloons across half a row when someone raises an arm.
5. **De-duplication**: overlapping tiles see the same person twice, so
   detections are merged by **head position** rather than box overlap. In a
   packed crowd two different people genuinely overlap, and a distant person
   often sits entirely inside a foreground person's box.
6. **Verification**: a face recognition model has the last word on every crop
   before it is written. None of the keypoint filters above stop the pose model
   claiming a person on a raised hand, a banner or a stretch of fence, and a
   model that only knows about faces does. On the sample set this dropped 189 of
   630 crops, so a third of the output was not a person at all. Turn it off with
   `--no-verify`. The identity vector read here is kept, which is what saves
   stages 2 and 3 from reading the same face again.

Use `--annotate` to write a full-size preview with the crop boxes (green) and
detected heads (red) drawn on, which makes tuning obvious.

### Common options

| Flag | Default | Notes |
| --- | --- | --- |
| `-o, --out` | `faces` | Output folder |
| `-m, --model` | `yolo11m-pose.pt` | `yolo11x-pose.pt` finds more, `yolo11n-pose.pt` is fastest |
| `--aspect` | off | Force a ratio, e.g. `3:4`, `1:1` |
| `--min-out` | off | Upscale so the short side is at least N px |
| `--min-face` | `16` | Minimum head width in px |
| `--min-kps` | `2` | Visible facial keypoints required, the main quality filter |
| `--conf` | `0.25` | Lower finds more people, and more junk |
| `--tile` | `768` | Smaller tiles find smaller faces, but cost time |
| `--no-tile` | off | Single full-frame pass; much faster, misses distant people |
| `--body` | `2.5` | Torso to keep, in head-heights below the chin |
| `--frame-width` | `3.2` | Max crop width in head-widths |
| `--no-facing-check` | off | Keep backs of heads too |
| `--no-require-face` | off | Crop every detected person, face or not |
| `--no-verify` | off | Keep crops the face model finds no face in, and save no vectors |
| `--annotate` | off | Save a preview with boxes drawn |

**Getting more faces:** `--model yolo11x-pose.pt --tile 512 --conf 0.15`.
**Going faster:** `--tile 1280`, `--raw-half`, or `--model yolo11n-pose.pt`.

`faces/manifest.json` is always written, because it is the boundary between
this stage and the next. It records the source photo, crop box, confidence,
head box, facial keypoint count, facing score and the face model's reading of
gender per crop, so stage 2 knows where the head is inside each crop without
re-running detection.

The 512-d identity vector of each crop goes in `faces/vectors.npy` beside it,
one row per manifest entry, in the same order. It lives in its own file because
one vector is 14 KB and 532 lines of indented JSON, which would bury everything
in the manifest that a person actually reads. Stage 2 uses these rows instead
of running the face model a second time, and re-reads the faces itself only
when `--no-verify` left none.

The device is auto-detected (CUDA, then Apple `mps`, then CPU); override it
with `--device cpu|mps|0`.

### Input formats

JPEG, PNG, TIFF, WebP, BMP and HEIC/HEIF (via pillow-heif), plus camera RAW
(`.ARW` `.CR2` `.CR3` `.NEF` `.DNG` `.RAF` `.ORF` `.RW2` `.PEF` `.SRW` `.IIQ`
`.X3F` and friends) decoded through rawpy/LibRaw.

RAW files embed the JPEG the camera itself produced, and on modern bodies that
preview is full resolution. `--raw-source auto` (the default) uses it when it
is at least 90% of the raw width, because it is several times faster than
demosaicing and, on a camera newer than your LibRaw build, more correct.

That last point is not hypothetical. On a Sony **ILCE-7M5**, LibRaw 0.22.1 has
no profile for the body and returns the entire sensor frame (7168×5120, with
431 unlit rows and 127 unlit columns) instead of the real 7008×4672 image. The
embedded preview is exactly 7008×4672.

| `--raw-source` | Behaviour |
| --- | --- |
| `auto` *(default)* | Embedded JPEG if full size, otherwise develop the RAW |
| `preview` | Always use the embedded JPEG |
| `develop` | Always demosaic; unlit sensor margins are trimmed automatically |

When developing, the camera's own white balance is used so crops match the
out-of-camera JPEG. Auto-brightening is **off** by default because LibRaw
rescales each frame independently, which makes a burst from one shoot come out
inconsistently exposed. Turn it on with `--raw-auto-bright`.

## Stage 2: profile

`profile` turns a folder of crops into one profile per person.

```sh
uv run python -m src.main profile faces/ --out profiles
```

1. **Recognition**: stage 1 already read every crop with the `buffalo_l` face
   model to verify it, so the 512-d ArcFace vector and the gender it found come
   straight out of `faces/vectors.npy` and the manifest. The face model is not
   loaded here at all unless the crops were extracted with `--no-verify`, in
   which case this stage does the reading, using the head box from stage 1 to
   decide which face a crop holding more than one is actually of. A crop with
   no face in it is dropped.
2. **Clustering**: single-linkage over face pairs above `--threshold`, closest
   pair first, with one constraint: two crops from the *same* photo are never
   merged, because stage 1 already de-duplicated within a photo, so they must be
   two different people.
3. **Keywords**: CLIP (`ViT-B-32`) scores the crop against written prompts for
   clothing, headwear, hair, scarf, glasses and so on. The vocabulary is plain
   dictionaries in [`src/profiles/attributes.py`](src/profiles/attributes.py):
   adding a keyword means adding a prompt, not training anything.
4. **Colours**: the region below the head is reduced to hue, saturation and
   value and named with no model at all.
5. **Voting**: a person seen in four photos gets one label per group, weighted
   by how sure each appearance was. `--min-confidence` decides which of them
   become keywords.

Each profile gets a folder of its crops so it can be checked by eye, and an
entry in `profiles/profiles.json`:

```json
{
  "id": "p002",
  "appearances": 4,
  "photos": ["data/fck fans 6.JPG", "data/fck fans 7.ARW"],
  "crops": ["fck fans 6_031.jpg", "fck fans 7_029.jpg"],
  "keywords": ["football shirt", "bald", "beard", "man", "black", "dark blue"],
  "attributes": { "beard": { "label": "beard", "confidence": 0.93 } },
  "colours": ["black", "dark blue"]
}
```

The averaged identity vector of each profile is saved in `profiles/vectors.npy`
in the same order, which is what stage 3 matches against its gallery.

### Why two models

Identity and keywords are different questions. CLIP answers "what does this
photo look like", which in a stand full of supporters in black caps behind a
turquoise fence groups people by background and clothing rather than by face.
Measured over the sample set's cross-photo pairs, CLIP head embeddings sit at a
median of 0.66 and run all the way to 0.96, leaving no threshold that separates
anything: at 0.85 the largest clusters were plainly several different men.
ArcFace over the same pairs sits at a median of 0.036, with the 99.9th
percentile at 0.56, which is a usable gap.

### Common options

| Flag | Default | Notes |
| --- | --- | --- |
| `-o, --out` | `profiles` | Output folder |
| `--threshold` | `0.45` | Face similarity needed to call two crops the same person |
| `--min-confidence` | `0.6` | Drop keywords the models are less sure of |
| `--batch` | `32` | Crops per CLIP forward pass |
| `--model` | `ViT-B-32` | open_clip model for the keywords |

Raising `--threshold` splits people apart, lowering it merges strangers. At
0.30 the sample set over-merged visibly; 0.45 held up under inspection.

### Not implemented

The text printed on a shirt is **not** read. CLIP has a yes/no `printed shirt`
prompt and nothing more, because it cannot reliably read text. Real OCR needs
another dependency and another model download.

## Stage 3: lookup

`lookup` puts a name on a profile by comparing it with photos of people you
already have.

```sh
uv run python -m src.main lookup profiles/ known/ --out identities
```

The gallery is a folder you build by hand, either way round:

```text
known/
  anna.jpg          one photo, named after the person it shows
  bo/               or a folder per person, for several photos of them
    portrait.jpg
    stand.jpg
```

1. **References**: every gallery photo is read by the same `buffalo_l` model
   the earlier stages use. A reference photo is taken to be of the person it is
   filed under, so the surest face in it wins and anyone else in the frame is
   ignored. Several photos of one person are averaged into one vector, which
   makes the reference less dependent on the angle of a single photo.
2. **Profile vectors**: stage 2 averaged each profile's crops into one vector
   and saved it in `profiles/vectors.npy`, so no crop is read again here. A
   lookup can therefore be re-run against a different gallery for the cost of
   reading the gallery.
3. **Matching**: cosine similarity against every reference. The closest one
   above `--threshold` names the profile, and the runner-up is recorded beside
   it, because a winner at 0.51 with a second place at 0.49 means the gallery
   cannot tell those two people apart.

Every profile gets an entry in `identities/identities.json`, matched or not:

```json
{
  "id": "p002",
  "appearances": 4,
  "photos": ["data/fck fans 6.JPG", "data/fck fans 7.ARW"],
  "match": "anna",
  "score": 0.612,
  "runner_up": { "name": "bo", "score": 0.204 }
}
```

An unmatched profile keeps the score it reached, so a threshold set slightly
too high is visible instead of silent.

### Common options

| Flag | Default | Notes |
| --- | --- | --- |
| `-o, --out` | `identities` | Output folder |
| `--threshold` | `0.45` | Face similarity needed before a profile is named |
| `--device` | `auto` | `cpu`, `mps` or a cuda index |

Unlike stage 2, there is no head box to go by in the gallery, so the surest
face in each reference photo is used. A profile's own vector is the average of
its crops, which is what limits the damage when a neighbour caught at the edge
of one crop won it back in stage 1.

### What this stage does not do

It matches against the gallery folder and nothing else. There is no web search,
no face search service and no social media, so a profile can only ever be named
after someone whose photo you put in the folder yourself.

That is a deliberate limit. Running crowd photos through a face search engine
to find the accounts behind them de-anonymises people who never agreed to it,
and a face vector is biometric data that needs a legal basis before it is
processed at all (GDPR Art. 9). Keep the gallery to people you have a reason
and a right to identify.

## Layout

```text
src/main.py             subcommand dispatcher, one per pipeline stage
src/manifests.py        the artefacts that join one stage to the next
src/device.py           CUDA / mps / CPU selection
src/faces.py            ArcFace face recognition, used by every stage
src/extract/            stage 1
  cli.py                flags and entry point for `extract`
  pipeline.py           one photo, from load to written crops
  inference.py          pose model and tiling
  detections.py         people, head geometry and de-duplication
  cropping.py           aspect ratios and crop framing
  images.py             input discovery, decoding (incl. RAW) and output
src/profiles/           stage 2
  cli.py                flags and entry point for `profile`
  pipeline.py           describe every crop, then build the profiles
  encoder.py            CLIP image and text embeddings
  attributes.py         the keyword vocabulary
  colours.py            naming colours without a model
  regions.py            stage 1 boxes mapped onto the saved crop
  clustering.py         grouping crops into people
src/lookup/             stage 3
  cli.py                flags and entry point for `lookup`
  gallery.py            reference faces from a folder of named photos
  pipeline.py           profile vectors, then the identities manifest
  matching.py           cosine matching against the gallery
tests/                  pytest suite, mirroring src/
pyproject.toml          dependencies, locked in uv.lock
ruff.toml               lint configuration
```

Model weights are downloaded on first run: the pose model (~40 MB) into the
working directory, `buffalo_l` (~280 MB) into `~/.insightface` and CLIP into
the Hugging Face cache. Weights, source photos, the gallery and the `faces/`,
`profiles/` and `identities/` output are all gitignored, because they are
enormous or personal.

## Development

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which also
manages the Python version from [`.python-version`](.python-version).

```sh
uv sync                     # sync the environment with uv.lock
uv sync --no-dev            # runtime dependencies only
uv add httpx                # add a dependency (--dev / --group lint for groups)
uv lock --upgrade           # update the lock file
```

Activate the environment with `. .venv/bin/activate`, or prefix commands with
`uv run`.

**NOTE:** If your editor reports missing dependencies, select the `.venv`
interpreter for the project (in VSCode: `F1` → `Python: Select interpreter`).

### Linting and tests

[Ruff](https://docs.astral.sh/ruff/) is the linter and formatter; the enabled
rule sets live in [`ruff.toml`](ruff.toml).

```sh
uv run ruff check           # lint (--fix to autofix)
uv run ruff format          # format
uv run pytest               # tests
```

### Githooks

Opt-in: `ruff check` runs before every push to `main`/`master`/`staging`.

```sh
git config --local core.hooksPath .githooks/
```

### Docker

```sh
docker build -t aleph .
docker run --rm -v "$PWD/photos:/app/photos" aleph extract photos/ --out photos/faces
```
