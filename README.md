# 💄 TryMyLook — Virtual Makeup Try-On

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://trymylookprototypegit-dncx6weaa3jptnrouebrh4.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/built%20with-PyTorch-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blueviolet)](#contributing)

**TryMyLook** is an open-source virtual makeup try-on system. Upload a
selfie, pick a product — lipstick, eyeshadow, foundation, blush, or a full
coordinated look — and see it applied to your photo in real time, with
adjustable intensity and a before/after comparison.

It's built as a deep-learning *computer vision pipeline*, not a filter: face
landmarks are detected with a neural network, face regions are semantically
segmented pixel-by-pixel, and makeup colors are blended using
luminance-aware compositing so the result tracks lighting and skin texture
instead of sitting on top like a flat sticker.

**🔗 Live demo:** https://trymylookprototypegit-dncx6weaa3jptnrouebrh4.streamlit.app/

---

## Table of contents

- [Why this project exists](#why-this-project-exists)
- [How it works](#how-it-works)
  - [1. Face detection & landmarks](#1-face-detection--landmarks)
  - [2. Face parsing / segmentation](#2-face-parsing--segmentation)
  - [3. Makeup compositing](#3-makeup-compositing)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Optional: enabling BiSeNet (higher-fidelity masks)](#optional-enabling-bisenet-higher-fidelity-masks)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Tech stack](#tech-stack)
- [License](#license)

---

## Why this project exists

Most "virtual try-on" demos online are either a static color overlay
(no understanding of *where* the lips or eyelids actually are in *this*
specific photo) or a closed commercial SDK you can't read the code for.

TryMyLook is an attempt at the middle ground: a small, inspectable,
from-scratch pipeline that demonstrates the actual building blocks a
production AR-beauty feature would use — landmark detection, semantic face
parsing, and intensity-aware color blending — in plain, readable Python,
with no black boxes.

It's intentionally built to **degrade gracefully**: if you don't have GPU
infrastructure or a deep face-parsing model handy, the app automatically
falls back to a lighter geometric method and keeps working. That fallback
behavior is a deliberate design choice, not a missing feature — see
[Architecture](#architecture) below.

## How it works

The pipeline has three stages, each implemented as its own module in
[`src/trymylook/`](src/trymylook/):

### 1. Face detection & landmarks

[`face_detection.py`](src/trymylook/face_detection.py) — `DeepLearningFaceDetector`

Uses the [`face-alignment`](https://github.com/1adrianb/face-alignment)
network (a PyTorch model built on a ResNet/Hourglass-style backbone) to
locate **68 facial landmark points** — the same dense landmark scheme
used by classic dlib-based pipelines, but predicted by a neural network
instead of a hand-engineered detector. From the landmark set, the detector
also derives the face's bounding box, center point, and in-plane rotation
angle, which downstream stages use for masking and alignment.

### 2. Face parsing / segmentation

[`segmentation.py`](src/trymylook/segmentation.py) — `HybridSegmenter`

This is the part of the pipeline that decides *which pixels* belong to
lips, eyelids, skin, etc. It supports two interchangeable strategies
behind one interface:

| Method | Class | How it works | Quality |
|---|---|---|---|
| **BiSeNet face parsing** | `BiSeNetSegmenter` | A Bilateral Segmentation Network classifies every pixel into one of 19 semantic face classes (skin, eyebrows, eyes, nose, lips, hair, etc.) | Pixel-accurate, handles odd angles and expressions well |
| **Landmark-geometric** | `LandmarkSegmenter` | Builds masks directly from convex hulls / polygons over the 68 landmark points (e.g. the lip mask is the polygon connecting the mouth landmarks) | Fast, dependency-light, no extra model download |

`HybridSegmenter` tries to load BiSeNet on startup; if the model weights
aren't present (they're ~50 MB and deliberately **not committed to this
repo** — see [below](#optional-enabling-bisenet-higher-fidelity-masks)),
it transparently uses the landmark-geometric method instead, and the rest
of the app behaves identically either way. This is why the live demo runs
on a free, CPU-only, memory-constrained host without ever crashing on
startup.

### 3. Makeup compositing

[`makeup_application.py`](src/trymylook/makeup_application.py) — `NeuralMakeupApplicator`

Once a mask exists for a region, applying makeup isn't just "paint this
color here." The applicator:

- Builds a colored overlay and blends it against the original pixels using
  one of several blend modes (`multiply`, `overlay`, `screen`, etc.) —
  the same compositing math used in image-editing software, chosen per
  product (e.g. lipstick uses a different blend curve than foundation).
- Scales blend strength by the user's **intensity** slider (0–100%).
- Includes texture-preservation and adaptive-lighting helpers so the
  result respects the photo's original shading rather than flattening it.

`app.py` orchestrates all three stages behind a Streamlit UI: it collects
the uploaded image and user selections, calls
`detector → segmenter → applicator` in sequence, and renders the result,
a before/after view, processing-time metrics, and a download button.

## Architecture

```
                 ┌─────────────────────┐
   selfie  ───▶  │ DeepLearningFaceDet. │ ──▶ 68 landmarks, bbox, angle
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   HybridSegmenter    │
                 │  ┌───────────────┐   │
                 │  │ BiSeNet (if   │   │ ──▶ per-region binary masks
                 │  │ weights found)│   │      (lips / eyes / skin / …)
                 │  ├───────────────┤   │
                 │  │ Landmark      │   │
                 │  │ fallback      │   │
                 │  └───────────────┘   │
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ NeuralMakeupApplic.  │ ──▶ blended result image
                 │ (blend mode +        │
                 │  intensity + shade)  │
                 └─────────────────────┘
```

The "Complete Look" mode runs all four product steps
(foundation → blush → eyeshadow → lipstick) through the same pipeline in
one pass, batching mask creation when BiSeNet is active so it's roughly
4× faster than calling each product individually.

## Project structure

```
.
├── app.py                          # Streamlit UI — orchestrates the pipeline
├── src/trymylook/
│   ├── config.py                   # Shades, presets, app constants
│   ├── utils.py                    # Image I/O / resize / validation helpers
│   ├── face_detection.py           # DeepLearningFaceDetector
│   ├── segmentation.py             # HybridSegmenter (BiSeNet + landmark fallback)
│   └── makeup_application.py       # NeuralMakeupApplicator (blending engine)
├── models/face-parsing.PyTorch/    # BiSeNet repo + weights go here (gitignored)
├── assets/                         # Sample images
├── tests/test_system.py            # End-to-end smoke tests for every module
├── .devcontainer/                  # One-click GitHub Codespaces setup
├── .streamlit/config.toml          # Server/theme config for deployment
├── packages.txt                    # apt-level system deps (libgl1, for OpenCV)
├── pyproject.toml                  # Makes `trymylook` pip-installable (-e .)
└── requirements.txt
```

**5 products** (Lipstick, Eyeshadow, Foundation, Blush, Complete Look) ·
**25 individual shades** · **5 curated Complete Look presets**.

## Getting started

```bash
git clone https://github.com/Ayush-23479/trymylook_prototype.git
cd trymylook_prototype
pip3 install --user -r requirements.txt
streamlit run app.py
```

`requirements.txt` installs this repo itself in editable mode (`-e .`),
so `from trymylook.config import ...` works regardless of your working
directory.

> **Note on OpenCV:** this project uses `opencv-python-headless` and
> needs the system library `libgl1` to import correctly on minimal Linux
> environments (Docker, Codespaces, CI). It's already declared in
> [`packages.txt`](packages.txt) and installed automatically by the
> devcontainer — but if you're not using a devcontainer, run
> `sudo apt-get install -y libgl1` once first.

### GitHub Codespaces

This repo includes a [`.devcontainer`](.devcontainer/devcontainer.json)
config — open it in a Codespace and dependencies install automatically;
the app starts and forwards port `8501` for you.

## Optional: enabling BiSeNet (higher-fidelity masks)

The app runs perfectly well without this step (it uses the landmark
fallback automatically). To enable pixel-accurate BiSeNet face parsing:

1. Clone the [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) repo into `models/face-parsing.PyTorch/`.
2. Download the pretrained checkpoint `79999_iter.pth` (see that repo's README for the link) and place it at:
   ```
   models/face-parsing.PyTorch/res/cp/79999_iter.pth
   ```
3. Restart the app. You'll see the startup log switch from
   `"Using landmark-based segmentation only"` to
   `"BiSeNet segmenter loaded - Using deep learning mode"`.

The checkpoint is intentionally excluded from version control (it's
~50 MB and a derived artifact, not source code) — see
[`.gitignore`](.gitignore).

## Deployment

The live demo runs on [Streamlit Community Cloud](https://streamlit.io/cloud),
deployed straight from this repo's `main` branch. A few choices worth
calling out for anyone deploying their own fork:

- **CPU-only PyTorch.** `requirements.txt` pins `torch==2.3.0+cpu` /
  `torchvision==0.18.0+cpu` from PyTorch's CPU wheel index. The default
  `torch>=2.0.0` resolves to a multi-GB CUDA build that comfortably blows
  past free-tier memory/disk limits for an app that only ever runs
  inference on CPU anyway.
- **No BiSeNet weights committed.** Keeps the repo light and avoids
  Git LFS entirely, at the cost of using the (still fully functional)
  landmark fallback in the hosted demo.
- **`packages.txt` is minimal on purpose.** Streamlit Cloud's base image
  doesn't track every Debian release in lockstep with `apt`'s package
  index, so over-specifying system packages (we initially included
  `libglib2.0-0`) can produce unsatisfiable dependency errors on deploy.
  Only `libgl1` — the actual library OpenCV needs — is listed.

## Roadmap

Contributions toward any of these are very welcome:

- [ ] Re-introduce BiSeNet on the hosted demo via a lazy, on-demand model download instead of committing weights
- [ ] GPU-aware device selection (`cuda` / `mps` / `cpu`) instead of hardcoded CPU
- [ ] Batch image processing (apply a look to multiple photos at once)
- [ ] A REST API layer around the pipeline (decoupled from the Streamlit UI) for integration into other apps
- [ ] Additional shades / skin-tone-aware shade recommendations
- [ ] Unit tests with `pytest` and CI (GitHub Actions) on top of the existing smoke tests

## Contributing

This project is open source and contributions are genuinely welcome —
bug fixes, new features, better blending algorithms, documentation, or
just opening issues for things that don't work as expected.

1. Fork the repo and create a branch: `git checkout -b feature/your-idea`
2. Make your changes. Run the smoke tests:
   ```bash
   python3 tests/test_system.py
   ```
3. Commit with a clear message and open a Pull Request describing **what**
   changed and **why**.

If you're picking something off the [Roadmap](#roadmap), feel free to open
an issue first to say you're working on it, so effort doesn't overlap.

## Tech stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Face landmarks | [`face-alignment`](https://github.com/1adrianb/face-alignment) (PyTorch) |
| Face parsing | [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) (optional) + custom landmark-geometric fallback |
| Image processing | OpenCV, NumPy, scikit-image, Pillow |
| Inference runtime | PyTorch (CPU), ONNX Runtime |
| Packaging | `pyproject.toml` (editable install) |

## License

MIT — see [`LICENSE`](LICENSE). Use it, fork it, ship it.

---

*Built by [Ayush Verma](https://github.com/Ayush-23479). If you use this
project or build on it, a ⭐ on the repo is always appreciated.*
