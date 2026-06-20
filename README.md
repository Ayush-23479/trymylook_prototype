# 💄 TryMyLook — Virtual Makeup Try-On

A Streamlit app that applies realistic virtual makeup (foundation, blush,
eyeshadow, lipstick) to a face photo using deep-learning face detection and
BiSeNet-based face parsing, with an automatic landmark-based fallback when
BiSeNet weights aren't available.

## Project structure

```
.
├── app.py                          # Streamlit UI entrypoint
├── src/trymylook/
│   ├── config.py                   # Shades, presets, constants
│   ├── utils.py                    # Image helper functions
│   ├── face_detection.py           # DeepLearningFaceDetector
│   ├── segmentation.py             # HybridSegmenter (BiSeNet + landmark fallback)
│   ├── makeup_application.py       # NeuralMakeupApplicator
│   └── pipeline.py                 # Orchestrates detect -> mask -> apply
├── models/face-parsing.PyTorch/    # BiSeNet repo + weights (not committed)
├── assets/                         # Sample images
├── tests/                          # test_system.py
└── .devcontainer/                  # Codespaces config
```

## Setup

```bash
pip3 install --user -r requirements.txt
```

`requirements.txt` installs this project itself in editable mode (`-e .`),
so `from trymylook.config import ...` works from anywhere.

### BiSeNet weights (optional but recommended)

The app works without this — it automatically falls back to a
landmark-based segmenter — but for best quality, place the BiSeNet
face-parsing repo + checkpoint at:

```
models/face-parsing.PyTorch/
└── res/cp/79999_iter.pth
```

## Run

```bash
streamlit run app.py
```

In GitHub Codespaces, open the forwarded port `8501` from the **Ports** tab.

## System dependencies

OpenCV needs `libgl1` on minimal Linux containers (already handled by
`packages.txt` in this repo, and auto-installed by the devcontainer).
