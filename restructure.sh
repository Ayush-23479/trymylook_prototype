#!/usr/bin/env bash
# ===================================================================
# TryMyLook restructure script
# Run this ONCE from your project root (where app.py currently lives)
# ===================================================================
set -e

echo "📁 Creating new structure..."
mkdir -p src/trymylook
mkdir -p models/face-parsing.PyTorch
mkdir -p assets
mkdir -p tests
mkdir -p .streamlit

echo "🚚 Moving modules into src/trymylook/ ..."
git mv config.py src/trymylook/config.py
git mv utils.py src/trymylook/utils.py
git mv segmentation.py src/trymylook/segmentation.py
git mv face_detection_dl.py src/trymylook/face_detection.py
git mv makeup_application_dl.py src/trymylook/makeup_application.py

echo "🚚 Moving tests and assets ..."
git mv test_system.py tests/test_system.py
git mv sample.jpg assets/sample.jpg

echo "🧪 Rewriting tests/test_system.py to match current API + new package paths ..."
cat > tests/test_system.py << 'TESTEOF'
"""
System Test Script
Verifies all components are working correctly.

Run from the project root:
    python3 tests/test_system.py
"""

import sys
import traceback

import numpy as np


def test_package_imports():
    """Test if all required third-party packages can be imported"""
    print("Testing package imports...")
    try:
        import cv2
        print("  [OK] opencv-python imported successfully")

        import numpy
        print("  [OK] numpy imported successfully")

        from PIL import Image
        print("  [OK] Pillow imported successfully")

        import streamlit
        print("  [OK] streamlit imported successfully")

        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_opencv_cascades():
    """Test if OpenCV Haar Cascades are available"""
    print("\nTesting OpenCV Haar Cascades...")
    try:
        import cv2

        for name in ("haarcascade_frontalface_default.xml", "haarcascade_eye.xml", "haarcascade_smile.xml"):
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if cascade.empty():
                print(f"  [FAIL] {name} failed to load")
                return False
            print(f"  [OK] {name} loaded")

        return True
    except Exception as e:
        print(f"  [FAIL] Cascade test failed: {e}")
        return False


def test_custom_modules():
    """Test if the trymylook package modules can be imported"""
    print("\nTesting trymylook package modules...")
    try:
        from trymylook import config
        print("  [OK] trymylook.config imported")

        from trymylook.face_detection import DeepLearningFaceDetector
        print("  [OK] trymylook.face_detection imported")

        from trymylook.segmentation import HybridSegmenter
        print("  [OK] trymylook.segmentation imported")

        from trymylook.makeup_application import NeuralMakeupApplicator
        print("  [OK] trymylook.makeup_application imported")

        from trymylook import utils
        print("  [OK] trymylook.utils imported")

        return True
    except ImportError as e:
        print(f"  [FAIL] Module import failed: {e}")
        traceback.print_exc()
        return False


def test_face_detection():
    """Test face detection functionality"""
    print("\nTesting face detection...")
    try:
        from trymylook.face_detection import DeepLearningFaceDetector

        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detector = DeepLearningFaceDetector(device="cpu")
        print("  [OK] DeepLearningFaceDetector initialized")

        # Random noise won't contain a real face -- should return None, not crash.
        result = detector.detect_all(test_image)
        print(f"  [OK] detect_all executed (no errors); face found: {result is not None}")

        return True
    except Exception as e:
        print(f"  [FAIL] Face detection test failed: {e}")
        traceback.print_exc()
        return False


def test_segmentation():
    """Test segmentation functionality (landmark fallback path)"""
    print("\nTesting segmentation...")
    try:
        from trymylook.segmentation import HybridSegmenter

        segmenter = HybridSegmenter(device="cpu")
        print(f"  [OK] HybridSegmenter initialized (BiSeNet active: {segmenter.use_bisenet})")

        image_shape = (480, 640, 3)
        # No landmarks available here -- exercises the shape-only fallback path.
        mask = segmenter.create_mask_for_product(image_shape, "Lipstick", landmarks=None)
        print(f"  [OK] Mask created, shape={getattr(mask, 'shape', None)}")

        return True
    except Exception as e:
        print(f"  [FAIL] Segmentation test failed: {e}")
        traceback.print_exc()
        return False


def test_makeup_application():
    """Test makeup application functionality"""
    print("\nTesting makeup application...")
    try:
        import cv2
        from trymylook.makeup_application import NeuralMakeupApplicator

        applicator = NeuralMakeupApplicator()
        print("  [OK] NeuralMakeupApplicator initialized")

        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        test_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(test_mask, (320, 240), 50, 255, -1)

        applicator.apply_lipstick(test_image, test_mask, (220, 20, 60), 50)
        print("  [OK] Lipstick application executed")

        applicator.apply_eyeshadow(test_image, test_mask, (139, 90, 60), 50)
        print("  [OK] Eyeshadow application executed")

        applicator.apply_foundation(test_image, test_mask, (245, 222, 179), 50)
        print("  [OK] Foundation application executed")

        applicator.apply_blush(test_image, test_mask, (255, 182, 193), 50)
        print("  [OK] Blush application executed")

        return True
    except Exception as e:
        print(f"  [FAIL] Makeup application test failed: {e}")
        traceback.print_exc()
        return False


def test_utility_functions():
    """Test utility functions"""
    print("\nTesting utility functions...")
    try:
        from PIL import Image
        from trymylook.utils import (
            pil_to_cv, cv_to_pil, resize_image,
            create_side_by_side, validate_image
        )

        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_array)

        cv_img = pil_to_cv(pil_img)
        print("  [OK] PIL to CV conversion")

        cv_to_pil(cv_img)
        print("  [OK] CV to PIL conversion")

        resize_image(test_array, (50, 50))
        print("  [OK] Image resize")

        validate_image(test_array)
        print("  [OK] Image validation")

        create_side_by_side(test_array, test_array)
        print("  [OK] Side-by-side comparison")

        return True
    except Exception as e:
        print(f"  [FAIL] Utility functions test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("TRYMYLOOK SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("Package Imports", test_package_imports),
        ("OpenCV Cascades", test_opencv_cascades),
        ("Custom Modules", test_custom_modules),
        ("Face Detection", test_face_detection),
        ("Segmentation", test_segmentation),
        ("Makeup Application", test_makeup_application),
        ("Utility Functions", test_utility_functions),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[CRASH] Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {test_name}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\nAll tests passed! System is ready.")
        print("\nRun the app with: streamlit run app.py")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please fix issues before running the app.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
TESTEOF

echo "📦 Creating package __init__.py ..."
cat > src/trymylook/__init__.py << 'EOF'
"""
TryMyLook — Virtual Makeup Try-On

Applies realistic virtual makeup (foundation, blush, eyeshadow,
lipstick) to a face photo using deep-learning face detection and
BiSeNet face parsing, with a landmark-based fallback when BiSeNet
weights are unavailable.
"""

__version__ = "2.1.0"
EOF

echo "🔧 Patching segmentation.py model path for new layout ..."
python3 - << 'PYEOF'
import re

path = "src/trymylook/segmentation.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_path_block = (
    "# Add face-parsing.PyTorch to path\n"
    "FACE_PARSING_PATH = os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch')\n"
    "if FACE_PARSING_PATH not in sys.path:\n"
    "    sys.path.insert(0, FACE_PARSING_PATH)\n"
)

new_path_block = (
    "# Add face-parsing.PyTorch to path\n"
    "# Layout: <root>/src/trymylook/segmentation.py -> <root>/models/face-parsing.PyTorch\n"
    "PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))\n"
    "FACE_PARSING_PATH = os.path.join(PROJECT_ROOT, 'models', 'face-parsing.PyTorch')\n"
    "DEFAULT_BISENET_WEIGHTS = os.path.join(FACE_PARSING_PATH, 'res', 'cp', '79999_iter.pth')\n\n"
    "if FACE_PARSING_PATH not in sys.path:\n"
    "    sys.path.insert(0, FACE_PARSING_PATH)\n"
)

if old_path_block not in content:
    raise SystemExit("ERROR: expected FACE_PARSING_PATH block not found — aborting patch, please patch manually.")

content = content.replace(old_path_block, new_path_block)
content = content.replace(
    "model_path: str = 'face-parsing.PyTorch/res/cp/79999_iter.pth'",
    "model_path: str = DEFAULT_BISENET_WEIGHTS",
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("   segmentation.py patched OK")
PYEOF

echo "📝 Writing new pyproject.toml (editable install) ..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "trymylook"
version = "2.1.0"
description = "Virtual makeup try-on (Streamlit + PyTorch + BiSeNet)"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
EOF

echo "📝 Writing .streamlit/config.toml ..."
cat > .streamlit/config.toml << 'EOF'
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[theme]
base = "light"
EOF

echo "📝 Updating packages.txt ..."
cat > packages.txt << 'EOF'
libgl1
libglib2.0-0
EOF

echo "📝 Updating requirements.txt (opencv-headless + editable install) ..."
if grep -q "^opencv-python>=" requirements.txt; then
  sed -i 's/^opencv-python>=/opencv-python-headless>=/' requirements.txt
fi
# Ensure file ends with a newline before appending (original file may not)
[ -n "$(tail -c1 requirements.txt)" ] && echo "" >> requirements.txt
if ! grep -q "^-e \." requirements.txt; then
  echo "-e ." >> requirements.txt
fi

echo "📝 Rewriting app.py import header to use the new package ..."
python3 - << 'PYEOF'
path = "app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_imports = (
    "from config import (\n"
    "    APP_TITLE, APP_ICON, VERSION, \n"
    "    PRODUCTS, get_shades_for_product, \n"
    "    DEFAULT_INTENSITY, MIN_INTENSITY, MAX_INTENSITY,\n"
    "    MAX_IMAGE_SIZE, MIN_IMAGE_SIZE\n"
    ")\n"
    "from face_detection_dl import DeepLearningFaceDetector\n"
    "from segmentation import HybridSegmenter  # ✅ Changed to unified module\n"
    "from makeup_application_dl import NeuralMakeupApplicator\n"
    "from utils import (\n"
    "    pil_to_cv, cv_to_pil, resize_image, ensure_min_size,\n"
    "    create_side_by_side, validate_image, get_image_info\n"
    ")\n"
)

new_imports = (
    "from trymylook.config import (\n"
    "    APP_TITLE, APP_ICON, VERSION,\n"
    "    PRODUCTS, get_shades_for_product,\n"
    "    DEFAULT_INTENSITY, MIN_INTENSITY, MAX_INTENSITY,\n"
    "    MAX_IMAGE_SIZE, MIN_IMAGE_SIZE\n"
    ")\n"
    "from trymylook.face_detection import DeepLearningFaceDetector\n"
    "from trymylook.segmentation import HybridSegmenter\n"
    "from trymylook.makeup_application import NeuralMakeupApplicator\n"
    "from trymylook.utils import (\n"
    "    pil_to_cv, cv_to_pil, resize_image, ensure_min_size,\n"
    "    create_side_by_side, validate_image, get_image_info\n"
    ")\n"
)

if old_imports not in content:
    raise SystemExit("ERROR: app.py import block didn't match exactly — aborting patch, please patch manually.")

content = content.replace(old_imports, new_imports)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("   app.py imports patched OK")
PYEOF

echo "📝 Updating .devcontainer/devcontainer.json install command ..."
python3 - << 'PYEOF'
import json

path = ".devcontainer/devcontainer.json"
with open(path, "r", encoding="utf-8") as f:
    raw = f.read()

# devcontainer.json has // comments which aren't valid strict JSON;
# strip simple // line comments before parsing.
import re
no_comments = re.sub(r'(?m)^\s*//.*$', '', raw)
data = json.loads(no_comments)

data["updateContentCommand"] = (
    "[ -f packages.txt ] && sudo apt-get update && sudo xargs -a packages.txt apt-get install -y; "
    "pip3 install --user -r requirements.txt; "
    "echo '✅ Packages installed and Requirements met'"
)

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print("   devcontainer.json updated OK")
PYEOF

echo "📝 Writing README.md ..."
cat > README.md << 'EOF'
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
EOF

echo ""
echo "✅ Restructure complete."
echo "👉 Next steps:"
echo "   1. Review changes:  git status"
echo "   2. Reinstall:       pip3 install --user -r requirements.txt"
echo "   3. Run:             streamlit run app.py"