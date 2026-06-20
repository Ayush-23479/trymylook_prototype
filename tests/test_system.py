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
