"""
System Test Script
Verifies all components are working correctly
"""

import sys
import traceback

def test_package_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    try:
        import cv2
        print("  ✓ opencv-python imported successfully")
        
        import numpy
        print("  ✓ numpy imported successfully")
        
        from PIL import Image
        print("  ✓ Pillow imported successfully")
        
        import streamlit
        print("  ✓ streamlit imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_opencv_cascades():
    """Test if OpenCV Haar Cascades are available"""
    print("\nTesting OpenCV Haar Cascades...")
    try:
        import cv2
        
        # Test face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if face_cascade.empty():
            print("  ✗ Face cascade failed to load")
            return False
        print("  ✓ Face cascade loaded")
        
        # Test eye cascade
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        if eye_cascade.empty():
            print("  ✗ Eye cascade failed to load")
            return False
        print("  ✓ Eye cascade loaded")
        
        # Test smile cascade
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        if smile_cascade.empty():
            print("  ✗ Smile cascade failed to load")
            return False
        print("  ✓ Smile cascade loaded")
        
        return True
    except Exception as e:
        print(f"  ✗ Cascade test failed: {e}")
        return False

def test_custom_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    try:
        import config
        print("  ✓ config.py imported")
        
        from face_detection import FaceDetector
        print("  ✓ face_detection.py imported")
        
        from segmentation_dl import MakeupSegmenter
        print("  ✓ segmentation.py imported")
        
        from makeup_application import MakeupApplicator
        print("  ✓ makeup_application.py imported")
        
        import utils
        print("  ✓ utils.py imported")
        
        return True
    except ImportError as e:
        print(f"  ✗ Module import failed: {e}")
        traceback.print_exc()
        return False

def test_face_detection():
    """Test face detection functionality"""
    print("\nTesting face detection...")
    try:
        import cv2
        import numpy as np
        from face_detection import FaceDetector
        
        # Create test image with a simple face-like structure
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        print("  ✓ FaceDetector initialized")
        
        # Test detection (may not find face in random image, but should not crash)
        result = detector.detect_face(test_image)
        print("  ✓ Face detection executed (no errors)")
        
        return True
    except Exception as e:
        print(f"  ✗ Face detection test failed: {e}")
        traceback.print_exc()
        return False

def test_segmentation():
    """Test segmentation functionality"""
    print("\nTesting segmentation...")
    try:
        import numpy as np
        from segmentation_dl import MakeupSegmenter
        
        segmenter = MakeupSegmenter()
        print("  ✓ MakeupSegmenter initialized")
        
        # Create dummy face data
        face_data = {
            'face': {'bbox': (100, 100, 200, 250)},
            'eyes': [
                {'bbox': (150, 150, 40, 20)},
                {'bbox': (210, 150, 40, 20)}
            ],
            'mouth': {'bbox': (170, 280, 60, 30)}
        }
        
        # Test mask creation
        image_shape = (480, 640, 3)
        
        lip_mask = segmenter.create_lip_mask(image_shape, face_data['mouth'])
        print("  ✓ Lip mask created")
        
        eye_mask = segmenter.create_eye_mask(image_shape, face_data['eyes'])
        print("  ✓ Eye mask created")
        
        skin_mask = segmenter.create_skin_mask(
            image_shape,
            face_data['face'],
            face_data['eyes'],
            face_data['mouth']
        )
        print("  ✓ Skin mask created")
        
        return True
    except Exception as e:
        print(f"  ✗ Segmentation test failed: {e}")
        traceback.print_exc()
        return False

def test_makeup_application():
    """Test makeup application functionality"""
    print("\nTesting makeup application...")
    try:
        import cv2
        import numpy as np
        from makeup_application import MakeupApplicator
        
        applicator = MakeupApplicator()
        print("  ✓ MakeupApplicator initialized")
        
        # Create test image and mask
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        test_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(test_mask, (320, 240), 50, 255, -1)
        
        # Test lipstick application
        result = applicator.apply_lipstick(
            test_image,
            test_mask,
            (220, 20, 60),
            50
        )
        print("  ✓ Lipstick application executed")
        
        # Test eyeshadow application
        result = applicator.apply_eyeshadow(
            test_image,
            test_mask,
            (139, 90, 60),
            50
        )
        print("  ✓ Eyeshadow application executed")
        
        # Test foundation application
        result = applicator.apply_foundation(
            test_image,
            test_mask,
            (245, 222, 179),
            50
        )
        print("  ✓ Foundation application executed")
        
        return True
    except Exception as e:
        print(f"  ✗ Makeup application test failed: {e}")
        traceback.print_exc()
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\nTesting utility functions...")
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from utils import (
            pil_to_cv, cv_to_pil, resize_image,
            create_side_by_side, validate_image
        )
        
        # Test image conversion
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_array)
        
        cv_img = pil_to_cv(pil_img)
        print("  ✓ PIL to CV conversion")
        
        back_to_pil = cv_to_pil(cv_img)
        print("  ✓ CV to PIL conversion")
        
        # Test resize
        resized = resize_image(test_array, (50, 50))
        print("  ✓ Image resize")
        
        # Test validation
        is_valid, msg = validate_image(test_array)
        print("  ✓ Image validation")
        
        # Test side-by-side
        combined = create_side_by_side(test_array, test_array)
        print("  ✓ Side-by-side comparison")
        
        return True
    except Exception as e:
        print(f"  ✗ Utility functions test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TRYMYLOOK SYSTEM TEST")
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
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print("\nRun the app with: streamlit run app.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before running the app.")
        return 1

if __name__ == "__main__":
    sys.exit(main())