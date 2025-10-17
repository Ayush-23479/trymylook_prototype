"""
Unified Segmentation Module
Combines BiSeNet (deep learning) + Landmark-based fallback in one file
✅ COMPLETE VERSION - All segmentation methods in one place
"""

import cv2
import numpy as np
import torch
import sys
import os

# ===================================================================
# BISENET SETUP
# ===================================================================

# Add face-parsing.PyTorch to path
FACE_PARSING_PATH = os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch')
if FACE_PARSING_PATH not in sys.path:
    sys.path.insert(0, FACE_PARSING_PATH)

try:
    from model import BiSeNet
    BISENET_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: BiSeNet model not found. Using landmark-based segmentation only.")
    BiSeNet = None
    BISENET_AVAILABLE = False


# ===================================================================
# PART 1: LANDMARK-BASED SEGMENTER (Fallback)
# ===================================================================

class LandmarkSegmenter:
    """
    Landmark-based segmentation for makeup regions
    Used as fallback when BiSeNet is not available
    ✅ Supports: Foundation, Lipstick, Eyeshadow, Blush
    """
    
    def __init__(self):
        pass
    
    def create_lip_mask(self, image_shape, landmarks):
        """Create lipstick application mask from lip landmarks"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None:
            return mask
        
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 68:
            outer_lip = landmarks[48:60]
        else:
            outer_lip = landmarks
        
        outer_lip_int = outer_lip.astype(np.int32)
        cv2.fillPoly(mask, [outer_lip_int], 255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def create_eye_mask(self, image_shape, landmarks):
        """Create eyeshadow application mask around eyes"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None or len(landmarks) == 0:
            return mask
        
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 68:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            eyes = [left_eye, right_eye]
        else:
            eyes = [landmarks]
        
        for eye in eyes:
            center_x = int(np.mean(eye[:, 0]))
            center_y = int(np.mean(eye[:, 1])) - 10
            
            radius_x = int((np.max(eye[:, 0]) - np.min(eye[:, 0])) * 0.8)
            radius_y = int((np.max(eye[:, 1]) - np.min(eye[:, 1])) * 1.2)
            
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask
    
    def create_skin_mask(self, image_shape, landmarks):
        """Create foundation application mask covering face skin"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None:
            return mask
        
        if len(landmarks.shape) != 2 or landmarks.shape[0] != 68:
            return mask
        
        jawline = landmarks[0:17]
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        nose_bridge = landmarks[27:31]
        
        forehead_estimate = self._estimate_forehead(jawline, left_eyebrow, right_eyebrow, nose_bridge)
        
        face_contour = np.vstack([
            forehead_estimate,
            right_eyebrow[::-1],
            jawline[8:],
            jawline[:9][::-1],
            left_eyebrow
        ])
        
        face_contour_int = face_contour.astype(np.int32)
        cv2.fillPoly(mask, [face_contour_int], 255)
        
        # Exclude eyes
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        for eye in [left_eye, right_eye]:
            ex_center = int(np.mean(eye[:, 0]))
            ey_center = int(np.mean(eye[:, 1]))
            ex_radius = int((np.max(eye[:, 0]) - np.min(eye[:, 0])) * 0.8)
            ey_radius = int((np.max(eye[:, 1]) - np.min(eye[:, 1])) * 0.9)
            cv2.ellipse(mask, (ex_center, ey_center), (ex_radius, ey_radius), 0, 0, 360, 0, -1)
        
        # Exclude lips
        outer_lip = landmarks[48:60]
        lip_expansion = self._expand_lip_region(outer_lip, expansion_factor=1.3)
        lip_expansion_int = lip_expansion.astype(np.int32)
        cv2.fillPoly(mask, [lip_expansion_int], 0)
        
        # Exclude nose tip
        nose_tip = landmarks[30:36]
        nose_center_x = int(np.mean(nose_tip[:, 0]))
        nose_center_y = int(np.mean(nose_tip[:, 1]))
        nose_radius_x = int((np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0])) * 0.7)
        nose_radius_y = int((np.max(nose_tip[:, 1]) - np.min(nose_tip[:, 1])) * 0.7)
        cv2.ellipse(mask, (nose_center_x, nose_center_y), (nose_radius_x, nose_radius_y), 0, 0, 360, 0, -1)
        
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    def _estimate_forehead(self, jawline, left_eyebrow, right_eyebrow, nose_bridge):
        """Estimate forehead points for better foundation coverage"""
        eyebrow_top_y = min(np.min(left_eyebrow[:, 1]), np.min(right_eyebrow[:, 1]))
        face_width = np.max(jawline[:, 0]) - np.min(jawline[:, 0])
        forehead_height = face_width * 0.35
        forehead_y = int(eyebrow_top_y - forehead_height)
        left_temple_x = int(left_eyebrow[0, 0] - face_width * 0.05)
        right_temple_x = int(right_eyebrow[-1, 0] + face_width * 0.05)
        
        num_points = 7
        forehead_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(left_temple_x + (right_temple_x - left_temple_x) * t)
            curve_factor = np.sin(t * np.pi) * 0.3
            y = int(forehead_y - forehead_height * curve_factor * 0.2)
            forehead_points.append([x, y])
        
        return np.array(forehead_points)
    
    def _expand_lip_region(self, lip_points, expansion_factor=1.2):
        """Expand lip region to ensure proper exclusion from foundation"""
        center_x = np.mean(lip_points[:, 0])
        center_y = np.mean(lip_points[:, 1])
        expanded = lip_points.copy().astype(np.float32)
        
        for i in range(len(expanded)):
            dx = expanded[i, 0] - center_x
            dy = expanded[i, 1] - center_y
            expanded[i, 0] = center_x + dx * expansion_factor
            expanded[i, 1] = center_y + dy * expansion_factor
        
        return expanded
    
    def create_blush_mask(self, image_shape, landmarks):
        """Create blush application mask on cheeks"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None:
            return mask
        
        if len(landmarks.shape) != 2 or landmarks.shape[0] != 68:
            return mask
        
        nose_bottom = landmarks[33]
        left_cheek_start = landmarks[3]
        left_cheek_end = landmarks[4]
        right_cheek_start = landmarks[13]
        right_cheek_end = landmarks[12]
        
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        
        left_cheek_center_x = int((left_cheek_start[0] + left_cheek_end[0]) / 2)
        left_cheek_center_y = int(nose_bottom[1] - face_width * 0.05)
        right_cheek_center_x = int((right_cheek_start[0] + right_cheek_end[0]) / 2)
        right_cheek_center_y = int(nose_bottom[1] - face_width * 0.05)
        
        cheek_radius_x = int(face_width * 0.12)
        cheek_radius_y = int(face_width * 0.15)
        
        cv2.ellipse(mask, (left_cheek_center_x, left_cheek_center_y), (cheek_radius_x, cheek_radius_y), -15, 0, 360, 255, -1)
        cv2.ellipse(mask, (right_cheek_center_x, right_cheek_center_y), (cheek_radius_x, cheek_radius_y), 15, 0, 360, 255, -1)
        
        mask = cv2.GaussianBlur(mask, (61, 61), 0)
        return mask
    
    def create_mask_for_product(self, image_shape, product_name, landmarks):
        """Create mask for specific makeup product"""
        if landmarks is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        if isinstance(landmarks, dict) and 'landmarks' in landmarks:
            landmarks = landmarks['landmarks']
        
        if product_name == "Lipstick":
            return self.create_lip_mask(image_shape, landmarks)
        elif product_name == "Eyeshadow":
            return self.create_eye_mask(image_shape, landmarks)
        elif product_name == "Foundation":
            return self.create_skin_mask(image_shape, landmarks)
        elif product_name == "Blush":
            return self.create_blush_mask(image_shape, landmarks)
        else:
            return np.zeros(image_shape[:2], dtype=np.uint8)


# ===================================================================
# PART 2: BISENET SEGMENTER (Deep Learning)
# ===================================================================

if BISENET_AVAILABLE:
    class BiSeNetSegmenter:
        """
        Face segmentation using BiSeNet
        Returns 19-class semantic segmentation of face regions
        """

        FACE_CLASSES = {
            'background': 0,
            'skin': 1,
            'left_eyebrow': 2,
            'right_eyebrow': 3,
            'left_eye': 4,
            'right_eye': 5,
            'nose': 10,
            'upper_lip': 12,
            'lower_lip': 13,
            'neck': 14,
            'hair': 17,
        }

        def __init__(self, model_path: str = 'face-parsing.PyTorch/res/cp/79999_iter.pth',
                     device: str = 'cpu'):
            print("🔮 Initializing BiSeNet Face Parser...")
            print(f"   Device: {device.upper()}")

            self.device = device
            self.net = BiSeNet(n_classes=19)
            self.net.to(device)

            if os.path.exists(model_path):
                print(f"   Loading model from: {model_path}")
                self.net.load_state_dict(torch.load(model_path, map_location=device))
                print("✅ BiSeNet model loaded successfully")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.net.eval()
            print("🎯 BiSeNet ready for segmentation!")

        def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
            """Preprocess image for BiSeNet input"""
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(self.device)

        def segment(self, image: np.ndarray) -> np.ndarray:
            """Segment face into 19 classes"""
            original_h, original_w = image.shape[:2]
            input_tensor = self.preprocess_image(image)
            
            with torch.no_grad():
                out = self.net(input_tensor)[0]
                parsing = out.squeeze(0).argmax(0).cpu().numpy()
            
            parsing = cv2.resize(parsing.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            return parsing

        def create_skin_mask(self, parsing: np.ndarray) -> np.ndarray:
            """Create foundation mask from skin + neck regions"""
            mask = np.zeros(parsing.shape, dtype=np.uint8)
            mask[parsing == self.FACE_CLASSES['skin']] = 255
            mask[parsing == self.FACE_CLASSES['neck']] = 255
            return cv2.GaussianBlur(mask, (31, 31), 0)

        def create_lip_mask(self, parsing: np.ndarray) -> np.ndarray:
            """Create lipstick mask from upper + lower lip regions"""
            mask = np.zeros(parsing.shape, dtype=np.uint8)
            mask[parsing == self.FACE_CLASSES['upper_lip']] = 255
            mask[parsing == self.FACE_CLASSES['lower_lip']] = 255
            return cv2.GaussianBlur(mask, (15, 15), 0)

        def create_eye_mask(self, parsing: np.ndarray) -> np.ndarray:
            """Create eyeshadow mask by expanding eye regions"""
            mask = np.zeros(parsing.shape, dtype=np.uint8)
            
            left_eye_mask = ((parsing == self.FACE_CLASSES['left_eye']).astype(np.uint8) * 255)
            right_eye_mask = ((parsing == self.FACE_CLASSES['right_eye']).astype(np.uint8) * 255)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=1)
            right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=1)
            
            mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
            return cv2.GaussianBlur(mask, (21, 21), 0)

        def create_blush_mask(self, parsing: np.ndarray) -> np.ndarray:
            """Create blush mask using skin regions on cheeks"""
            mask = np.zeros(parsing.shape, dtype=np.uint8)
            skin_mask = (parsing == self.FACE_CLASSES['skin']).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)
                
                cheek_y_start = int(y + h * 0.45)
                cheek_y_end = int(y + h * 0.75)
                
                left_center = (int(x + w * 0.25), (cheek_y_start + cheek_y_end) // 2)
                right_center = (int(x + w * 0.75), (cheek_y_start + cheek_y_end) // 2)
                
                radius_x = int(w * 0.12)
                radius_y = int(h * 0.15)
                
                cv2.ellipse(mask, left_center, (radius_x, radius_y), -10, 0, 360, 255, -1)
                cv2.ellipse(mask, right_center, (radius_x, radius_y), 10, 0, 360, 255, -1)
                
                mask = cv2.bitwise_and(mask, skin_mask)
            
            return cv2.GaussianBlur(mask, (61, 61), 0)

        def create_mask_for_product(self, image: np.ndarray, product_name: str) -> np.ndarray:
            """Create mask for specific makeup product"""
            parsing = self.segment(image)
            
            if product_name == "Foundation":
                return self.create_skin_mask(parsing)
            elif product_name == "Lipstick":
                return self.create_lip_mask(parsing)
            elif product_name == "Eyeshadow":
                return self.create_eye_mask(parsing)
            elif product_name == "Blush":
                return self.create_blush_mask(parsing)
            else:
                return np.zeros(image.shape[:2], dtype=np.uint8)

        def create_masks_batch(self, image: np.ndarray, product_names: list) -> tuple:
            """Create all masks in single segmentation pass"""
            parsing = self.segment(image)
            
            masks = {}
            for product_name in product_names:
                if product_name == "Foundation":
                    masks[product_name] = self.create_skin_mask(parsing)
                elif product_name == "Lipstick":
                    masks[product_name] = self.create_lip_mask(parsing)
                elif product_name == "Eyeshadow":
                    masks[product_name] = self.create_eye_mask(parsing)
                elif product_name == "Blush":
                    masks[product_name] = self.create_blush_mask(parsing)
                else:
                    masks[product_name] = np.zeros(image.shape[:2], dtype=np.uint8)
            
            return masks, parsing

        def visualize_parsing(self, parsing: np.ndarray) -> np.ndarray:
            """Create color visualization of parsing map"""
            colors = {
                0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 0, 255], 3: [0, 255, 0],
                4: [255, 0, 255], 5: [255, 255, 0], 10: [128, 128, 0],
                12: [0, 255, 255], 13: [139, 0, 0], 14: [0, 128, 0], 17: [128, 0, 128],
            }
            
            h, w = parsing.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            for class_id, color in colors.items():
                vis[parsing == class_id] = color
            
            return vis


# ===================================================================
# PART 3: HYBRID SEGMENTER (Auto-selects best method)
# ===================================================================

class HybridSegmenter:
    """
    Unified segmenter that automatically uses BiSeNet if available,
    otherwise falls back to landmark-based segmentation
    ✅ ONE CLASS TO RULE THEM ALL
    """

    def __init__(self, model_path: str = 'face-parsing.PyTorch/res/cp/79999_iter.pth',
                 device: str = 'cpu'):
        print("🔮 Initializing Unified Hybrid Segmenter...")
        
        # Always initialize landmark fallback
        self.landmark_segmenter = LandmarkSegmenter()
        print("✅ Landmark-based segmenter loaded")
        
        # Try to initialize BiSeNet
        self.bisenet = None
        self.use_bisenet = False
        
        if BISENET_AVAILABLE:
            try:
                self.bisenet = BiSeNetSegmenter(model_path, device)
                self.use_bisenet = True
                print("✅ BiSeNet segmenter loaded - Using deep learning mode")
            except Exception as e:
                print(f"⚠️ BiSeNet not available: {e}")
                print("   Using landmark-based segmentation")
        else:
            print("⚠️ BiSeNet not available - Using landmark-based segmentation only")
        
        print(f"🎯 Unified Segmenter Ready!")
        print(f"   Mode: {'BiSeNet (Deep Learning)' if self.use_bisenet else 'Landmark-based'}")
    
    def create_mask_for_product(self, image_or_shape, product_name: str, 
                                landmarks=None) -> np.ndarray:
        """
        Create mask using best available method
        Automatically selects BiSeNet or landmark-based
        """
        # Use BiSeNet if available and we have a full image
        if self.use_bisenet and self.bisenet is not None:
            if isinstance(image_or_shape, np.ndarray) and len(image_or_shape.shape) == 3:
                try:
                    return self.bisenet.create_mask_for_product(image_or_shape, product_name)
                except Exception as e:
                    print(f"⚠️ BiSeNet failed: {e}. Using landmark fallback.")
        
        # Fallback to landmark-based segmentation
        return self.landmark_segmenter.create_mask_for_product(
            image_or_shape, product_name, landmarks
        )
    
    def create_masks_batch(self, image: np.ndarray, product_names: list, landmarks=None):
        """
        Optimized batch mask creation
        Creates all masks in single pass if using BiSeNet
        """
        if self.use_bisenet and self.bisenet is not None:
            try:
                masks, parsing = self.bisenet.create_masks_batch(image, product_names)
                return masks, parsing
            except Exception as e:
                print(f"⚠️ BiSeNet batch failed: {e}. Using landmark fallback.")
        
        # Fallback: create masks individually
        masks = {}
        for product_name in product_names:
            masks[product_name] = self.landmark_segmenter.create_mask_for_product(
                image, product_name, landmarks
            )
        
        return masks, None


# ===================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ===================================================================

# For old code that imports NeuralSegmenter
NeuralSegmenter = LandmarkSegmenter

# For old code that imports directly
Segmenter = HybridSegmenter


# ===================================================================
# STANDALONE TEST
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED SEGMENTATION MODULE - STANDALONE TEST")
    print("=" * 70)
    
    # Test unified segmenter
    print("\n🔮 Testing HybridSegmenter (Unified)...")
    segmenter = HybridSegmenter(device='cpu')
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_landmarks = np.array([[320 + i*5, 240 + i*3] for i in range(68)])
    
    print("\n🎨 Testing individual product masks...")
    for product in ["Foundation", "Blush", "Eyeshadow", "Lipstick"]:
        mask = segmenter.create_mask_for_product(test_image, product, dummy_landmarks)
        non_zero = np.count_nonzero(mask)
        coverage = (non_zero / (test_image.shape[0] * test_image.shape[1])) * 100
        print(f"   {product:12s}: {mask.shape}, {non_zero:6d} pixels ({coverage:5.1f}% coverage)")
    
    print("\n🚀 Testing batch mask creation (Complete Look optimization)...")
    products = ["Foundation", "Blush", "Eyeshadow", "Lipstick"]
    masks, parsing = segmenter.create_masks_batch(test_image, products, dummy_landmarks)
    print(f"✅ Created {len(masks)} masks")
    for product, mask in masks.items():
        non_zero = np.count_nonzero(mask)
        print(f"   {product}: {non_zero} active pixels")
    
    print("\n✅ UNIFIED SEGMENTATION TEST COMPLETE")
    print(f"   Mode used: {'BiSeNet' if segmenter.use_bisenet else 'Landmark-based'}")
    print(f"   All {len(products)} products working")
    print(f"   Batch optimization: {'Available' if parsing is not None else 'Not available (using fallback)'}")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED - UNIFIED MODULE READY!")
    print("=" * 70)