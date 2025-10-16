"""
BiSeNet Face Parsing Segmentation Module
Uses face-parsing.PyTorch for accurate face segmentation + landmark fallback
FIXED: Added blush mask support and optimized Complete Look processing
"""

import cv2
import numpy as np
import torch
import sys
import os

# Add face-parsing.PyTorch to path
FACE_PARSING_PATH = os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch')
if FACE_PARSING_PATH not in sys.path:
    sys.path.insert(0, FACE_PARSING_PATH)

try:
    from model import BiSeNet
except ImportError:
    print("⚠️ Warning: BiSeNet model not found. Make sure face-parsing.PyTorch is in the project directory.")
    BiSeNet = None


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

        if BiSeNet is None:
            raise ImportError("BiSeNet model not available. Check face-parsing.PyTorch installation.")

        self.device = device
        self.net = BiSeNet(n_classes=19)
        self.net.to(device)

        # Load pretrained weights
        if os.path.exists(model_path):
            print(f"   Loading model from: {model_path}")
            self.net.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ BiSeNet model loaded successfully")
        else:
            print(f"⚠️ Model not found at: {model_path}")
            print("   Please ensure the model file exists")
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
        """
        Segment face into 19 classes
        Returns: parsing map (H x W) with class IDs
        """
        original_h, original_w = image.shape[:2]
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            out = self.net(input_tensor)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        
        # Resize back to original dimensions
        parsing = cv2.resize(
            parsing.astype(np.uint8), 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        )
        return parsing

    def create_skin_mask(self, parsing: np.ndarray) -> np.ndarray:
        """Create foundation mask from skin + neck regions"""
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        mask[parsing == self.FACE_CLASSES['skin']] = 255
        mask[parsing == self.FACE_CLASSES['neck']] = 255
        
        # Smooth edges for natural blending
        return cv2.GaussianBlur(mask, (31, 31), 0)

    def create_lip_mask(self, parsing: np.ndarray) -> np.ndarray:
        """Create lipstick mask from upper + lower lip regions"""
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        mask[parsing == self.FACE_CLASSES['upper_lip']] = 255
        mask[parsing == self.FACE_CLASSES['lower_lip']] = 255
        
        # Light blur for precise lip application
        return cv2.GaussianBlur(mask, (15, 15), 0)

    def create_eye_mask(self, parsing: np.ndarray) -> np.ndarray:
        """Create eyeshadow mask by expanding eye regions"""
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        
        # Get left and right eye masks
        left_eye = (parsing == self.FACE_CLASSES['left_eye'])
        right_eye = (parsing == self.FACE_CLASSES['right_eye'])
        
        left_eye_mask = (left_eye.astype(np.uint8) * 255)
        right_eye_mask = (right_eye.astype(np.uint8) * 255)
        
        # Expand eye regions for eyeshadow coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=1)
        right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=1)
        
        # Combine both eyes
        mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
        
        # Smooth for natural eyeshadow blending
        return cv2.GaussianBlur(mask, (21, 21), 0)

    def create_eyebrow_mask(self, parsing: np.ndarray) -> np.ndarray:
        """Create eyebrow mask (optional feature)"""
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        mask[parsing == self.FACE_CLASSES['left_eyebrow']] = 255
        mask[parsing == self.FACE_CLASSES['right_eyebrow']] = 255
        
        return cv2.GaussianBlur(mask, (11, 11), 0)

    def create_blush_mask(self, parsing: np.ndarray) -> np.ndarray:
        """
        ✅ NEW: Create blush mask using skin regions on cheeks
        Uses morphological operations to isolate cheek areas
        """
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        
        # Start with full skin mask
        skin_mask = (parsing == self.FACE_CLASSES['skin']).astype(np.uint8) * 255
        
        # Find skin contours to locate cheeks
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the largest skin region (main face)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Define cheek regions (left and right sides, below eyes)
            cheek_y_start = int(y + h * 0.45)  # Start below eyes
            cheek_y_end = int(y + h * 0.75)    # End above mouth
            
            # Left cheek
            left_cheek_x = int(x + w * 0.15)
            left_cheek_w = int(w * 0.25)
            
            # Right cheek  
            right_cheek_x = int(x + w * 0.60)
            right_cheek_w = int(w * 0.25)
            
            # Create circular blush regions
            left_center = (left_cheek_x + left_cheek_w // 2, (cheek_y_start + cheek_y_end) // 2)
            right_center = (right_cheek_x + right_cheek_w // 2, (cheek_y_start + cheek_y_end) // 2)
            
            radius_x = int(w * 0.12)
            radius_y = int(h * 0.15)
            
            # Draw elliptical blush regions
            cv2.ellipse(mask, left_center, (radius_x, radius_y), -10, 0, 360, 255, -1)
            cv2.ellipse(mask, right_center, (radius_x, radius_y), 10, 0, 360, 255, -1)
            
            # Intersect with actual skin regions (avoid eyes, nose, lips)
            mask = cv2.bitwise_and(mask, skin_mask)
        
        # Heavy blur for natural blush effect
        return cv2.GaussianBlur(mask, (61, 61), 0)

    def create_mask_for_product(self, image: np.ndarray, product_name: str) -> np.ndarray:
        """
        Create mask for specific makeup product
        ✅ FIXED: Added Blush support
        """
        # Segment image first
        parsing = self.segment(image)
        
        # Return appropriate mask based on product
        if product_name == "Foundation":
            return self.create_skin_mask(parsing)
        elif product_name == "Lipstick":
            return self.create_lip_mask(parsing)
        elif product_name == "Eyeshadow":
            return self.create_eye_mask(parsing)
        elif product_name == "Eyebrow":
            return self.create_eyebrow_mask(parsing)
        elif product_name == "Blush":
            return self.create_blush_mask(parsing)
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def create_masks_batch(self, image: np.ndarray, product_names: list) -> dict:
        """
        ✅ NEW: Optimized method for Complete Look
        Segments image once and creates all masks
        """
        # Segment once
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
            0: [0, 0, 0],        # background
            1: [255, 0, 0],      # skin - red
            2: [0, 0, 255],      # left_eyebrow - blue
            3: [0, 255, 0],      # right_eyebrow - green
            4: [255, 0, 255],    # left_eye - magenta
            5: [255, 255, 0],    # right_eye - yellow
            10: [128, 128, 0],   # nose - olive
            12: [0, 255, 255],   # upper_lip - cyan
            13: [139, 0, 0],     # lower_lip - dark red
            14: [0, 128, 0],     # neck - dark green
            17: [128, 0, 128],   # hair - purple
        }
        
        h, w = parsing.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            vis[parsing == class_id] = color
        
        return vis


# -------------------------------------------------------------------
# ✅ HYBRID SEGMENTER (BiSeNet + Landmarks fallback) - FIXED
# -------------------------------------------------------------------

class HybridSegmenter:
    """
    Combines BiSeNet with landmark-based fallback.
    Uses BiSeNet by default; falls back to landmark-based masks if needed.
    ✅ FIXED: Proper blush handling and optimized Complete Look support
    """

    def __init__(self, model_path: str = 'face-parsing.PyTorch/res/cp/79999_iter.pth',
                 device: str = 'cpu'):
        print("🔮 Initializing Hybrid Segmenter...")
        
        # Always initialize fallback segmenter
        from segmentation_dl import NeuralSegmenter
        self.fallback_segmenter = NeuralSegmenter()
        
        # Try to initialize BiSeNet
        self.bisenet = None
        self.use_bisenet = False
        
        try:
            self.bisenet = BiSeNetSegmenter(model_path, device)
            self.use_bisenet = True
            print("✅ Using BiSeNet for segmentation")
        except Exception as e:
            print(f"⚠️ BiSeNet not available: {e}")
            print("   Falling back to landmark-based segmentation")
    
    def create_mask_for_product(self, image_or_shape, product_name: str, 
                                landmarks=None) -> np.ndarray:
        """
        Create mask using BiSeNet if available, otherwise use landmarks.
        ✅ FIXED: Proper handling of all products including Blush
        """
        # Use BiSeNet if available and we have a full image
        if self.use_bisenet and self.bisenet is not None:
            if isinstance(image_or_shape, np.ndarray) and len(image_or_shape.shape) == 3:
                try:
                    return self.bisenet.create_mask_for_product(image_or_shape, product_name)
                except Exception as e:
                    print(f"⚠️ BiSeNet failed: {e}. Using landmark fallback.")
                    # Fall through to landmark-based method
            else:
                print("⚠️ BiSeNet requires full image. Using landmark fallback.")
        
        # Fallback to landmark-based segmentation
        return self.fallback_segmenter.create_mask_for_product(
            image_or_shape, product_name, landmarks
        )
    
    def create_masks_batch(self, image: np.ndarray, product_names: list, landmarks=None):
        """
        ✅ NEW: Optimized batch mask creation for Complete Look
        Segments once with BiSeNet, or creates all landmark-based masks
        Returns: (masks_dict, parsing_map or None)
        """
        if self.use_bisenet and self.bisenet is not None:
            try:
                # Use BiSeNet's optimized batch method
                masks, parsing = self.bisenet.create_masks_batch(image, product_names)
                return masks, parsing
            except Exception as e:
                print(f"⚠️ BiSeNet batch failed: {e}. Using landmark fallback.")
        
        # Fallback: create masks individually with landmarks
        masks = {}
        for product_name in product_names:
            masks[product_name] = self.fallback_segmenter.create_mask_for_product(
                image, product_name, landmarks
            )
        
        return masks, None


# -------------------------------------------------------------------
# ✅ STANDALONE TEST
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("BISENET SEGMENTATION - STANDALONE TEST (FIXED)")
    print("=" * 70)

    try:
        segmenter = BiSeNetSegmenter(device='cpu')
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\n🔍 Testing segmentation...")
        parsing = segmenter.segment(test_image)
        print(f"✅ Parsing map created: {parsing.shape}")
        print(f"   Unique classes: {np.unique(parsing)}")
        
        print("\n🎨 Testing individual mask creation...")
        for product in ["Foundation", "Lipstick", "Eyeshadow", "Blush"]:
            mask = segmenter.create_mask_for_product(test_image, product)
            non_zero = np.count_nonzero(mask)
            print(f"   {product}: {mask.shape}, {non_zero} active pixels")
        
        print("\n🚀 Testing optimized batch mask creation...")
        products = ["Foundation", "Blush", "Eyeshadow", "Lipstick"]
        masks, parsing = segmenter.create_masks_batch(test_image, products)
        print(f"✅ Created {len(masks)} masks in single segmentation pass")
        for product, mask in masks.items():
            non_zero = np.count_nonzero(mask)
            print(f"   {product}: {non_zero} active pixels")
        
        print("\n✅ BiSeNet test completed successfully!")
        print("   ✓ Blush mask support added")
        print("   ✓ Batch processing optimized")
        print("   ✓ Complete Look ready")
        
    except Exception as e:
        print(f"\n❌ BiSeNet test failed: {e}")
        print("   Make sure face-parsing.PyTorch is properly set up")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)