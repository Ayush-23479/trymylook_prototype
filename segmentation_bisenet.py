"""
BiSeNet Face Parsing Segmentation Module
Uses face-parsing.PyTorch for accurate face segmentation
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
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
    
    # Face-parsing class indices
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
        """
        Initialize BiSeNet segmenter
        
        Args:
            model_path: Path to pretrained BiSeNet model
            device: 'cpu' or 'cuda'
        """
        print(f"🔮 Initializing BiSeNet Face Parser...")
        print(f"   Device: {device.upper()}")
        
        if BiSeNet is None:
            raise ImportError("BiSeNet model not available. Check face-parsing.PyTorch installation.")
        
        self.device = device
        
        # Initialize model
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
        """
        Preprocess image for BiSeNet
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 512x512 (BiSeNet input size)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor and transpose to CHW format
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform face segmentation
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Segmentation map (H, W) with class indices 0-18
        """
        original_h, original_w = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            out = self.net(input_tensor)[0]  # Get main output
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        
        # Resize back to original size
        parsing = cv2.resize(
            parsing.astype(np.uint8), 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return parsing
    
    def create_skin_mask(self, parsing: np.ndarray) -> np.ndarray:
        """
        Create skin mask from parsing map
        
        Args:
            parsing: Segmentation map from BiSeNet
            
        Returns:
            Binary mask for skin region
        """
        # Combine skin, neck (exclude face features)
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        
        # Skin regions
        mask[parsing == self.FACE_CLASSES['skin']] = 255
        mask[parsing == self.FACE_CLASSES['neck']] = 255
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    def create_lip_mask(self, parsing: np.ndarray) -> np.ndarray:
        """
        Create lip mask from parsing map
        
        Args:
            parsing: Segmentation map from BiSeNet
            
        Returns:
            Binary mask for lips
        """
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        
        # Upper and lower lips
        mask[parsing == self.FACE_CLASSES['upper_lip']] = 255
        mask[parsing == self.FACE_CLASSES['lower_lip']] = 255
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def create_eye_mask(self, parsing: np.ndarray) -> np.ndarray:
        """
        Create eye/eyeshadow mask from parsing map
        
        Args:
            parsing: Segmentation map from BiSeNet
            
        Returns:
            Binary mask for eye region (for eyeshadow)
        """
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        
        # Get eye regions
        left_eye = (parsing == self.FACE_CLASSES['left_eye'])
        right_eye = (parsing == self.FACE_CLASSES['right_eye'])
        
        # Dilate eye regions to create eyeshadow area
        left_eye_mask = left_eye.astype(np.uint8) * 255
        right_eye_mask = right_eye.astype(np.uint8) * 255
        
        # Dilate to extend above eyes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=1)
        right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=1)
        
        mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def create_eyebrow_mask(self, parsing: np.ndarray) -> np.ndarray:
        """
        Create eyebrow mask from parsing map
        
        Args:
            parsing: Segmentation map from BiSeNet
            
        Returns:
            Binary mask for eyebrows
        """
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        
        mask[parsing == self.FACE_CLASSES['left_eyebrow']] = 255
        mask[parsing == self.FACE_CLASSES['right_eyebrow']] = 255
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return mask
    
    def create_mask_for_product(self, image: np.ndarray, product_name: str) -> np.ndarray:
        """
        Create mask for specific makeup product using BiSeNet
        
        Args:
            image: Input image (BGR)
            product_name: Product name ("Foundation", "Lipstick", "Eyeshadow", etc.)
            
        Returns:
            Binary mask for the product
        """
        # Get face parsing
        parsing = self.segment(image)
        
        # Create appropriate mask based on product
        if product_name == "Foundation":
            return self.create_skin_mask(parsing)
        elif product_name == "Lipstick":
            return self.create_lip_mask(parsing)
        elif product_name == "Eyeshadow":
            return self.create_eye_mask(parsing)
        elif product_name == "Eyebrow":
            return self.create_eyebrow_mask(parsing)
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def visualize_parsing(self, parsing: np.ndarray) -> np.ndarray:
        """
        Visualize face parsing with colors
        
        Args:
            parsing: Segmentation map
            
        Returns:
            Colored visualization (BGR)
        """
        # Color map for visualization
        colors = {
            0: [0, 0, 0],           # background - black
            1: [255, 0, 0],         # skin - red
            2: [0, 0, 255],         # left eyebrow - blue
            3: [0, 255, 0],         # right eyebrow - green
            4: [255, 0, 255],       # left eye - magenta
            5: [255, 255, 0],       # right eye - yellow
            10: [128, 128, 0],      # nose - olive
            12: [0, 255, 255],      # upper lip - cyan
            13: [139, 0, 0],        # lower lip - dark red
            14: [0, 128, 0],        # neck - dark green
            17: [128, 0, 128],      # hair - purple
        }
        
        h, w = parsing.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            vis[parsing == class_id] = color
        
        return vis


class HybridSegmenter:
    """
    Combines BiSeNet with landmark-based fallback
    """
    
    def __init__(self, model_path: str = 'face-parsing.PyTorch/res/cp/79999_iter.pth',
                 device: str = 'cpu'):
        """
        Initialize hybrid segmenter
        """
        try:
            self.bisenet = BiSeNetSegmenter(model_path, device)
            self.use_bisenet = True
            print("✅ Using BiSeNet for segmentation")
        except Exception as e:
            print(f"⚠️ BiSeNet not available: {e}")
            print("   Falling back to landmark-based segmentation")
            self.use_bisenet = False
            from segmentation_dl import NeuralSegmenter
            self.fallback_segmenter = NeuralSegmenter()
    
    def create_mask_for_product(self, image_or_shape, product_name: str, 
                                landmarks=None) -> np.ndarray:
        """
        Create mask using BiSeNet if available, otherwise use landmarks
        
        Args:
            image_or_shape: Either image (ndarray) or image shape (tuple)
            product_name: Product name
            landmarks: Facial landmarks (for fallback only)
            
        Returns:
            Binary mask
        """
        if self.use_bisenet:
            # BiSeNet needs the actual image
            if isinstance(image_or_shape, np.ndarray):
                return self.bisenet.create_mask_for_product(image_or_shape, product_name)
            else:
                print("⚠️ BiSeNet requires image, falling back to landmarks")
                return self.fallback_segmenter.create_mask_for_product(
                    image_or_shape, product_name, landmarks
                )
        else:
            # Landmark-based fallback
            return self.fallback_segmenter.create_mask_for_product(
                image_or_shape, product_name, landmarks
            )


if __name__ == "__main__":
    print("=" * 70)
    print("BISENET SEGMENTATION - STANDALONE TEST")
    print("=" * 70)
    
    # Test BiSeNet
    try:
        segmenter = BiSeNetSegmenter(device='cpu')
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\n🔍 Testing segmentation...")
        parsing = segmenter.segment(test_image)
        print(f"✅ Parsing map created: {parsing.shape}")
        print(f"   Unique classes: {np.unique(parsing)}")
        
        print("\n🎨 Testing mask creation...")
        for product in ["Foundation", "Lipstick", "Eyeshadow"]:
            mask = segmenter.create_mask_for_product(test_image, product)
            non_zero = np.count_nonzero(mask)
            print(f"   {product}: {mask.shape}, {non_zero} active pixels")
        
        print("\n✅ BiSeNet test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ BiSeNet test failed: {e}")
        print("   Make sure face-parsing.PyTorch is properly set up")
    
    print("\n" + "=" * 70)