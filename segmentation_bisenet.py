"""
BiSeNet Face Parsing Segmentation Module
Uses face-parsing.PyTorch for accurate face segmentation + landmark fallback
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
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def segment(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]
        input_tensor = self.preprocess_image(image)
        with torch.no_grad():
            out = self.net(input_tensor)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        parsing = cv2.resize(parsing.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return parsing

    def create_skin_mask(self, parsing: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        mask[parsing == self.FACE_CLASSES['skin']] = 255
        mask[parsing == self.FACE_CLASSES['neck']] = 255
        return cv2.GaussianBlur(mask, (31, 31), 0)

    def create_lip_mask(self, parsing: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        mask[parsing == self.FACE_CLASSES['upper_lip']] = 255
        mask[parsing == self.FACE_CLASSES['lower_lip']] = 255
        return cv2.GaussianBlur(mask, (15, 15), 0)

    def create_eye_mask(self, parsing: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        left_eye = (parsing == self.FACE_CLASSES['left_eye'])
        right_eye = (parsing == self.FACE_CLASSES['right_eye'])
        left_eye_mask = (left_eye.astype(np.uint8) * 255)
        right_eye_mask = (right_eye.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=1)
        right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=1)
        mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
        return cv2.GaussianBlur(mask, (21, 21), 0)

    def create_eyebrow_mask(self, parsing: np.ndarray) -> np.ndarray:
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        mask[parsing == self.FACE_CLASSES['left_eyebrow']] = 255
        mask[parsing == self.FACE_CLASSES['right_eyebrow']] = 255
        return cv2.GaussianBlur(mask, (11, 11), 0)

    def create_mask_for_product(self, image: np.ndarray, product_name: str) -> np.ndarray:
        parsing = self.segment(image)
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
        colors = {
            0: [0, 0, 0],
            1: [255, 0, 0],
            2: [0, 0, 255],
            3: [0, 255, 0],
            4: [255, 0, 255],
            5: [255, 255, 0],
            10: [128, 128, 0],
            12: [0, 255, 255],
            13: [139, 0, 0],
            14: [0, 128, 0],
            17: [128, 0, 128],
        }
        h, w = parsing.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            vis[parsing == class_id] = color
        return vis


# -------------------------------------------------------------------
# ✅ HYBRID SEGMENTER (BiSeNet + Landmarks fallback)
# -------------------------------------------------------------------

class HybridSegmenter:
    """
    Combines BiSeNet with landmark-based fallback.
    Uses BiSeNet by default; falls back to landmark-based masks if needed.
    """

    def __init__(self, model_path: str = 'face-parsing.PyTorch/res/cp/79999_iter.pth',
                 device: str = 'cpu'):
        print("🔮 Initializing Hybrid Segmenter...")
        from segmentation_dl import NeuralSegmenter  # your fallback model

        self.device = device
        self.fallback_segmenter = NeuralSegmenter()  # ✅ always defined

        try:
            self.bisenet = BiSeNetSegmenter(model_path, device)
            self.use_bisenet = True
            print("✅ Using BiSeNet for segmentation")
        except Exception as e:
            print(f"⚠️ BiSeNet not available: {e}")
            print("   Falling back to landmark-based segmentation")
            self.bisenet = None
            self.use_bisenet = False

    def create_mask_for_product(self, image_or_shape, product_name: str, landmarks=None) -> np.ndarray:
        """
        Create mask using BiSeNet if available, otherwise use landmarks.
        """
        # ✅ Use BiSeNet if it’s working
        if self.use_bisenet and self.bisenet is not None:
            try:
                if isinstance(image_or_shape, np.ndarray):
                    return self.bisenet.create_mask_for_product(image_or_shape, product_name)
                else:
                    print("⚠️ BiSeNet requires full image — switching to landmark-based segmentation.")
                    return self.fallback_segmenter.create_mask_for_product(
                        image_or_shape, product_name, landmarks
                    )
            except Exception as e:
                print(f"⚠️ BiSeNet failed during mask creation: {e}")
                print("   Falling back to landmark-based segmentation.")
                return self.fallback_segmenter.create_mask_for_product(
                    image_or_shape, product_name, landmarks
                )

        # ✅ If BiSeNet not available, always use fallback
        return self.fallback_segmenter.create_mask_for_product(
            image_or_shape, product_name, landmarks
        )


# -------------------------------------------------------------------
# ✅ STANDALONE TEST
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("BISENET SEGMENTATION - STANDALONE TEST")
    print("=" * 70)

    try:
        segmenter = BiSeNetSegmenter(device='cpu')
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        parsing = segmenter.segment(test_image)
        print(f"✅ Parsing map created: {parsing.shape}, unique classes: {np.unique(parsing)}")

        for product in ["Foundation", "Lipstick", "Eyeshadow"]:
            mask = segmenter.create_mask_for_product(test_image, product)
            print(f"   {product}: {np.count_nonzero(mask)} active pixels")

        print("✅ BiSeNet test completed successfully!")

    except Exception as e:
        print(f"❌ BiSeNet test failed: {e}")
        print("   Make sure face-parsing.PyTorch is properly set up")

    print("=" * 70)
