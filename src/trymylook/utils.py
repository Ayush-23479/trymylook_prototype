"""
Utility Functions
Helper functions for image processing and display
"""

import cv2
import numpy as np
from PIL import Image
import io


def resize_image(image, max_size=(1920, 1920)):
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    scale = min(max_w / w, max_h / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def ensure_min_size(image, min_size=(400, 400)):
    h, w = image.shape[:2]
    min_w, min_h = min_size
    
    if w < min_w or h < min_h:
        scale = max(min_w / w, min_h / h)
        
        if scale > 2.0:
            return None
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return image


def pil_to_cv(pil_image):
    rgb_array = np.array(pil_image.convert('RGB'))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv_to_pil(cv_image):
    rgb_array = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_array)
    return pil_image


def create_side_by_side(image1, image2, labels=None):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    target_h = max(h1, h2)
    
    if h1 != target_h:
        scale = target_h / h1
        image1 = cv2.resize(image1, (int(w1 * scale), target_h))
    
    if h2 != target_h:
        scale = target_h / h2
        image2 = cv2.resize(image2, (int(w2 * scale), target_h))
    
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        text_size1 = cv2.getTextSize(labels[0], font, font_scale, thickness)[0]
        text_x1 = (image1.shape[1] - text_size1[0]) // 2
        cv2.putText(image1, labels[0], (text_x1, 30), font, font_scale, color, thickness)
        
        text_size2 = cv2.getTextSize(labels[1], font, font_scale, thickness)[0]
        text_x2 = (image2.shape[1] - text_size2[0]) // 2
        cv2.putText(image2, labels[1], (text_x2, 30), font, font_scale, color, thickness)
    
    combined = np.hstack([image1, image2])
    
    return combined


def apply_brightness_adjustment(image, brightness=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_contrast_adjustment(image, contrast=1.0):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return adjusted


def sharpen_image(image, amount=1.0):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) * amount / 9
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def smooth_skin(image, mask, amount=15):
    if amount % 2 == 0:
        amount += 1
    
    smoothed = cv2.bilateralFilter(image, amount, 75, 75)
    
    mask_float = (mask.astype(np.float32) / 255.0)
    mask_3d = np.stack([mask_float, mask_float, mask_float], axis=2)
    
    result = image.astype(np.float32) * (1 - mask_3d) + smoothed.astype(np.float32) * mask_3d
    
    return result.astype(np.uint8)


def draw_detection_boxes(image, face_data, color=(0, 255, 0)):
    result = image.copy()
    
    if face_data is None:
        return result
    
    if 'face' in face_data and face_data['face']:
        x, y, w, h = face_data['face']['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if 'eyes' in face_data:
        for i, eye in enumerate(face_data['eyes']):
            x, y, w, h = eye['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, f'Eye {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    if 'mouth' in face_data and face_data['mouth']:
        x, y, w, h = face_data['mouth']['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result, 'Mouth', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return result


def create_color_swatch(color_rgb, size=(50, 50)):
    swatch = np.ones((size[1], size[0], 3), dtype=np.uint8)
    swatch[:, :] = color_rgb
    return swatch


def validate_image(image):
    if image is None:
        return False, "No image provided"
    
    if len(image.shape) != 3:
        return False, "Image must be color (3 channels)"
    
    h, w = image.shape[:2]
    
    if w < 200 or h < 200:
        return False, "Image too small (minimum 200x200 pixels)"
    
    if w > 4000 or h > 4000:
        return False, "Image too large (maximum 4000x4000 pixels)"
    
    return True, "Valid image"


def get_image_info(image):
    if image is None:
        return {}
    
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_mb': image.nbytes / (1024 * 1024)
    }


def auto_color_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def enhance_colors(image, saturation=1.2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_vignette(image, strength=0.5):
    rows, cols = image.shape[:2]
    
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    
    mask = kernel / kernel.max()
    mask = mask ** strength
    
    result = image.copy().astype(np.float32)
    
    for i in range(3):
        result[:, :, i] = result[:, :, i] * mask
    
    return result.astype(np.uint8)


def crop_to_face(image, face_bbox, padding=0.2):
    x, y, w, h = face_bbox
    
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    
    cropped = image[y1:y2, x1:x2]
    
    return cropped


def blend_images(base, overlay, alpha=0.5):
    blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    return blended


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("UTILS MODULE - STANDALONE TEST")
    print("=" * 70)
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n📐 Testing image operations...")
    
    resized = resize_image(test_image, (320, 240))
    print(f"✅ Resize: {test_image.shape} → {resized.shape}")
    
    is_valid, msg = validate_image(test_image)
    print(f"✅ Validation: {is_valid} - {msg}")
    
    info = get_image_info(test_image)
    print(f"✅ Image info: {info['width']}x{info['height']}, {info['size_mb']:.2f} MB")
    
    print("\n🎨 Testing image adjustments...")
    
    bright = apply_brightness_adjustment(test_image, 20)
    print(f"✅ Brightness adjusted: {bright.shape}")
    
    contrast = apply_contrast_adjustment(test_image, 1.2)
    print(f"✅ Contrast adjusted: {contrast.shape}")
    
    sharp = sharpen_image(test_image, 1.0)
    print(f"✅ Sharpened: {sharp.shape}")
    
    print("\n🖼️  Testing composition...")
    
    side_by_side = create_side_by_side(test_image, test_image, labels=("Before", "After"))
    print(f"✅ Side-by-side: {side_by_side.shape}")
    
    print("\n✨ Testing enhancements...")
    
    balanced = auto_color_balance(test_image)
    print(f"✅ Color balanced: {balanced.shape}")
    
    enhanced = enhance_colors(test_image, 1.3)
    print(f"✅ Colors enhanced: {enhanced.shape}")
    
    clahe_result = apply_clahe(test_image)
    print(f"✅ CLAHE applied: {clahe_result.shape}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)