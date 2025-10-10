"""
Utility Functions
Helper functions for image processing and display
"""

import cv2
import numpy as np
from PIL import Image
import io

def resize_image(image, max_size=(1920, 1920)):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: numpy array (BGR or RGB)
        max_size: tuple (max_width, max_height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    # Calculate scaling factor
    scale = min(max_w / w, max_h / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def ensure_min_size(image, min_size=(400, 400)):
    """
    Ensure image meets minimum size requirements
    
    Args:
        image: numpy array
        min_size: tuple (min_width, min_height)
        
    Returns:
        Image meeting minimum size or None if too small to upscale
    """
    h, w = image.shape[:2]
    min_w, min_h = min_size
    
    if w < min_w or h < min_h:
        scale = max(min_w / w, min_h / h)
        
        # Don't upscale more than 2x (quality degradation)
        if scale > 2.0:
            return None
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return image

def pil_to_cv(pil_image):
    """
    Convert PIL Image to OpenCV format (BGR)
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        numpy array (BGR)
    """
    # Convert PIL to RGB numpy array
    rgb_array = np.array(pil_image.convert('RGB'))
    
    # Convert RGB to BGR for OpenCV
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array

def cv_to_pil(cv_image):
    """
    Convert OpenCV image (BGR) to PIL Image
    
    Args:
        cv_image: numpy array (BGR)
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_array = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_array)
    
    return pil_image

def create_side_by_side(image1, image2, labels=None):
    """
    Create side-by-side comparison of two images
    
    Args:
        image1: First image (BGR)
        image2: Second image (BGR)
        labels: Optional tuple of (label1, label2)
        
    Returns:
        Combined image (BGR)
    """
    # Ensure images are same size
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    target_h = max(h1, h2)
    
    if h1 != target_h:
        scale = target_h / h1
        image1 = cv2.resize(image1, (int(w1 * scale), target_h))
    
    if h2 != target_h:
        scale = target_h / h2
        image2 = cv2.resize(image2, (int(w2 * scale), target_h))
    
    # Add labels if provided
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        # Add label to image1
        text_size1 = cv2.getTextSize(labels[0], font, font_scale, thickness)[0]
        text_x1 = (image1.shape[1] - text_size1[0]) // 2
        cv2.putText(image1, labels[0], (text_x1, 30), font, font_scale, color, thickness)
        
        # Add label to image2
        text_size2 = cv2.getTextSize(labels[1], font, font_scale, thickness)[0]
        text_x2 = (image2.shape[1] - text_size2[0]) // 2
        cv2.putText(image2, labels[1], (text_x2, 30), font, font_scale, color, thickness)
    
    # Concatenate horizontally
    combined = np.hstack([image1, image2])
    
    return combined

def apply_brightness_adjustment(image, brightness=0):
    """
    Adjust image brightness
    
    Args:
        image: BGR image
        brightness: -100 to 100
        
    Returns:
        Adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_contrast_adjustment(image, contrast=1.0):
    """
    Adjust image contrast
    
    Args:
        image: BGR image
        contrast: 0.5 to 2.0 (1.0 = no change)
        
    Returns:
        Adjusted image
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return adjusted

def sharpen_image(image, amount=1.0):
    """
    Sharpen image
    
    Args:
        image: BGR image
        amount: 0.0 to 2.0
        
    Returns:
        Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) * amount / 9
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def smooth_skin(image, mask, amount=15):
    """
    Apply skin smoothing effect
    
    Args:
        image: BGR image
        mask: Skin mask
        amount: Blur amount (odd number)
        
    Returns:
        Smoothed image
    """
    if amount % 2 == 0:
        amount += 1
    
    # Bilateral filter for skin smoothing
    smoothed = cv2.bilateralFilter(image, amount, 75, 75)
    
    # Apply only to masked regions
    mask_float = (mask.astype(np.float32) / 255.0)
    mask_3d = np.stack([mask_float, mask_float, mask_float], axis=2)
    
    result = image.astype(np.float32) * (1 - mask_3d) + smoothed.astype(np.float32) * mask_3d
    
    return result.astype(np.uint8)

def draw_detection_boxes(image, face_data, color=(0, 255, 0)):
    """
    Draw bounding boxes for detected regions (debugging)
    
    Args:
        image: BGR image
        face_data: Dict with detected regions
        color: Box color (BGR)
        
    Returns:
        Image with boxes drawn
    """
    result = image.copy()
    
    if face_data is None:
        return result
    
    # Draw face
    if 'face' in face_data and face_data['face']:
        x, y, w, h = face_data['face']['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw eyes
    if 'eyes' in face_data:
        for i, eye in enumerate(face_data['eyes']):
            x, y, w, h = eye['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, f'Eye {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Draw mouth
    if 'mouth' in face_data and face_data['mouth']:
        x, y, w, h = face_data['mouth']['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result, 'Mouth', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return result

def create_color_swatch(color_rgb, size=(50, 50)):
    """
    Create a color swatch image
    
    Args:
        color_rgb: (R, G, B) tuple
        size: (width, height)
        
    Returns:
        Color swatch image (RGB)
    """
    swatch = np.ones((size[1], size[0], 3), dtype=np.uint8)
    swatch[:, :] = color_rgb
    return swatch

def validate_image(image):
    """
    Validate if image is suitable for processing
    
    Args:
        image: numpy array
        
    Returns:
        tuple (is_valid, message)
    """
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
    """
    Get image information
    
    Args:
        image: numpy array
        
    Returns:
        dict with image info
    """
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