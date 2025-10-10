"""
Configuration file for Trymylook Virtual Makeup
Contains all product shades, constants, and settings
"""

import numpy as np

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_TITLE = "✨ Trymylook - Virtual Makeup Try-On"
APP_ICON = "💄"
VERSION = "1.0.0"

# Image processing settings
MAX_IMAGE_SIZE = (1920, 1920)  # Max width, height
MIN_IMAGE_SIZE = (400, 400)    # Min width, height

# Detection settings
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)

# Blending settings
DEFAULT_INTENSITY = 60
MIN_INTENSITY = 0
MAX_INTENSITY = 100

# ============================================================================
# PRODUCT SHADE DEFINITIONS (RGB VALUES)
# ============================================================================

LIPSTICK_SHADES = {
    "Classic Red": (220, 20, 60),
    "Pink Bliss": (255, 105, 180),
    "Nude Beige": (210, 150, 130),
    "Coral Pop": (255, 127, 80),
    "Berry Wine": (150, 50, 80),
    "Mauve Dream": (224, 176, 255),
    "Deep Wine": (128, 0, 32),
}

EYESHADOW_SHADES = {
    "Warm Brown": (139, 90, 60),
    "Golden Shimmer": (218, 165, 32),
    "Purple Haze": (147, 112, 219),
    "Neutral Taupe": (169, 149, 137),
    "Silver Frost": (192, 192, 192),
    "Blue Velvet": (100, 149, 237),
}

FOUNDATION_SHADES = {
    "Fair Porcelain": (255, 228, 196),
    "Light Beige": (245, 222, 179),
    "Medium Tan": (222, 184, 135),
    "Deep Caramel": (205, 133, 63),
    "Rich Mocha": (139, 90, 60),
    "Olive Undertone": (195, 176, 145),
}

# Product categories
PRODUCTS = ["Lipstick", "Eyeshadow", "Foundation"]

# ============================================================================
# OPENCV CASCADE PATHS (built into opencv-python)
# ============================================================================

CASCADE_FACE = "haarcascade_frontalface_default.xml"
CASCADE_EYE = "haarcascade_eye.xml"
CASCADE_MOUTH = "haarcascade_smile.xml"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_shades_for_product(product_name):
    """Get available shades for a given product"""
    if product_name == "Lipstick":
        return LIPSTICK_SHADES
    elif product_name == "Eyeshadow":
        return EYESHADOW_SHADES
    elif product_name == "Foundation":
        return FOUNDATION_SHADES
    else:
        return {}

def rgb_to_bgr(rgb_tuple):
    """Convert RGB to BGR for OpenCV"""
    return (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])

def bgr_to_rgb(bgr_tuple):
    """Convert BGR to RGB for display"""
    return (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])