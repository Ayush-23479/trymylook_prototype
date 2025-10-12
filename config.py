"""
Configuration file for Trymylook Virtual Makeup
Contains all product shades, constants, and settings
"""

import numpy as np


APP_TITLE = "✨ Trymylook - Virtual Makeup Try-On"
APP_ICON = "💄"
VERSION = "2.0.0"


MAX_IMAGE_SIZE = (1920, 1920)
MIN_IMAGE_SIZE = (400, 400)


HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)


DEFAULT_INTENSITY = 70
MIN_INTENSITY = 0
MAX_INTENSITY = 100


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


PRODUCTS = ["Lipstick", "Eyeshadow", "Foundation"]


CASCADE_FACE = "haarcascade_frontalface_default.xml"
CASCADE_EYE = "haarcascade_eye.xml"
CASCADE_MOUTH = "haarcascade_smile.xml"


def get_shades_for_product(product_name):
    if product_name == "Lipstick":
        return LIPSTICK_SHADES
    elif product_name == "Eyeshadow":
        return EYESHADOW_SHADES
    elif product_name == "Foundation":
        return FOUNDATION_SHADES
    else:
        return {}


def rgb_to_bgr(rgb_tuple):
    return (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])


def bgr_to_rgb(bgr_tuple):
    return (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])


def get_all_products():
    return PRODUCTS


def get_total_shades():
    return len(LIPSTICK_SHADES) + len(EYESHADOW_SHADES) + len(FOUNDATION_SHADES)


def get_product_info(product_name):
    shades = get_shades_for_product(product_name)
    return {
        'name': product_name,
        'shades': list(shades.keys()),
        'shade_count': len(shades),
        'colors': list(shades.values())
    }


def validate_product(product_name):
    return product_name in PRODUCTS


def validate_shade(product_name, shade_name):
    shades = get_shades_for_product(product_name)
    return shade_name in shades


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIG MODULE - STANDALONE TEST")
    print("=" * 70)
    
    print(f"\n📦 Application: {APP_TITLE}")
    print(f"   Version: {VERSION}")
    print(f"   Icon: {APP_ICON}")
    
    print(f"\n🎨 Available Products: {len(PRODUCTS)}")
    for product in PRODUCTS:
        info = get_product_info(product)
        print(f"   • {product}: {info['shade_count']} shades")
    
    print(f"\n💄 Lipstick Shades ({len(LIPSTICK_SHADES)}):")
    for shade, rgb in LIPSTICK_SHADES.items():
        print(f"   • {shade}: RGB{rgb}")
    
    print(f"\n👁️  Eyeshadow Shades ({len(EYESHADOW_SHADES)}):")
    for shade, rgb in EYESHADOW_SHADES.items():
        print(f"   • {shade}: RGB{rgb}")
    
    print(f"\n🎭 Foundation Shades ({len(FOUNDATION_SHADES)}):")
    for shade, rgb in FOUNDATION_SHADES.items():
        print(f"   • {shade}: RGB{rgb}")
    
    print(f"\n📊 Total Shades: {get_total_shades()}")
    
    print(f"\n⚙️  Settings:")
    print(f"   Max Image Size: {MAX_IMAGE_SIZE}")
    print(f"   Min Image Size: {MIN_IMAGE_SIZE}")
    print(f"   Default Intensity: {DEFAULT_INTENSITY}%")
    print(f"   Intensity Range: {MIN_INTENSITY}% - {MAX_INTENSITY}%")
    
    print("\n🧪 Testing helper functions...")
    
    test_product = "Lipstick"
    test_shade = "Classic Red"
    
    print(f"   Product '{test_product}' valid: {validate_product(test_product)}")
    print(f"   Shade '{test_shade}' valid: {validate_shade(test_product, test_shade)}")
    
    rgb = (255, 0, 0)
    bgr = rgb_to_bgr(rgb)
    print(f"   RGB{rgb} → BGR{bgr}")
    
    back_to_rgb = bgr_to_rgb(bgr)
    print(f"   BGR{bgr} → RGB{back_to_rgb}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)