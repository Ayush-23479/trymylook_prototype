"""
Makeup Application Module
Applies virtual makeup to segmented regions with realistic blending
"""

import cv2
import numpy as np
from typing import Tuple


class NeuralMakeupApplicator:
    
    def __init__(self):
        pass
    
    def apply_makeup(self, image, mask, color_rgb, intensity, blend_mode='multiply'):
        alpha = intensity / 100.0
        
        mask_float = mask.astype(np.float32) / 255.0
        
        colored_overlay = self._create_colored_overlay(image, color_rgb, mask_float)
        
        if blend_mode == 'multiply':
            result = self._blend_multiply(image, colored_overlay, mask_float, alpha)
        elif blend_mode == 'overlay':
            result = self._blend_overlay(image, colored_overlay, mask_float, alpha)
        else:
            result = self._blend_normal(image, colored_overlay, mask_float, alpha)
        
        return result
    
    def _create_colored_overlay(self, image, color_rgb, mask):
        overlay = image.copy().astype(np.float32)
        
        b, g, r = color_rgb[2], color_rgb[1], color_rgb[0]
        
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask) + b * mask
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask) + g * mask
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - mask) + r * mask
        
        overlay = overlay * 0.7 + image.astype(np.float32) * 0.3
        
        return overlay.astype(np.uint8)
    
    def _blend_normal(self, base, overlay, mask, alpha):
        base_float = base.astype(np.float32)
        overlay_float = overlay.astype(np.float32)
        
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        blended = base_float * (1 - alpha * mask_3d) + overlay_float * (alpha * mask_3d)
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def _blend_multiply(self, base, overlay, mask, alpha):
        base_float = base.astype(np.float32) / 255.0
        overlay_float = overlay.astype(np.float32) / 255.0
        
        multiplied = base_float * overlay_float
        
        multiplied = (multiplied * 255).astype(np.float32)
        
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result = base.astype(np.float32) * (1 - alpha * mask_3d) + multiplied * (alpha * mask_3d)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_overlay(self, base, overlay, mask, alpha):
        base_float = base.astype(np.float32) / 255.0
        overlay_float = overlay.astype(np.float32) / 255.0
        
        overlayed = np.where(
            base_float < 0.5,
            2 * base_float * overlay_float,
            1 - 2 * (1 - base_float) * (1 - overlay_float)
        )
        
        overlayed = (overlayed * 255).astype(np.float32)
        
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result = base.astype(np.float32) * (1 - alpha * mask_3d) + overlayed * (alpha * mask_3d)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_screen(self, base, overlay, mask, alpha):
        base_float = base.astype(np.float32) / 255.0
        overlay_float = overlay.astype(np.float32) / 255.0
        
        screened = 1 - (1 - base_float) * (1 - overlay_float)
        screened = (screened * 255).astype(np.float32)
        
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result = base.astype(np.float32) * (1 - alpha * mask_3d) + screened * (alpha * mask_3d)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_lipstick(self, image, mask, color_rgb, intensity):
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='multiply')
    
    def apply_eyeshadow(self, image, mask, color_rgb, intensity):
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='multiply')
    
    def apply_foundation(self, image, mask, color_rgb, intensity):
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='overlay')
    
    def apply_blush(self, image, mask, color_rgb, intensity):
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='overlay')
    
    def apply_highlighter(self, image, mask, color_rgb, intensity):
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='screen')
    
    def adjust_color_temperature(self, image, mask, temperature):
        result = image.copy().astype(np.float32)
        mask_float = (mask.astype(np.float32) / 255.0)
        
        if temperature > 0:
            result[:, :, 2] += temperature * mask_float
            result[:, :, 0] -= temperature * 0.5 * mask_float
        else:
            result[:, :, 0] -= temperature * mask_float
            result[:, :, 2] += temperature * 0.5 * mask_float
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def adjust_saturation(self, image, mask, saturation_factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        mask_float = mask.astype(np.float32) / 255.0
        
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + (saturation_factor - 1) * mask_float)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def add_glossiness(self, image, mask, gloss_intensity):
        result = image.copy().astype(np.float32)
        
        mask_float = mask.astype(np.float32) / 255.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        highlight = np.where(gray > 180, 255, 0).astype(np.float32)
        
        gloss_effect = highlight * gloss_intensity / 100.0
        
        for i in range(3):
            result[:, :, i] += gloss_effect * mask_float
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def preserve_texture(self, original, processed, mask, amount=0.5):
        original_float = original.astype(np.float32)
        processed_float = processed.astype(np.float32)
        
        blur = cv2.GaussianBlur(original, (5, 5), 0).astype(np.float32)
        details = original_float - blur
        
        mask_3d = np.stack([mask, mask, mask], axis=2).astype(np.float32) / 255.0
        
        result = processed_float + (details * amount * mask_3d)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def adaptive_lighting(self, image, mask, makeup_color):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        brightness_map = gray / 255.0
        
        result = image.copy().astype(np.float32)
        mask_float = mask.astype(np.float32) / 255.0
        
        b, g, r = makeup_color[2], makeup_color[1], makeup_color[0]
        
        result[:, :, 0] = result[:, :, 0] * (1 - mask_float) + (b * brightness_map * mask_float)
        result[:, :, 1] = result[:, :, 1] * (1 - mask_float) + (g * brightness_map * mask_float)
        result[:, :, 2] = result[:, :, 2] * (1 - mask_float) + (r * brightness_map * mask_float)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def enhance_lips(self, image, mask, enhancement_level):
        result = image.copy()
        
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        mask_float = mask.astype(np.float32) / 255.0
        
        result_hsv[:, :, 1] = result_hsv[:, :, 1] * (1 + 0.3 * enhancement_level * mask_float)
        result_hsv[:, :, 2] = result_hsv[:, :, 2] * (1 + 0.2 * enhancement_level * mask_float)
        
        result_hsv[:, :, 1] = np.clip(result_hsv[:, :, 1], 0, 255)
        result_hsv[:, :, 2] = np.clip(result_hsv[:, :, 2], 0, 255)
        
        result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def smooth_application(self, image, mask, smoothing_level=5):
        if smoothing_level % 2 == 0:
            smoothing_level += 1
        
        smoothed = cv2.GaussianBlur(image, (smoothing_level, smoothing_level), 0)
        
        mask_3d = np.stack([mask, mask, mask], axis=2).astype(np.float32) / 255.0
        
        result = image.astype(np.float32) * (1 - mask_3d) + smoothed.astype(np.float32) * mask_3d
        
        return result.astype(np.uint8)


if __name__ == "__main__":
    print("=" * 70)
    print("MAKEUP APPLICATION MODULE - STANDALONE TEST")
    print("=" * 70)
    
    applicator = NeuralMakeupApplicator()
    
    test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(test_mask, (320, 240), 50, 255, -1)
    
    print("\n🎨 Testing makeup application...")
    
    red_color = (220, 20, 60)
    
    result = applicator.apply_lipstick(test_image, test_mask, red_color, 70)
    print(f"✅ Lipstick applied: {result.shape}")
    
    result = applicator.apply_eyeshadow(test_image, test_mask, (139, 90, 60), 60)
    print(f"✅ Eyeshadow applied: {result.shape}")
    
    result = applicator.apply_foundation(test_image, test_mask, (245, 222, 179), 50)
    print(f"✅ Foundation applied: {result.shape}")
    
    print("\n✨ Testing blend modes...")
    
    result_normal = applicator.apply_makeup(test_image, test_mask, red_color, 70, 'normal')
    print(f"   Normal blend: {result_normal.shape}")
    
    result_multiply = applicator.apply_makeup(test_image, test_mask, red_color, 70, 'multiply')
    print(f"   Multiply blend: {result_multiply.shape}")
    
    result_overlay = applicator.apply_makeup(test_image, test_mask, red_color, 70, 'overlay')
    print(f"   Overlay blend: {result_overlay.shape}")
    
    print("\n🌟 Testing enhancement features...")
    
    enhanced = applicator.enhance_lips(test_image, test_mask, 0.5)
    print(f"✅ Lips enhanced: {enhanced.shape}")
    
    glossy = applicator.add_glossiness(test_image, test_mask, 50)
    print(f"✅ Glossiness added: {glossy.shape}")
    
    textured = applicator.preserve_texture(test_image, result, test_mask, 0.5)
    print(f"✅ Texture preserved: {textured.shape}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)