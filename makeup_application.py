"""
Makeup Application Module
Applies virtual makeup to segmented regions with realistic blending
"""

import cv2
import numpy as np

class MakeupApplicator:
    def __init__(self):
        pass
    
    def apply_makeup(self, image, mask, color_rgb, intensity, blend_mode='multiply'):
        """
        Apply makeup to image using mask and color
        
        Args:
            image: BGR image (OpenCV format)
            mask: Binary mask (0-255)
            color_rgb: Tuple (R, G, B) 0-255
            intensity: Float 0-100
            blend_mode: str ('multiply', 'overlay', 'normal')
            
        Returns:
            Image with makeup applied (BGR)
        """
        # Convert intensity to 0-1 range
        alpha = intensity / 100.0
        
        # Ensure mask is float and normalized
        mask_float = mask.astype(np.float32) / 255.0
        
        # Create colored overlay
        colored_overlay = self._create_colored_overlay(image, color_rgb, mask_float)
        
        # Blend based on mode
        if blend_mode == 'multiply':
            result = self._blend_multiply(image, colored_overlay, mask_float, alpha)
        elif blend_mode == 'overlay':
            result = self._blend_overlay(image, colored_overlay, mask_float, alpha)
        else:  # normal
            result = self._blend_normal(image, colored_overlay, mask_float, alpha)
        
        return result
    
    def _create_colored_overlay(self, image, color_rgb, mask):
        """
        Create colored overlay preserving luminance
        
        Args:
            image: BGR image
            color_rgb: (R, G, B) tuple
            mask: Normalized mask (0-1)
            
        Returns:
            Colored overlay (BGR)
        """
        overlay = image.copy().astype(np.float32)
        
        # Convert to LAB color space to preserve luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Convert RGB to BGR for OpenCV
        b, g, r = color_rgb[2], color_rgb[1], color_rgb[0]
        
        # Apply color where mask is active
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask) + b * mask
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask) + g * mask
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - mask) + r * mask
        
        # Preserve some original texture
        overlay = overlay * 0.7 + image.astype(np.float32) * 0.3
        
        return overlay.astype(np.uint8)
    
    def _blend_normal(self, base, overlay, mask, alpha):
        """
        Normal alpha blending
        
        Args:
            base: Base image (BGR)
            overlay: Colored overlay (BGR)
            mask: Normalized mask (0-1)
            alpha: Blend strength (0-1)
            
        Returns:
            Blended image (BGR)
        """
        base_float = base.astype(np.float32)
        overlay_float = overlay.astype(np.float32)
        
        # Expand mask to 3 channels
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Alpha blend
        blended = base_float * (1 - alpha * mask_3d) + overlay_float * (alpha * mask_3d)
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def _blend_multiply(self, base, overlay, mask, alpha):
        """
        Multiply blending (darkens - good for lipstick, eyeshadow)
        
        Args:
            base: Base image (BGR)
            overlay: Colored overlay (BGR)
            mask: Normalized mask (0-1)
            alpha: Blend strength (0-1)
            
        Returns:
            Blended image (BGR)
        """
        base_float = base.astype(np.float32) / 255.0
        overlay_float = overlay.astype(np.float32) / 255.0
        
        # Multiply blend
        multiplied = base_float * overlay_float
        
        # Convert back to 0-255 range
        multiplied = (multiplied * 255).astype(np.float32)
        
        # Apply with alpha and mask
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result = base.astype(np.float32) * (1 - alpha * mask_3d) + multiplied * (alpha * mask_3d)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _blend_overlay(self, base, overlay, mask, alpha):
        """
        Overlay blending (preserves highlights and shadows)
        
        Args:
            base: Base image (BGR)
            overlay: Colored overlay (BGR)
            mask: Normalized mask (0-1)
            alpha: Blend strength (0-1)
            
        Returns:
            Blended image (BGR)
        """
        base_float = base.astype(np.float32) / 255.0
        overlay_float = overlay.astype(np.float32) / 255.0
        
        # Overlay formula
        overlayed = np.where(
            base_float < 0.5,
            2 * base_float * overlay_float,
            1 - 2 * (1 - base_float) * (1 - overlay_float)
        )
        
        # Convert back to 0-255 range
        overlayed = (overlayed * 255).astype(np.float32)
        
        # Apply with alpha and mask
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result = base.astype(np.float32) * (1 - alpha * mask_3d) + overlayed * (alpha * mask_3d)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_lipstick(self, image, mask, color_rgb, intensity):
        """
        Apply lipstick with multiply blend
        
        Args:
            image: BGR image
            mask: Lip mask
            color_rgb: (R, G, B) tuple
            intensity: 0-100
            
        Returns:
            Image with lipstick applied
        """
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='multiply')
    
    def apply_eyeshadow(self, image, mask, color_rgb, intensity):
        """
        Apply eyeshadow with multiply blend
        
        Args:
            image: BGR image
            mask: Eye mask
            color_rgb: (R, G, B) tuple
            intensity: 0-100
            
        Returns:
            Image with eyeshadow applied
        """
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='multiply')
    
    def apply_foundation(self, image, mask, color_rgb, intensity):
        """
        Apply foundation with overlay blend
        
        Args:
            image: BGR image
            mask: Skin mask
            color_rgb: (R, G, B) tuple
            intensity: 0-100
            
        Returns:
            Image with foundation applied
        """
        return self.apply_makeup(image, mask, color_rgb, intensity, blend_mode='overlay')
    
    def adjust_color_temperature(self, image, mask, temperature):
        """
        Adjust color temperature in masked region
        
        Args:
            image: BGR image
            mask: Binary mask
            temperature: -100 to 100 (negative=cool, positive=warm)
            
        Returns:
            Image with adjusted temperature
        """
        result = image.copy().astype(np.float32)
        mask_float = (mask.astype(np.float32) / 255.0)
        
        if temperature > 0:  # Warm
            # Increase red, decrease blue
            result[:, :, 2] += temperature * mask_float  # Red
            result[:, :, 0] -= temperature * 0.5 * mask_float  # Blue
        else:  # Cool
            # Increase blue, decrease red
            result[:, :, 0] -= temperature * mask_float  # Blue
            result[:, :, 2] += temperature * 0.5 * mask_float  # Red
        
        return np.clip(result, 0, 255).astype(np.uint8)