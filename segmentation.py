"""
Segmentation Module
Creates pixel masks for makeup application regions
"""

import cv2
import numpy as np

class MakeupSegmenter:
    def __init__(self):
        pass
    
    def create_lip_mask(self, image_shape, mouth_region):
        """
        Create mask for lip region
        
        Args:
            image_shape: tuple (height, width, channels)
            mouth_region: dict with mouth coordinates
            
        Returns:
            Binary mask (numpy array)
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if mouth_region is None:
            return mask
        
        mx, my, mw, mh = mouth_region['bbox']
        
        # Create elliptical lip mask
        center_x = mx + mw // 2
        center_y = my + mh // 2
        
        # Slightly reduce the region for more natural look
        radius_x = int(mw * 0.45)
        radius_y = int(mh * 0.6)
        
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius_x, radius_y),
            0, 0, 360,
            255,
            -1
        )
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def create_eye_mask(self, image_shape, eye_regions):
        """
        Create mask for eye/eyelid region
        
        Args:
            image_shape: tuple (height, width, channels)
            eye_regions: list of eye dicts
            
        Returns:
            Binary mask (numpy array)
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if not eye_regions:
            return mask
        
        for eye in eye_regions:
            ex, ey, ew, eh = eye['bbox']
            
            # Create upper eyelid region for eyeshadow
            center_x = ex + ew // 2
            center_y = ey + eh // 4  # Upper part of eye
            
            # Extend upward for eyelid
            radius_x = int(ew * 0.65)
            radius_y = int(eh * 0.8)
            
            cv2.ellipse(
                mask,
                (center_x, center_y),
                (radius_x, radius_y),
                0, 0, 360,
                255,
                -1
            )
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def create_skin_mask(self, image_shape, face_region, eye_regions, mouth_region):
        """
        Create mask for facial skin (foundation)
        
        Args:
            image_shape: tuple (height, width, channels)
            face_region: dict with face coordinates
            eye_regions: list of eye dicts
            mouth_region: dict with mouth coordinates
            
        Returns:
            Binary mask (numpy array)
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if face_region is None:
            return mask
        
        fx, fy, fw, fh = face_region['bbox']
        
        # Create elliptical face mask
        center_x = fx + fw // 2
        center_y = fy + fh // 2
        
        radius_x = int(fw * 0.45)
        radius_y = int(fh * 0.50)
        
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius_x, radius_y),
            0, 0, 360,
            255,
            -1
        )
        
        # Exclude eyes from foundation mask
        if eye_regions:
            for eye in eye_regions:
                ex, ey, ew, eh = eye['bbox']
                cv2.ellipse(
                    mask,
                    (ex + ew//2, ey + eh//2),
                    (int(ew * 0.6), int(eh * 0.7)),
                    0, 0, 360,
                    0,  # Black (remove from mask)
                    -1
                )
        
        # Exclude mouth from foundation mask
        if mouth_region:
            mx, my, mw, mh = mouth_region['bbox']
            cv2.ellipse(
                mask,
                (mx + mw//2, my + mh//2),
                (int(mw * 0.5), int(mh * 0.6)),
                0, 0, 360,
                0,  # Black (remove from mask)
                -1
            )
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    def create_mask_for_product(self, image_shape, product_name, face_data):
        """
        Create appropriate mask based on product type
        
        Args:
            image_shape: tuple (height, width, channels)
            product_name: str ('Lipstick', 'Eyeshadow', 'Foundation')
            face_data: dict with detected facial features
            
        Returns:
            Binary mask (numpy array)
        """
        if face_data is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        if product_name == "Lipstick":
            return self.create_lip_mask(image_shape, face_data.get('mouth'))
        
        elif product_name == "Eyeshadow":
            return self.create_eye_mask(image_shape, face_data.get('eyes', []))
        
        elif product_name == "Foundation":
            return self.create_skin_mask(
                image_shape,
                face_data.get('face'),
                face_data.get('eyes', []),
                face_data.get('mouth')
            )
        
        else:
            return np.zeros(image_shape[:2], dtype=np.uint8)
    
    def refine_mask(self, mask, iterations=2):
        """
        Refine mask using morphological operations
        
        Args:
            mask: Binary mask
            iterations: Number of refinement iterations
            
        Returns:
            Refined mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask