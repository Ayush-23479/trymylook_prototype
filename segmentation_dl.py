"""
Segmentation Module
Creates pixel-perfect masks for makeup application regions
"""

import cv2
import numpy as np
from typing import Dict, Optional


class NeuralSegmenter:
    
    def __init__(self):
        pass
    
    def create_lip_mask(self, image_shape, landmarks):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None:
            return mask
        
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 68:
            outer_lip = landmarks[48:60]
        else:
            outer_lip = landmarks
        
        outer_lip_int = outer_lip.astype(np.int32)
        cv2.fillPoly(mask, [outer_lip_int], 255)
        
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def create_eye_mask(self, image_shape, landmarks):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None or len(landmarks) == 0:
            return mask
        
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 68:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            eyes = [left_eye, right_eye]
        else:
            eyes = [landmarks]
        
        for eye in eyes:
            center_x = int(np.mean(eye[:, 0]))
            center_y = int(np.mean(eye[:, 1])) - 10
            
            radius_x = int((np.max(eye[:, 0]) - np.min(eye[:, 0])) * 0.8)
            radius_y = int((np.max(eye[:, 1]) - np.min(eye[:, 1])) * 1.2)
            
            cv2.ellipse(
                mask,
                (center_x, center_y),
                (radius_x, radius_y),
                0, 0, 360,
                255,
                -1
            )
        
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def create_skin_mask(self, image_shape, landmarks):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if landmarks is None:
            return mask
        
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 68:
            face_points = landmarks
        else:
            return mask
        
        center_x = int(np.mean(face_points[:, 0]))
        center_y = int(np.mean(face_points[:, 1]))
        
        face_width = int(np.max(face_points[:, 0]) - np.min(face_points[:, 0]))
        face_height = int(np.max(face_points[:, 1]) - np.min(face_points[:, 1]))
        
        radius_x = int(face_width * 0.45)
        radius_y = int(face_height * 0.50)
        
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius_x, radius_y),
            0, 0, 360,
            255,
            -1
        )
        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        for eye in [left_eye, right_eye]:
            ex_center = int(np.mean(eye[:, 0]))
            ey_center = int(np.mean(eye[:, 1]))
            ex_radius = int((np.max(eye[:, 0]) - np.min(eye[:, 0])) * 0.7)
            ey_radius = int((np.max(eye[:, 1]) - np.min(eye[:, 1])) * 0.8)
            
            cv2.ellipse(
                mask,
                (ex_center, ey_center),
                (ex_radius, ey_radius),
                0, 0, 360,
                0,
                -1
            )
        
        outer_lip = landmarks[48:60]
        outer_lip_int = outer_lip.astype(np.int32)
        cv2.fillPoly(mask, [outer_lip_int], 0)
        
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    def create_mask_for_product(self, image_shape, product_name, landmarks):
        if landmarks is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        if isinstance(landmarks, dict) and 'landmarks' in landmarks:
            landmarks = landmarks['landmarks']
        
        if product_name == "Lipstick":
            return self.create_lip_mask(image_shape, landmarks)
        
        elif product_name == "Eyeshadow":
            return self.create_eye_mask(image_shape, landmarks)
        
        elif product_name == "Foundation":
            return self.create_skin_mask(image_shape, landmarks)
        
        else:
            return np.zeros(image_shape[:2], dtype=np.uint8)
    
    def refine_mask(self, mask, iterations=2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def dilate_mask(self, mask, kernel_size=5, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        return dilated
    
    def erode_mask(self, mask, kernel_size=5, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask, kernel, iterations=iterations)
        return eroded
    
    def smooth_mask_edges(self, mask, blur_amount=15):
        if blur_amount % 2 == 0:
            blur_amount += 1
        
        smoothed = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
        return smoothed
    
    def create_gradient_mask(self, image_shape, center, radius_inner, radius_outer):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        mask = np.clip((dist - radius_inner) / (radius_outer - radius_inner), 0, 1)
        mask = 1 - mask
        mask = (mask * 255).astype(np.uint8)
        
        return mask


if __name__ == "__main__":
    print("=" * 70)
    print("SEGMENTATION MODULE - STANDALONE TEST")
    print("=" * 70)
    
    segmenter = NeuralSegmenter()
    
    image_shape = (480, 640, 3)
    
    dummy_landmarks = np.array([
        [320 + i*5, 240 + i*3] for i in range(68)
    ])
    
    print("\n📐 Creating test masks...")
    
    lip_mask = segmenter.create_lip_mask(image_shape, dummy_landmarks)
    print(f"✅ Lip mask created: {lip_mask.shape}")
    
    eye_mask = segmenter.create_eye_mask(image_shape, dummy_landmarks)
    print(f"✅ Eye mask created: {eye_mask.shape}")
    
    skin_mask = segmenter.create_skin_mask(image_shape, dummy_landmarks)
    print(f"✅ Skin mask created: {skin_mask.shape}")
    
    print("\n🎯 Testing product-specific mask creation...")
    
    for product in ["Lipstick", "Eyeshadow", "Foundation"]:
        mask = segmenter.create_mask_for_product(image_shape, product, dummy_landmarks)
        non_zero = np.count_nonzero(mask)
        print(f"   {product}: {mask.shape}, {non_zero} active pixels")
    
    print("\n✨ Testing mask refinement...")
    
    refined = segmenter.refine_mask(lip_mask)
    print(f"✅ Refined mask: {refined.shape}")
    
    smoothed = segmenter.smooth_mask_edges(lip_mask, blur_amount=21)
    print(f"✅ Smoothed mask: {smoothed.shape}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)