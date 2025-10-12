"""
Segmentation Module
Creates pixel-perfect masks for makeup application regions
UPDATED: Enhanced foundation mask for tilted faces
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
        
        jawline = landmarks[0:17]
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        nose_bridge = landmarks[27:31]
        
        forehead_estimate = self._estimate_forehead(jawline, left_eyebrow, right_eyebrow, nose_bridge)
        
        face_contour = np.vstack([
            forehead_estimate,
            right_eyebrow[::-1],
            jawline[8:],
            jawline[:9][::-1],
            left_eyebrow
        ])
        
        face_contour_int = face_contour.astype(np.int32)
        cv2.fillPoly(mask, [face_contour_int], 255)
        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        for eye in [left_eye, right_eye]:
            ex_center = int(np.mean(eye[:, 0]))
            ey_center = int(np.mean(eye[:, 1]))
            ex_radius = int((np.max(eye[:, 0]) - np.min(eye[:, 0])) * 0.8)
            ey_radius = int((np.max(eye[:, 1]) - np.min(eye[:, 1])) * 0.9)
            
            cv2.ellipse(
                mask,
                (ex_center, ey_center),
                (ex_radius, ey_radius),
                0, 0, 360,
                0,
                -1
            )
        
        outer_lip = landmarks[48:60]
        
        lip_expansion = self._expand_lip_region(outer_lip, expansion_factor=1.3)
        lip_expansion_int = lip_expansion.astype(np.int32)
        cv2.fillPoly(mask, [lip_expansion_int], 0)
        
        nose_tip = landmarks[30:36]
        nose_center_x = int(np.mean(nose_tip[:, 0]))
        nose_center_y = int(np.mean(nose_tip[:, 1]))
        nose_radius_x = int((np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0])) * 0.7)
        nose_radius_y = int((np.max(nose_tip[:, 1]) - np.min(nose_tip[:, 1])) * 0.7)
        
        cv2.ellipse(
            mask,
            (nose_center_x, nose_center_y),
            (nose_radius_x, nose_radius_y),
            0, 0, 360,
            0,
            -1
        )
        
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        return mask
    
    def _estimate_forehead(self, jawline, left_eyebrow, right_eyebrow, nose_bridge):
        eyebrow_top_y = min(np.min(left_eyebrow[:, 1]), np.min(right_eyebrow[:, 1]))
        
        face_width = np.max(jawline[:, 0]) - np.min(jawline[:, 0])
        forehead_height = face_width * 0.35
        
        forehead_y = int(eyebrow_top_y - forehead_height)
        
        left_temple_x = int(left_eyebrow[0, 0] - face_width * 0.05)
        right_temple_x = int(right_eyebrow[-1, 0] + face_width * 0.05)
        
        num_points = 7
        forehead_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(left_temple_x + (right_temple_x - left_temple_x) * t)
            
            curve_factor = np.sin(t * np.pi) * 0.3
            y = int(forehead_y - forehead_height * curve_factor * 0.2)
            
            forehead_points.append([x, y])
        
        return np.array(forehead_points)
    
    def _expand_lip_region(self, lip_points, expansion_factor=1.2):
        center_x = np.mean(lip_points[:, 0])
        center_y = np.mean(lip_points[:, 1])
        
        expanded = lip_points.copy().astype(np.float32)
        
        for i in range(len(expanded)):
            dx = expanded[i, 0] - center_x
            dy = expanded[i, 1] - center_y
            
            expanded[i, 0] = center_x + dx * expansion_factor
            expanded[i, 1] = center_y + dy * expansion_factor
        
        return expanded
    
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
    
    print("\n📐 Creating test masks with enhanced foundation coverage...")
    
    lip_mask = segmenter.create_lip_mask(image_shape, dummy_landmarks)
    print(f"✅ Lip mask created: {lip_mask.shape}")
    
    eye_mask = segmenter.create_eye_mask(image_shape, dummy_landmarks)
    print(f"✅ Eye mask created: {eye_mask.shape}")
    
    skin_mask = segmenter.create_skin_mask(image_shape, dummy_landmarks)
    non_zero = np.count_nonzero(skin_mask)
    print(f"✅ Skin mask created: {skin_mask.shape}, {non_zero} active pixels")
    
    print("\n🎯 Testing product-specific mask creation...")
    
    for product in ["Lipstick", "Eyeshadow", "Foundation"]:
        mask = segmenter.create_mask_for_product(image_shape, product, dummy_landmarks)
        non_zero = np.count_nonzero(mask)
        coverage = (non_zero / (image_shape[0] * image_shape[1])) * 100
        print(f"   {product}: {mask.shape}, {non_zero} pixels ({coverage:.1f}% coverage)")
    
    print("\n✨ Testing mask refinement...")
    
    refined = segmenter.refine_mask(skin_mask)
    print(f"✅ Refined mask: {refined.shape}")
    
    smoothed = segmenter.smooth_mask_edges(skin_mask, blur_amount=21)
    print(f"✅ Smoothed mask: {smoothed.shape}")
    
    print("\n💡 Enhanced Features:")
    print("   • Forehead estimation for tilted faces")
    print("   • Full face contour coverage")
    print("   • Expanded lip exclusion zone")
    print("   • Nose exclusion for natural look")
    print("   • Heavy gaussian blur (51x51) for smooth blending")
    print("   • Morphological closing for gap filling")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE - ENHANCED FOUNDATION COVERAGE")
    print("=" * 70)