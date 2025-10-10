"""
Face Detection Module using OpenCV Haar Cascades
Detects face, eyes, and mouth regions
"""

import cv2
import numpy as np
from config import HAAR_SCALE_FACTOR, HAAR_MIN_NEIGHBORS, HAAR_MIN_SIZE

class FaceDetector:
    def __init__(self):
        """Initialize Haar Cascade classifiers"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
    def detect_face(self, image):
        """
        Detect face in image
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            dict with face coordinates or None if no face detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE
        )
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        
        return {
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'bbox': (int(x), int(y), int(w), int(h))
        }
    
    def detect_eyes(self, image, face_region):
        """
        Detect eyes within face region
        
        Args:
            image: BGR image
            face_region: dict with face coordinates
            
        Returns:
            list of eye regions
        """
        if face_region is None:
            return []
        
        x, y, w, h = face_region['bbox']
        
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Convert eye coordinates to full image coordinates
        eye_list = []
        for (ex, ey, ew, eh) in eyes:
            eye_list.append({
                'x': int(x + ex),
                'y': int(y + ey),
                'width': int(ew),
                'height': int(eh),
                'bbox': (int(x + ex), int(y + ey), int(ew), int(eh))
            })
        
        # Sort eyes by x-coordinate (left to right)
        eye_list.sort(key=lambda e: e['x'])
        
        return eye_list
    
    def detect_mouth(self, image, face_region):
        """
        Detect mouth within face region
        
        Args:
            image: BGR image
            face_region: dict with face coordinates
            
        Returns:
            dict with mouth region or None
        """
        if face_region is None:
            return None
        
        x, y, w, h = face_region['bbox']
        
        # Focus on lower half of face for mouth detection
        mouth_y_start = y + int(h * 0.5)
        mouth_roi = image[mouth_y_start:y+h, x:x+w]
        
        if mouth_roi.size == 0:
            return None
        
        gray_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        
        mouths = self.mouth_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(30, 20)
        )
        
        if len(mouths) == 0:
            # Fallback: estimate mouth position
            mouth_x = x + int(w * 0.25)
            mouth_y = y + int(h * 0.65)
            mouth_w = int(w * 0.5)
            mouth_h = int(h * 0.15)
            
            return {
                'x': mouth_x,
                'y': mouth_y,
                'width': mouth_w,
                'height': mouth_h,
                'bbox': (mouth_x, mouth_y, mouth_w, mouth_h),
                'estimated': True
            }
        
        # Use the mouth closest to center
        mouth = mouths[0]
        mx, my, mw, mh = mouth
        
        return {
            'x': int(x + mx),
            'y': int(mouth_y_start + my),
            'width': int(mw),
            'height': int(mh),
            'bbox': (int(x + mx), int(mouth_y_start + my), int(mw), int(mh)),
            'estimated': False
        }
    
    def detect_all(self, image):
        """
        Detect all facial features
        
        Args:
            image: BGR image
            
        Returns:
            dict with all detected regions
        """
        face = self.detect_face(image)
        
        if face is None:
            return None
        
        eyes = self.detect_eyes(image, face)
        mouth = self.detect_mouth(image, face)
        
        return {
            'face': face,
            'eyes': eyes,
            'mouth': mouth
        }