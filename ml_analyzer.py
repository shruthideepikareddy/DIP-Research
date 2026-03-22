import cv2
import numpy as np
from skimage import measure, morphology, segmentation, filters, feature
import pandas as pd
from analyzer import ParticleAnalyzer

class MLParticleAnalyzer(ParticleAnalyzer):
    """
    Advanced Analyzer using Machine Learning (Texture-based) features
    to distinguish background gaps from particles.
    """
    
    def preprocess_ml(self, image):
        """
        ML-enhanced preprocessing using multi-feature segmentation.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Denoise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Feature-based thresholding (Multi-Otsu or Adaptive on features)
        # We use the Hessian eigenvalue feature to highlight the spheres
        h_elems = filters.hessian_matrix(denoised, sigma=2, order='rc')
        h_eigs = filters.hessian_matrix_eigvals(h_elems)
        
        # The bright cores are "blobs" (negative second eigenvalue in bright regions)
        feature_map = -h_eigs[1] 
        feature_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Adaptive Threshold on the feature map
        binary = cv2.adaptiveThreshold(feature_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 101, 2)
        
        # Clean up
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=64)
        return (cleaned.astype(np.uint8) * 255)

    def classify_state(self, p, mean_intensity, global_mean, circularity, solidity, complexity_score, area, median_particle_area, intensity_std):
        """
        Advanced classification using texture-aware heuristics.
        """
        state = "Green"
        
        # 1. Global Intensity Bands (Background vs Particle)
        if mean_intensity > 240: # Pure bright background
            return "Red"
        if mean_intensity < (global_mean * 0.65): # Deep shadow/gap
            return "Red"
            
        # 2. Geometry Strictness
        if area < (median_particle_area * 0.35): # Clearly too small
            return "Red"
            
        if circularity < 0.65 or solidity < 0.8: # Very irregular
            # Only promote to Red if it's NOT in the middle size range (Clump override)
            if area < (median_particle_area * 0.7):
                return "Red"
                
        return state
