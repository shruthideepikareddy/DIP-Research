import cv2
import numpy as np
from skimage import measure, morphology, segmentation, filters, feature
import pandas as pd

class ParticleAnalyzer:
    def __init__(self):
        pass

    def preprocess(self, image):
        """
        Zero-parameter preprocessing pipeline.
        Converts to grayscale, applies blurring, and uses Otsu's binarization.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Denoising - Bilateral filter preserves edges while smoothing background better than Gaussian
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        
        # Otsu's Thresholding - Standard (Particles=White, Background=Black)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If the image is mostly white, we assume it's dark-on-light and invert
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
            
        # Morphological cleaning - Opening
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Aggressive Noise Filtering: Remove very small objects that aren't particles
        from skimage import morphology
        cleaned_mask = morphology.remove_small_objects(opening.astype(bool), min_size=64)
        binary = (cleaned_mask.astype(np.uint8) * 255)
        
        return binary

    def segment(self, binary):
        """
        Refined Watershed Segmentation with peak detection for better de-clumping.
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        
        peak_threshold = 0.1 * dist_transform.max()
        coords = peak_local_max(dist_transform, min_distance=7, threshold_abs=peak_threshold, labels=binary)
        
        markers = np.zeros_like(dist_transform, dtype=int)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        
        labels = watershed(-dist_transform, markers, mask=binary)
        
        return labels

    def calculate_metrics(self, labels, image, binary, min_area=50):
        """
        Calculates standard and novel metrics.
        'image' is the original (BGR or Gray), 'binary' is the thresholded mask.
        'min_area' filters out noise/ghost particles.
        """
        props = measure.regionprops(labels)
        data = []
        
        # Filter properties by area
        props = [p for p in props if p.area >= min_area]
        
        # Calculate median area of 'spherical' candidates for relative filtering
        p_areas = []
        for p in props:
            p_circ = (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0
            if p_circ > 0.7: p_areas.append(p.area)
        
        median_particle_area = np.median(p_areas) if p_areas else 1000
        
        total_area = np.sum(binary > 0)
        mean_area = np.mean([p.area for p in props]) if props else 0
        global_mean = np.mean(image)
        
        for p in props:
            # Standard metrics
            area = p.area
            perimeter = p.perimeter
            eccentricity = p.eccentricity
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            
            # Novel Metric: Solidity (Convexity)
            solidity = p.solidity
            
            # State classification handled downstream
            
            # Simple edge detection
            minr, minc, maxr, maxc = p.bbox
            if minr == 0 or minc == 0 or maxr == (labels.shape[0]-1) or maxc == (labels.shape[1]-1):
                if state == "Green":
                    state = "Yellow"

            # Intensity & Variance Features
            mask = (labels == p.label)
            try:
                if len(image.shape) == 3:
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    relevant_region = gray_img[mask]
                else:
                    relevant_region = image[mask]
                
                mean_intensity = np.mean(relevant_region)
                intensity_std = np.std(relevant_region) 
            except:
                mean_intensity = 0
                intensity_std = 0

            # State Classification logic (Modularized)
            state = self.classify_state(p, mean_intensity, global_mean, circularity, solidity, complexity_score, area, median_particle_area, intensity_std)

            data.append({
                "ID": p.label,
                "Area": area,
                "Perimeter": round(perimeter, 2),
                "Circularity": round(circularity, 2),
                "Complexity": round(complexity_score, 2),
                "Intensity": round(mean_intensity, 2),
                "State": state
            })
            
        return pd.DataFrame(data), pai

    def classify_state(self, p, mean_intensity, global_mean, circularity, solidity, complexity_score, area, median_particle_area, intensity_std):
        """
        Base classification logic (Guilty until proven innocent).
        """
        state = "Red" # Default to gap
        
        # Promotion to GREEN (Particle)
        if (110 < mean_intensity < 235) and (circularity > 0.7) and (solidity > 0.85) and (area > median_particle_area * 0.3):
            state = "Green"
        elif (130 < mean_intensity < 230) and (area > median_particle_area * 0.7):
            state = "Green"
            
        # Hard Gap Forcing
        if mean_intensity > 235 or intensity_std < 5:
            state = "Red"
        if mean_intensity < (global_mean * 0.6):
            state = "Red"
            
        return state

    def get_colored_output(self, image, labels, df, mode="Solid", color_mode="Spectral", alpha=0.5):
        """
        Generates classification image with solid masks or contours.
        'color_mode' can be 'Spectral' (intensity-based) or 'Uniform' (single color).
        """
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
            
        overlay = output.copy()
        
        # Qualitative colormap for "Spectral" mode
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab20')
        
        # Strictly only modify pixels that belong to a label
        label_mask = (labels > 0).astype(np.uint8)
        
        for _, row in df.iterrows():
            mask = (labels == row['ID']).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            
            if mode == "Solid":
                if color_mode == "Uniform":
                    color = (255, 128, 0) # High-contrast Blue (BGR)
                else:
                    intensity = row.get('Intensity', int(row['ID']) % 20)
                    color_idx = (intensity / 255.0) if 'Intensity' in row else ((int(row['ID']) % 20) / 20.0)
                    color_tuple = cmap(color_idx)[:3]
                    color = [int(p * 255) for p in color_tuple[::-1]]
                
                # Draw on the overlay
                cv2.drawContours(overlay, [cnts[0]], -1, color, -1)
                cv2.drawContours(output, [cnts[0]], -1, color, 1) # Thin outline
            else:
                # Original contour classification
                color_map = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255)}
                color = color_map.get(row['State'], (255, 255, 255))
                cv2.drawContours(output, cnts, -1, color, 2)
            
            # Numeric labels at centroid
            M = cv2.moments(cnts[0])
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(output, str(int(row['ID'])), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
                cv2.putText(output, str(int(row['ID'])), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            
        if mode == "Solid":
            # Strictly apply alpha blending only to labeled pixels
            mask_indices = np.where(label_mask > 0)
            combined = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
            output[mask_indices] = combined[mask_indices]
            
        return output
        
if __name__ == "__main__":
    # Test stub
    import matplotlib.pyplot as plt
    
    # Create synthetic image if needed
    img = np.zeros((400, 400), dtype=np.uint8)
    # Some "particles"
    cv2.circle(img, (100, 100), 20, 255, -1) # Isolated
    cv2.circle(img, (200, 200), 25, 255, -1) # Agglomerated 1
    cv2.circle(img, (220, 220), 25, 255, -1) # Agglomerated 2
    cv2.rectangle(img, (300, 300), (350, 320), 255, -1) # Jagged/Edge
    
    analyzer = ParticleAnalyzer()
    binary = analyzer.preprocess(img)
    labels = analyzer.segment(binary)
    df, pai = analyzer.calculate_metrics(labels, img, binary, min_area=50)
    output = analyzer.get_colored_output(img, labels, df)
    
    # Save for verification
    cv2.imwrite("test_input.png", img)
    cv2.imwrite("test_output.png", output)
    
    print(f"PAI: {pai:.2f}")
    print(df.head())
