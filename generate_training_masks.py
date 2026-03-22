import cv2
import numpy as np
import os
from ml_analyzer import MLParticleAnalyzer

def generate_masks(input_dir, output_dir):
    """
    Uses the ML Engine to generate ground-truth candidate masks for AI training.
    """
    analyzer = MLParticleAnalyzer()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "images"))
        os.makedirs(os.path.join(output_dir, "masks"))

    print(f"🚀 Starting Synthetic Labeling in {input_dir}...")
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(input_dir, filename)
            image = cv2.imread(path)
            
            # 1. Process with ML Engine
            binary = analyzer.preprocess_ml(image)
            labels = analyzer.segment(binary)
            
            # 2. Save Original Image
            cv2.imwrite(os.path.join(output_dir, "images", filename), image)
            
            # 3. Save Segmentation Mask (Instance labels)
            # We save as a 16-bit PNG to preserve particle IDs
            mask_filename = filename.rsplit('.', 1)[0] + "_mask.png"
            cv2.imwrite(os.path.join(output_dir, "masks", mask_filename), labels.astype(np.uint16))
            
            print(f"✅ Generated mask for {filename}")

if __name__ == "__main__":
    # Change these paths to your project folders
    input_folder = "c:/Users/sofia/DIP/samples" 
    output_folder = "c:/Users/sofia/DIP/ai_dataset"
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder {input_folder} not found. Please create it and add your images!")
    else:
        generate_masks(input_folder, output_folder)
