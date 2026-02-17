import cv2
import numpy as np
import os
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
# Alphabet (A–Z, 1–9) images: use WLASL-derived or other ASL sources only.
# Place images per class in wlasl_alphabet_images/<Class>/ (e.g. A/, B/, …, Z/, 1/, …).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'hand_landmarker.task')
if not os.path.isfile(MODEL_PATH):
    MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
DATA_FOLDER = os.path.join(SCRIPT_DIR, 'wlasl_alphabet_images')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'MP_Data')

def extract_keypoints_from_result(result):
    # Extracts keypoints from DetectionResult object
    # Returns flattened array of 63 floats (21 landmarks * 3 coords)
    if not result.hand_landmarks:
        return np.zeros(21*3)
    
    # Take first hand found
    landmarks = result.hand_landmarks[0]
    # Keep format consistent: [x, y, z, x, y, z...]
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return keypoints

def process_images_folder():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please download it first.")
        return

    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Dataset folder '{DATA_FOLDER}' not found!")
        return

    print(f"Initializing MediaPipe Tasks API with model: {MODEL_PATH}")
    
    # Create HandLandmarker (CPU delegate for consistency with training)
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=1,
                                           min_hand_detection_confidence=0.4)
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        print(f"Scanning '{DATA_FOLDER}' for class folders...")
        
        # Get all subdirectories (A, B, C...)
        classes = sorted([d for d in os.listdir(DATA_FOLDER) 
                          if os.path.isdir(os.path.join(DATA_FOLDER, d))])
        
        for cls in classes:
            print(f"Processing class: {cls}")
            class_path = os.path.join(DATA_FOLDER, cls)
            save_path = os.path.join(OUTPUT_FOLDER, cls)
            os.makedirs(save_path, exist_ok=True)
            
            # Get images
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Use up to max_per_class images (default 150) for better letter accuracy
            max_per_class = getattr(process_images_folder, '_max_per_class', 150)
            images = images[:max_per_class] 
            
            count = 0
            for idx, img_name in enumerate(images):
                img_path = os.path.join(class_path, img_name)
                
                # Read image
                # MediaPipe Tasks API requires mediapipe.Image
                np_image = cv2.imread(img_path)
                if np_image is None: continue
                
                # Convert BGR to RGB
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
                
                # Detect
                detection_result = detector.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    # Extract keypoints
                    kp = extract_keypoints_from_result(detection_result)
                    
                    # Save as sequence (repeat 30 times)
                    seq_len = 30
                    seq_dir = os.path.join(save_path, str(idx))
                    os.makedirs(seq_dir, exist_ok=True)
                    
                    for i in range(seq_len):
                        np.save(os.path.join(seq_dir, f"{i}.npy"), kp)
                        
                    count += 1
            print(f"  - Generated {count} sequences for class '{cls}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand keypoints from alphabet images")
    parser.add_argument("--max-per-class", type=int, default=150,
                        help="Max images per letter class (default 150, use more for better accuracy)")
    args = parser.parse_args()
    process_images_folder._max_per_class = args.max_per_class
    process_images_folder()
    print("\nProcessing Complete! Run: python3 sign_language_training/train_alphabet_model.py")
