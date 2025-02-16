import cv2
import os
import numpy as np
from skimage import exposure, restoration
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

def preprocess_supernova_image(image_path, save_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load {image_path}")
        return

    # Avoid division by zero during normalization
    min_val, max_val = np.min(image), np.max(image)
    if max_val == min_val:
        print(f"Warning: Skipping {image_path} (constant pixel value).")
        return
    image = (image - min_val) / (max_val - min_val)

    # Apply contrast stretching instead of CLAHE
    image = exposure.rescale_intensity(image, in_range=(0.02, 0.98))

    # Reduce noise while preserving edges
    denoised_image = restoration.denoise_wavelet(
        image, channel_axis=None, method='VisuShrink', rescale_sigma=True
    )

    # Background subtraction with less aggressive parameters
    sigma_clip = SigmaClip(sigma=5.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(denoised_image, (70, 70), filter_size=(7, 7), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    # Clip values to prevent negative intensities
    image_clean = np.clip(denoised_image - bkg.background, 0, 1)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save processed image
    cv2.imwrite(save_path, (image_clean * 255).astype(np.uint8))
    print(f"Processed & Saved: {save_path}")

def process_dataset(data_folder, txt_file, save_folder):
    os.makedirs(save_folder, exist_ok=True)  # Ensure save directory exists

    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} not found!")
        return

    with open(txt_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    for img_name in image_names:
        # Handle different formats of image paths
        if img_name.startswith("data/images/"):  
            image_path = os.path.join(os.getcwd(), img_name)  # Absolute path
        else:
            image_path = os.path.join(data_folder, img_name + ".png")  # Assuming missing extension

        save_path = os.path.join(save_folder, os.path.basename(image_path))

        if os.path.exists(image_path):
            preprocess_supernova_image(image_path, save_path)
        else:
            print(f"Warning: {image_path} does not exist!")

def main():
    data_folder = r"E:\HACKHATHON\NextAI\Data\Panstarrs\images"

    print(f"Checking data folder: {data_folder}")
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found!")
        return
    
    process_dataset(data_folder, "ImageSets/Main/Main/train.txt", "processed/train")
    process_dataset(data_folder, "ImageSets/Main/Main/test.txt", "processed/test")
    process_dataset(data_folder, "ImageSets/Main/Main/trainval.txt", "processed/trainval")

if __name__ == "__main__":
    main()
