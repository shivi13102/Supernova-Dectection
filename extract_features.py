import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft2

# Path to the dataset
TRAIN_FOLDER = r"E:\HACKHATHON\NextAI\Data\Panstarrs\processed\train"
OUTPUT_CSV = "train_features_extended.csv"

def extract_features(image_path):
    """Extracts multiple features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Normalize Image (Avoiding Division Errors)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Intensity-Based Features
    mean_intensity = np.mean(image)
    median_intensity = np.median(image)
    std_dev = np.std(image)
    skewness = skew(image.ravel())
    kurt = kurtosis(image.ravel())

    # Histogram-Based Features
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Avoid log(0)
    bright_pixel_count = np.sum(image >= np.percentile(image, 90))

    # Edge Features
    edges = cv2.Canny(image, 100, 200)
    edge_count = np.sum(edges > 0)
    laplacian_variance = cv2.Laplacian(image, cv2.CV_64F).var()

    # GLCM Texture Features
    glcm = graycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    texture_entropy = -np.sum(glcm * np.log2(glcm + 1e-7))

    # Shape Features
    labeled = label(image > np.percentile(image, 90))
    props = regionprops(labeled)
    circularity = 0
    bounding_box_area = 0
    if props:
        area = props[0].area
        perimeter = props[0].perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
        bounding_box_area = props[0].bbox_area

    # Frequency-Domain Features (FFT)
    fft_image = np.abs(fft2(image))
    fft_energy = np.sum(fft_image)
    dominant_freq = np.max(fft_image)

    return [
        image_path, mean_intensity, median_intensity, std_dev, skewness, kurt,
        entropy, bright_pixel_count, edge_count, laplacian_variance, contrast,
        homogeneity, energy, texture_entropy, circularity, bounding_box_area, fft_energy, dominant_freq
    ]

def process_dataset(train_folder):
    """Processes all images in the dataset and saves extracted features."""
    data = []
    
    for filename in os.listdir(train_folder):
        image_path = os.path.join(train_folder, filename)
        if image_path.endswith(".png") or image_path.endswith(".jpg"):
            features = extract_features(image_path)
            if features:
                data.append(features)

    # Convert to Pandas DataFrame
    columns = [
        "Image_Path", "Mean_Intensity", "Median_Intensity", "Std_Dev", "Skewness", "Kurtosis",
        "Entropy", "Bright_Pixel_Count", "Edge_Count", "Laplacian_Variance",
        "GLCM_Contrast", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Texture_Entropy",
        "Circularity", "Bounding_Box_Area", "FFT_Energy", "Dominant_Frequency"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature extraction complete! Saved as {OUTPUT_CSV}")

if __name__ == "__main__":
    process_dataset(TRAIN_FOLDER)