# Supernova Dectection

### This project utilizes cutting-edge machine learning models to detect supernovae in astronomical images. By analyzing patterns and anomalies, it efficiently identifies potential supernovae candidates, aiding in the study of these powerful cosmic events.
---

## 🔹 Categories of Features

### 1️⃣ Intensity-Based Features

These describe the brightness of the image:

- Mean Intensity: Average brightness of the pixels.
- Median Intensity: Middle pixel intensity value.
- Standard Deviation: Variation in brightness.
- Skewness: Asymmetry in intensity distribution.
- Kurtosis: Sharpness of the intensity distribution.

### 2️⃣ Histogram-Based Features

These provide a distribution of pixel intensities:

- Histogram Entropy: Measures randomness in brightness.
- Contrast Stretching Percentiles: (5th and 95th percentile).
- Number of Bright Pixels: Pixels above a certain threshold (e.g., 90th percentile).

### 3️⃣ Edge and Shape-Based Features

These detect the shape and structure of objects:

- Canny Edge Count: Number of edges detected using Canny edge detection.
- Sobel Filter Response: Measures gradients in intensity.
- Laplacian Variance: Detects blur.
- Circularity: Detects if an object is circular (good for supernovae).
- Bounding Box Area: Area covered by the detected object.

### 4️⃣ Texture Features (Haralick Features - GLCM)

Texture features help distinguish supernovae from other objects.

- Energy: Sum of squared GLCM elements.
- Entropy: Randomness in the texture pattern.
- Contrast: Measures intensity variations.
- Homogeneity: Measures smoothness of the image.

### 5️⃣ Frequency-Domain Features (FFT)

These extract hidden patterns in the image:

- Fourier Transform Energy: Detects high-frequency variations.
- Dominant Frequency: Peak frequency component.
