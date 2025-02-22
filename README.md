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
---

## 🔹 Results of EDA:

Based on the exploratory data analysis (EDA), several key insights can be drawn about the image regions and their characteristics:  

#### **1. Image Intensity and Variability**  
- Most image regions are dark, but some have significantly higher brightness, suggesting the presence of bright objects.  
- Standard deviation (Std_Dev) follows a similar pattern, indicating that most regions have low pixel variation, except for some areas with high variability, possibly due to bright objects or noise.  

#### **2. Texture and Edge Features**  
- Entropy and GLCM Contrast are highest in very bright regions, indicating that these areas have greater complexity and contrast.  
- Laplacian Variance shows a skewed distribution, meaning that most regions have weak edges, but some exhibit strong edge features.  
- Bright Pixel Count and Edge Count confirm that only a few regions contain significant bright objects or complex structures.  

#### **3. Object Shape and Size**  
- Circularity is mostly low, meaning detected objects are generally irregular in shape, though some circular objects exist.  
- Bounding Box Area shows that most objects are small, but a few significantly larger objects exist.  

#### **4. Frequency Domain Analysis**  
- FFT Energy and Dominant Frequency are skewed towards lower values, suggesting that most image regions have simple frequency components.  
- Higher Mean Intensity regions tend to have higher Dominant Frequency, indicating that brighter areas contain more high-frequency components.  

#### **5. Correlations Between Features**  
- **Mean Intensity & Median Intensity (0.71)**: Strongly related, as expected.  
- **Mean Intensity & Dominant Frequency (0.87)**: Bright regions have more high-frequency content.  
- **Entropy & FFT Energy (-0.82)**: More complex regions have lower frequency energy.  
- **GLCM Contrast & GLCM Homogeneity (0.96)**: High contrast areas also have high homogeneity, meaning structured textures.  
- **Edge Count & FFT Energy (0.88)**: More edges correspond to higher frequency energy.  

#### **6. Clustering Observations**  
- **K-Means Clustering** identified three groups based on brightness and texture contrast:  
  - One cluster with high contrast and varied brightness (likely bright, structured areas).  
  - One cluster with high contrast but low brightness (dark, high-contrast regions).  
  - One cluster with low contrast and low brightness (plain, dark areas).  
- **DBSCAN** identified three density-based clusters, capturing irregularly shaped groups and highlighting outliers, meaning some regions do not fit well into any cluster.  

### **Final Takeaways**  
- **Bright objects and high-contrast regions stand out distinctly from the majority of image regions.**  
- **Most regions are dark, low-texture, and relatively uniform, with few complex structures.**  
- **Outliers exist in terms of brightness, edge strength, and object size, indicating the presence of distinct features in certain regions.**  
- **Frequency domain and texture features are interrelated, showing that high-contrast areas tend to have strong frequency components.**  
- **Clustering methods suggest natural groupings, with DBSCAN capturing irregular shapes and outliers better than K-Means.**

---
## 🔹**Model Selection – Faster R-CNN with Multi-Scale Fusion**
- **Why Faster R-CNN?**  
  - It detects objects (supernovae) at different scales.
  - The **Region Proposal Network (RPN)** helps identify potential regions of interest.
- **Multi-Scale Fusion**:
  - Extract **multi-scale features** from CNN (e.g., ResNet, EfficientNet).
  - Fuse different levels of extracted features for enhanced detection.
