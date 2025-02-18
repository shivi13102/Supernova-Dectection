import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load your dataset
df = pd.read_csv("train_features_extended.csv")  # Replace with actual file path

# Define columns based on feature categories
intensity_features = ['Mean_Intensity', 'Median_Intensity', 'Std_Dev', 'Skewness', 'Kurtosis', 'Entropy']
edge_features = ['Bright_Pixel_Count', 'Edge_Count', 'Laplacian_Variance']
texture_features = ['GLCM_Contrast', 'GLCM_Homogeneity', 'GLCM_Energy', 'GLCM_Texture_Entropy']
shape_features = ['Circularity', 'Bounding_Box_Area']
frequency_features = ['FFT_Energy', 'Dominant_Frequency']

### 1. HISTOGRAMS & KDE PLOTS ###
for feature in intensity_features + texture_features + frequency_features:
    plt.figure(figsize=(6, 4))
    
    # Plot histogram with kde
    ax = sns.histplot(df[feature], kde=True, bins=30, color='blue')  # Histogram color
    
    # Change KDE line color manually
    if ax.lines:  # Ensure there is a KDE line
        ax.lines[0].set_color('red')  # Set KDE color to red
        ax.lines[0].set_linewidth(2)  # Set KDE line width
    
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    
    plt.show()

### 2. BOX PLOTS ###
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[intensity_features + edge_features + texture_features])
plt.xticks(rotation=45)
plt.title("Feature Boxplots (Detect Outliers)")
plt.show()

### 3. PAIRPLOT ###
# Select features
selected_features = [
    'Mean_Intensity', 'Std_Dev', 'Entropy', 'Edge_Count', 
    'Laplacian_Variance', 'GLCM_Contrast', 'Circularity', 
    'Bounding_Box_Area', 'FFT_Energy', 'Dominant_Frequency'
]

df_selected = df[selected_features].dropna()  # Drop NaN values if any

# OPTION 1: Color by Clustering (Unsupervised Grouping)
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose 3 clusters
df_selected['Cluster'] = kmeans.fit_predict(df_selected)

# Create a colorful pairplot
sns.pairplot(df_selected, diag_kind='kde', hue='Cluster', palette='viridis')  # Use 'hue_variable' for gradient coloring
plt.show()

### 4. CORRELATION HEATMAP ###
# Drop non-numeric columns before correlation
df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns

# Now, generate the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

### 5. SCATTER PLOTS (Edge & Shape Analysis) ###
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['Circularity'], y=df['Bounding_Box_Area'])
plt.title("Circularity vs Bounding Box Area")
plt.xlabel("Circularity")
plt.ylabel("Bounding Box Area")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['Edge_Count'], y=df['Laplacian_Variance'])
plt.title("Edge Count vs Laplacian Variance")
plt.xlabel("Edge Count")
plt.ylabel("Laplacian Variance")
plt.show()

### 6. PCA for Dimensionality Reduction ###
features = intensity_features + edge_features + texture_features + shape_features + frequency_features
X = df[features].dropna()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection of Features")
plt.show()

### 7. t-SNE for Clustering ###
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(6, 4))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Projection of Features")
plt.show()