import numpy as np
import cv2
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

# Step 1: Load Pretrained CNN for Feature Extraction
def extract_features(image_path, model, transform, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

# Define feature extractor (ResNet-50 without the final classification layer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
resnet.to(device)
resnet.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 2: Process Each Folder
folders = ["train", "test", "trainval"]
dataset_root = "processed"  # Adjust path if needed

for folder in folders:
    image_folder = os.path.join(dataset_root, folder)
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"No images found in {image_folder}, skipping...")
        continue
    
    print(f"Processing {len(image_paths)} images in {folder}...")
    
    features = np.array([extract_features(img, resnet, transform, device) for img in image_paths])
    
    # Step 3: Apply K-Means Clustering
    k = 5  # Number of clusters (adjust based on dataset)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_
    
    # Step 4: Assign Pseudo-Labels
    pseudo_labeled_data = list(zip(image_paths, labels))
    
    # Save Pseudo Labels
    txt_file = f"pseudo_labels_{folder}.txt"
    with open(txt_file, "w") as f:
        for img, label in pseudo_labeled_data:
            f.write(f"{img} {label}\n")
    
    print(f"Pseudo-labeling completed for {folder}! Labels saved in {txt_file}")
