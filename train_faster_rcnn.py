import torch
import torchvision
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


# Define Dataset Class
class PseudoLabelDataset(Dataset):
    def __init__(self, label_file, transform=None):
        self.data = []
        self.transform = transform

        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split()
                self.data.append((img_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error loading image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize

        if self.transform:
            image = self.transform(image)

        # Sample bounding box
        height, width, _ = image.shape
        bbox = [width // 4, height // 4, 3 * width // 4, 3 * height // 4]  

        target = {
            "boxes": torch.tensor([bbox], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64)
        }

        return image, target

# Define Image Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Dataset
dataset = PseudoLabelDataset("pseudo_labels_train.txt", transform=transform)
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


# Load Pretrained Faster R-CNN Model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = 6  # 5 clusters + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Training Settings
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "faster_rcnn_pseudo_labels.pth")
print("Training Complete. Model saved!")

# Inference Function
def infer(image_path):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform and Convert to Tensor
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_img = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_img)

    print("Predictions:", prediction)

# Run Inference (Example)
test_img_path = "processed/test/1011030551161304700_58508.png"  # Change to a real image
infer(test_img_path)
