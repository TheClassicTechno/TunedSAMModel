

# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms

# Access dataset (replace with correct folder)
hair_folder = 'C://Users/ellaa/Documents/StanfordResearch/img_to_type/images/img2'

# CNN model modified for feature extraction
class HairFeatureExtractor(nn.Module):
    def __init__(self):
        super(HairFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),  # Single output neuron for curliness score
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.encoder(x)
        return x.squeeze(1) * 6 + 1  # Scale output from [0,1] to [1,7]

# Custom dataset class
class HairDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # Add labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)

# Load images and generate random curliness labels
image_paths = sorted(
    [os.path.join(hair_folder, f) for f in os.listdir(hair_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
)[:45]
labels = [random.uniform(1, 7) for _ in range(len(image_paths))]  # Randomized labels between 1 and 7

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create dataset
dataset = HairDataset(image_paths, labels, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=9, shuffle=False)

# Initialize model, loss function, optimizer
model = HairFeatureExtractor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_loss_over_time = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  # Compare against real values
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_loss_over_time.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Plot loss over time
plt.plot(train_loss_over_time)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Example: Extract features and visualize predictions
def extract_features(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        curliness_score = model(image).item()
    return curliness_score

# Visualize curliness predictions
num_examples = 6
random_indices = random.sample(range(len(image_paths)), num_examples)

fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))

for i, idx in enumerate(random_indices):
    img = Image.open(image_paths[idx])
    curliness_score = extract_features(image_paths[idx], model, transform)
    
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"Curliness: {curliness_score:.2f}")

plt.tight_layout()
plt.show()

# Display a single image with its curliness score
sample_idx = random.choice(range(len(image_paths)))
sample_image = Image.open(image_paths[sample_idx])
sample_curliness_score = extract_features(image_paths[sample_idx], model, transform)

plt.figure(figsize=(5, 5))
plt.imshow(sample_image)
plt.axis('off')
plt.title(f"Predicted Curliness: {sample_curliness_score:.2f}")
plt.show()
