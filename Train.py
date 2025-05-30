import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import RandAugment

# ✅ Augmentations - Stronger Generalization
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomResizedCrop(150, scale=(0.7, 1.0)),  # Prevents excessive zoom-in
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
    RandAugment(num_ops=2, magnitude=9),  # ✅ FIXED
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load Dataset
train_data = torchvision.datasets.ImageFolder(root="D:/Animal Detection/Dataset/Train",
                                              transform=transform)
val_data = torchvision.datasets.ImageFolder(root="D:/Animal Detection/Dataset/Validation",
                                            transform=transform)

# ✅ Compute Class Weights (Fixes Imbalance)
labels = [label for _, label in train_data.samples]
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# ✅ Create Dataloaders
num_workers = 4 if torch.cuda.is_available() else 2  # ✅ Use multi-threading
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=num_workers)

# ✅ Load Pre-trained ResNet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features

# ✅ Modify Classifier
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(train_data.classes))
)
model.to(device)  # ✅ Load model on device before training

# ✅ Focal Loss (Fixes Class Domination Issue)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss *= self.alpha[targets]
        return focal_loss.mean()

# ✅ Set Up Training Components
if __name__ == "__main__":
    class_weights = class_weights.to(device)

    criterion = FocalLoss(gamma=2, alpha=class_weights)  # ✅ Uses Focal Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)  # ✅ Slightly Higher Weight Decay
    epochs = 20  # ✅ Increased training time for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # ✅ Use `epochs`

    # ✅ Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ✅ Validation Loop + Per-Class Accuracy
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        class_correct = [0] * len(train_data.classes)
        class_total = [0] * len(train_data.classes)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")

        # ✅ Print Per-Class Accuracy
        for i, class_name in enumerate(train_data.classes):
            if class_total[i] > 0:
                print(f"Class {class_name}: {100 * class_correct[i] / class_total[i]:.2f}%")

        scheduler.step()

    # ✅ Save Model
    torch.save(model.state_dict(), "animal_detector.pth")
    print("✅ Model saved successfully!")
