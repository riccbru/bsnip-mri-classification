import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from adni_dataset import ADNIDataset
from model_3dcnn import Simple3DCNN
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Configuration
csv_file = './adni_preprocessed_npy_metadata.csv'
batch_size = 32
epochs = 500
lr = 0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
full_dataset = ADNIDataset(csv_file, transform=None)
print(f"Total samples in dataset: {len(full_dataset)}")

# Label mappings
label_to_name = {0: "MCI", 1: "AD", 2: "CN"}
df = pd.read_csv(csv_file)
labels = df['diagnosis'].astype(int).tolist()

from sklearn.model_selection import train_test_split

# First: train+val vs test split
trainval_idx, test_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Second: train vs val split (from trainval)
train_labels = [labels[i] for i in trainval_idx]
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.2,
    stratify=train_labels,
    random_state=42
)

# Subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = Simple3DCNN(num_classes=len(label_to_name)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop with validation tracking
best_val_loss = float('inf')
best_epoch = -1

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_3dcnn.pth")

# Final Evaluation on Test Set
model.load_state_dict(torch.load("best_3dcnn.pth"))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report (Best Model @ Epoch {} | Val Loss: {:.4f}):".format(best_epoch, best_val_loss))
target_names = [label_to_name[i] for i in sorted(set(y_true))]
print(classification_report(y_true, y_pred, target_names=target_names))

print("\n✅ Training complete. Best model saved as best_3dcnn.pth")
