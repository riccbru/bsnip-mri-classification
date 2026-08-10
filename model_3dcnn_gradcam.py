import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_3dcnn import Simple3DCNN

# --- GradCAM-Compatible Model ---
class GradCAM3DCNN(Simple3DCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.feature_maps = None

        def hook_fn(module, input, output):
            self.feature_maps = output
            if self.feature_maps.requires_grad:
                self.feature_maps.retain_grad()

        self.conv3.register_forward_hook(hook_fn)

# --- Training Script ---
def train_model():
    from adni_dataset import ADNIDataset
    import pandas as pd
    import csv
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # --- Config ---
    csv_file = 'adni_preprocessed_npy_metadata.csv'
    batch_size = 32
    epochs = 500
    lr = 1e-5
    patience = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Dataset ---
    full_dataset = ADNIDataset(csv_file)
    print(f"Total samples in dataset: {len(full_dataset)}")

    df = pd.read_csv(csv_file)
    labels = df['diagnosis'].astype(int).tolist()

    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GradCAM3DCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Initialize CSV Logging ---
    log_file = "training_log.csv"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # --- Logging ---
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss])

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_3dcnn_gradcam.pth")
            print(f"✅ Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    # --- Final Eval ---
    model.load_state_dict(torch.load("best_3dcnn_gradcam.pth"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    label_to_name = {0: "MCI", 1: "AD", 2: "CN"}
    target_names = [label_to_name[i] for i in sorted(set(y_true))]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

# --- Only run training if script is executed directly ---
if __name__ == "__main__":
    train_model()
