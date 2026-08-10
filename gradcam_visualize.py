# gradcam_visualize.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from adni_dataset import ADNIDataset
from model_3dcnn_gradcam import GradCAM3DCNN

# ---- Configuration ----
model_path = "best_3dcnn_gradcam.pth"
csv_file = "/project/aereditato/abhat/adni-mri-classification/adni_preprocessed_npy_metadata.csv"
idx_to_visualize = 72  # Change this to visualize a different subject
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['MCI', 'AD', 'CN']

# ---- Load Model ----
model = GradCAM3DCNN(num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---- Load Dataset and Sample ----
dataset = ADNIDataset(csv_file)
print(f"Total samples in dataset: {len(dataset)}")
img, label = dataset[idx_to_visualize]
img = img.unsqueeze(0).to(device)  # Shape: [1, 1, D, H, W]

# ---- Forward Pass ----
output = model(img)
probs = torch.softmax(output, dim=1).squeeze().detach().cpu().numpy()
pred_class = np.argmax(probs)

print(f"Predicted class index: {pred_class}, probabilities:")
for i, p in enumerate(probs):
    print(f"  {class_names[i]:<3} ({i}): {p:.4f}")
print(f"True label: {label} ({class_names[label]})")

# ---- Grad-CAM Calculation ----
model.zero_grad()
output[0, pred_class].backward()

# Feature maps and gradients
grads = model.feature_maps.grad.detach().cpu().squeeze(0)     # [C, D, H, W]
feature_maps = model.feature_maps.detach().cpu().squeeze(0)   # [C, D, H, W]
weights = grads.mean(dim=[1, 2, 3])  # [C]

# Weighted combination
cam = torch.zeros_like(feature_maps[0])
for i, w in enumerate(weights):
    cam += w * feature_maps[i]
cam = cam.numpy()
cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ---- Improve resolution ----
cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
cam_interp = F.interpolate(cam_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
cam_interp = cam_interp.squeeze().numpy()

# ---- Plot middle ±2 slices ----
original_img = img.cpu().squeeze().numpy()  # [D,H,W]
mid = cam_interp.shape[0] // 2
slice_indices = range(mid - 2, mid + 3)

fig, axs = plt.subplots(5, 2, figsize=(8, 12))
for i, z in enumerate(slice_indices):
    orig_slice = original_img[z, :, :]
    heat_slice = cam_interp[z, :, :]
    
    axs[i, 0].imshow(orig_slice, cmap='gray')
    axs[i, 0].set_title(f"Original Slice {z}")
    axs[i, 0].axis('off')
    
    axs[i, 1].imshow(orig_slice, cmap='gray')
    axs[i, 1].imshow(heat_slice, cmap='jet', alpha=0.5)
    axs[i, 1].set_title(f"Grad-CAM Overlay {z}")
    axs[i, 1].axis('off')

plt.tight_layout()
fname = f"gradcam_idx{idx_to_visualize}_true{class_names[label]}_pred{class_names[pred_class]}.pdf"
plt.savefig(fname)
print(f"✅ Grad-CAM visualization saved to {fname}")
