# BSNIP 3D CNN - Run v1 (Baseline)
##Experiment Configuration
- **Architecture:** Simple3DCNN with dynamic shape adaptation for native SPM12 Grey Matter volumes `(121, 145, 121)`
- **Regularization:** Dropout `p=0.4` (fc1), Weight Decay `1e-4`
- **Optimizer:** Adam (`lr=2e-5`), `ReduceLROnPlateau` scheduler (patience=5, factor=0.5, tracking `val_auc`)
- **Batch Size & Epochs:** `batch_size=4`, `epochs=50`
- **Data Split:** Stratified Subject-Level 70/15/15 (Train: 261, Val: 57, Test: 57)

## Results
- **Best Validation AUC:** `0.8294` (Epoch 17)
- **Best Validation Accuracy:** `78.95%` (Epoch 26, Balanced Accuracy: `78.63%`)

### Test Set Performance (57 Held-out Subjects)
- **Overall Accuracy:** `61%`
- **Schizophrenia (SZ) Recall:** `92%` (23/25 true SZ cases correctly identified)
- **Healthy Control (HC) Precision:** `86%`

## Artifacts
- `best_bsnip_3dcnn.pth`: Trained model checkpoint saved at peak validation AUC (`0.8294`)
- `training_log_bsnip.csv`: Per-epoch metrics log (Loss, Accuracy, Balanced Accuracy, AUC-ROC)
- `training_curves_bsnip.png`: Publication-ready plot of training/validation Loss and AUC-ROC curves
- `gradcam_results/`: 3D Grad-CAM saliency map overlays (Axial, Coronal, Sagittal cross-sections) for 10 test subjects
