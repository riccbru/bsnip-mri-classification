# ADNI MRI Classification with 3D CNN and Grad-CAM

This repository contains code for classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects using preprocessed 3D MRI volumes from the ADNI dataset. It includes data preprocessing, metadata matching, model training, evaluation, and Grad-CAM visualizations for interpretability.

---

## 🧠 Project Overview
- **Dataset**: ADNI1/GO/2/3 3D MRI volumes (MPRAGE or equivalent T1-weighted scans)
- **Preprocessing**: 128×128×128 volume normalization
- **Model**: 3D CNN trained with early stopping
- **Explainability**: Grad-CAM to visualize important brain regions
- **Classes**: AD (Alzheimer’s), MCI (Mild Cognitive Impairment), CN (Cognitively Normal)

---

## 🔧 Environment Setup
Run the following from your RCC shell:
```bash
module load anaconda3
conda create -n adni_rcc python=3.10 -y
conda activate adni_rcc
pip install torch torchvision numpy pandas matplotlib nibabel scikit-learn tqdm
```

## 🔧 Directory Structure
```
adni-mri-classification/
├── preprocessing.py
├── match_mri_to_diagnosis.py
├── generate_metadata_csv.py
├── generate_npy_metadata.py
├── train_3dcnn.py
├── model_3dcnn_gradcam.py
├── gradcam_visualize.py
├── adni_dataset.py
├── plot_training_log.py
├── adni_matched_nii_with_labels.csv         # Generated metadata nii files
├── adni_preprocessed_npy_metadata.csv       # Metadata npy volumes
├── training_log.csv                         # Saved training logs
├── preprocessed_npy/                        # Preprocessed npy volumes
```

## Step-by-Step Instructions
### 1. Clone the repository
```bash
git clone https://github.com/abhatfnal/adni-mri-classification.git
cd adni-mri-classification
conda activate adni_rcc
```

### 2. Match raw `.nii` MRIs to ADNI diagnosis metadata
Ensure you have extracted ADNI `.nii` files and the DXSUM CSVs.
```bash
python match_mri_to_diagnosis.py
```
This generates `adni_matched_nii_with_labels.csv`.

### 3. Preprocess MRI `.nii` to `.npy` (128×128×128)
```bash
python preprocessing.py
```
OUTPUT: `preprocessed_npy/`

### 4. Generate `.npy` metadata
```bash
python generate_npy_metadata.py
```
This produces `adni_preprocessed_npy_metadata.csv` used for training.

### 5. Train the 3D CNN
```bash
python model_3dcnn_gradcam.py
```
The model uses early stopping and saves `best_3dcnn_gradcam.pth`.
Logs are written to `training_log.csv`.

### 6. Plot training curves
```bash
python plot_training_log.py
```
OUTPUT: `training_log_plot.png`

### 7. Grad-CAM Visualization

Edit `gradcam_visualize.py` to set the sample index:
```
idx_to_visualize = 42 
# change to any value in [0, len(dataset) - 1]
```

Then run:
```bash
python gradcam_visualize.py
```
OUTPUT: `gradcam_output.pdf`

🧪 Example Results
Example classification report after training:

| Accuracy | Precision/Recall/F1-Score |
|:-:|:-:|
| 82% | **AD**: ~0.83, **MCI**: ~0.86, **CN**: ~0.76 |

Grad-CAM shows that the model focuses on relevant brain regions for classification.

> This repository assumes access to ADNI `.nii` scans and diagnosis metadata. Preprocessed `.npy` volumes are stored in `preprocessed_npy/` and not version controlled.

👤 Authors
Avinay Bhat — abhatfnal
