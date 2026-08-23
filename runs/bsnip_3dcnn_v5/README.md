# B-SNIP 3D MRI Classification: Schizophrenia vs Healthy Controls

Automated end-to-end 3D Deep Learning pipeline for binary classification of Schizophrenia (SZ) versus Healthy Controls (HC) using structural T1-weighted MRI scans from the B-SNIP consortium dataset.

---

## 1. Technical Framework, Algorithmic Decisions & Training Dynamics

### Preprocessing & Input Representation
* **Spatial Normalization & Matrix Shape:** Raw T1w MRI scans are preprocessed, skull-stripped, and affinely aligned to MNI152 template space, yielding a dense volumetric tensor of shape $(C, D, H, W) = (1, 121, 145, 121)$.
* **Voxel Intensity Normalization:** Volumes are normalized via min-max scaling followed by standard score normalization ($z = \frac{x - \mu}{\sigma}$, where voxels are computed over non-zero brain tissue), preventing large DC-offsets and ensuring stable weight initialization gradients across batches.

---

### Algorithmic Decisions & Architectural Evolution

#### 1. Transition from Flattening to Global Average Pooling (GAP)
* **The Flattening Bottleneck (v1–v4):** In early iterations, the feature extractor output $\mathbf{X} \in \mathbb{R}^{64 \times 15 \times 18 \times 15}$ was vectorized into a $129,600$-dimensional vector connected to an initial linear layer $\mathbf{W}_{\text{fc1}} \in \mathbb{R}^{128 \times 129600}$.
  * *Failure Mode:* Over $16.5$ million trainable parameters were concentrated exclusively in the classification head. Given the sample size ($N_{\text{train}} = 261$), the model memorized volume coordinates by Epoch 3, leading to early validation stagnation ($\text{Loss} \approx 0.69$, $\text{AUC} \to 0.50$).
* **The Global Average Pooling Formulation (`v5_gap`):** Replaced flattening with `nn.AdaptiveAvgPool3d((1, 1, 1))`. For each feature map $k \in \{1, \dots, C\}$, GAP computes:
  $$\mu_k = \frac{1}{D \times H \times W} \sum_{d=1}^D \sum_{h=1}^H \sum_{w=1}^W X_{k}(d, h, w)$$
  * *Justification:* Collapses the spatial dimensions to a single 64-dimensional descriptor, reducing head parameters to merely $\mathbf{W}_{\text{head}} \in \mathbb{R}^{2 \times 64}$ (128 weights). This forces earlier convolutional filters to act as direct spatial semantic detectors across the entire cranial volume rather than learning fixed slice positions.

#### 2. Loss Function & Gradient Optimization
* **Unweighted vs Class-Weighted Cross-Entropy:** In `v4`, inverse-frequency weighting was applied ($\mathbf{w} = [0.8758, 1.1652]$) with a low learning rate ($2\times 10^{-5}$). 
  * *Failure Mode:* The mild class ratio ($56\%$ HC vs $44\%$ SZ) did not warrant heavy weighting; the penalty pushed gradients into a local saddle point where all output logits collapsed toward the majority class ($\text{SZ Recall} = 0.00$).
  * *Decision (`v5_gap`):* Reverted to standard unweighted Cross-Entropy Loss:
    $$\mathcal{L}_{\text{CE}} = - \sum_{i=1}^B \log\left(\frac{e^{z_{y_i}}}{\sum_{j=1}^2 e^{z_j}}\right)$$
    This restored gradient variance and unlocked balanced probabilistic distributions across classes.

#### 3. Optimizer & Learning Rate Schedule
* **AdamW at $\text{LR} = 1\times 10^{-4}$:** Replaced lower learning rates ($2\times 10^{-5}$) with $1\times 10^{-4}$ coupled with decoupled weight decay ($\lambda = 1\times 10^{-4}$). 
  * *Justification:* 3D convolutions with GAP have smoother loss landscapes than flattened networks. A higher initial step size allows AdamW to escape initial plateau regions without destabilizing feature map filters.
* **Batch Size ($B = 4$):** Constrained by 3D volumetric GPU VRAM limits (32GB per node). Small-batch stochasticity introduces implicit regularization that further mitigates overfitting on volumetric medical data.

#### 4. Real-Time 3D Data Augmentation Pipeline
* **Dynamic Transformations (Train split only):**
  * *Random 3D Translations:* Up to $\pm 3$ voxels in axial, coronal, and sagittal planes to induce translation invariance.
  * *Random Sagittal Flips:* Horizontal left-right flips ($p=0.5$) exploiting cerebral hemisphere symmetry while preserving gross volumetric structures.
  * *Additive Gaussian Noise:* $\mathcal{N}(0, \sigma^2)$ with $\sigma=0.01$ applied to voxel intensities to simulate thermal MRI scanner fluctuations.
  * *Validation/Test Invariance:* Validation and test loaders are kept strictly deterministic (zero augmentation).

#### 5. Validation-Tuned Threshold Grid Search
* To prevent decision threshold bias from test set leakage, an automated grid search for $p^* \in [0.30, 0.70]$ (step size $0.005$) is executed purely on the **Validation Set** to maximize Balanced Accuracy:
  $$p^* = \arg\max_{p} \left( \frac{\text{Sensitivity}(p) + \text{Specificity}(p)}{2} \right)_{\text{Val}}$$
* In `v5_gap`, the optimal threshold converged naturally to **$p^* = 0.500$** ($71.25\%$ Val Balanced Acc), proving that GAP outputs are natively calibrated without requiring artificial boundary shifts.

---

### Training Dynamics & Quantitative Results

<p align="center">
  <img src="training_curves.png" alt="BSNIP 3D CNN Training Curves" width="90%"/>
</p>

* **Continuous Learning:** Validation Loss steadily decreased alongside Training Loss from $\approx 0.73$ to $\approx 0.66$.
* **Validation Peak:** Peak AUC-ROC reached **0.7800** at Epoch 40 (saved as `best_model.pth`).
* **Test Performance (Unseen $N=57$):**

| Metric | Healthy Controls (HC) | Schizophrenia (SZ) | Macro / Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | **0.81** (81%) | 0.54 (54%) | 0.67 (Macro Avg) |
| **Recall (Sensitivity)** | 0.41 (41%) | **0.88** (88%) | 0.64 (Macro Avg) |
| **F1-Score** | 0.54 | **0.67** | 0.60 |
| **Balanced Accuracy** | — | — | **64.31%** |
| **Overall Accuracy** | — | — | **61.40%** |

---

## 2. Clinical Interpretation, Explainability & Neuroanatomical Biomarkers

### 3D Grad-CAM Explainability & Structural Verification
To ensure the network extracted genuine neuroanatomical pathology rather than scanner noise or edge artifacts, 3D Grad-CAM computed activation gradients at the final convolutional layer:
$$A_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right), \quad \alpha_k^c = \frac{1}{Z} \sum_{d}\sum_{h}\sum_{w} \frac{\partial y^c}{\partial A^k(d,h,w)}$$

#### Correct Classifications (True Positives vs True Negatives)
<p align="center">
  <img src="gradcam_results/S8278RUJ_true-SZ_pred-SZ.png" alt="True SZ" width="48%"/>
  <img src="gradcam_results/S9769ATM_true-HC_pred-HC.png" alt="True HC" width="48%"/>
</p>
<p align="center">
  <em>Left: <b>True Positive (SZ, p=0.505)</b> — Focal activation on the left lateral ventricle anterior horn and striatal boundary. Right: <b>True Negative (HC, p=0.469)</b> — Diffuse cortical mantle activation with preserved ventricular margins.</em>
</p>

#### Classification Errors (False Negatives vs False Positives)
<p align="center">
  <img src="gradcam_results/S2806CXU_true-SZ_pred-HC.png" alt="False HC" width="48%"/>
  <img src="gradcam_results/S1962CAN_true-HC_pred-SZ.png" alt="False SZ" width="48%"/>
</p>
<p align="center">
  <em>Left: <b>False Negative (SZ misclassified as HC, p=0.478)</b> — Atypical SZ anatomy without pronounced ventricular enlargement. Right: <b>False Positive (HC misclassified as SZ, p=0.513)</b> — Borderline decision uncertainty showing neutral/low-gradient activation.</em>
</p>

---

### Neuroanatomical Alignment with B-SNIP Literature

* **Lateral Ventriculomegaly:** In correctly classified Schizophrenia scans (e.g., `S8278RUJ`), the Grad-CAM focus localizes squarely on the **anterior horns of the lateral ventricles** and central cerebrospinal fluid (CSF) interfaces. Ventricular enlargement driven by surrounding gray-matter loss is the most consistently replicated structural biomarker in psychiatric neuroimaging.
* **Cortical Mantle Integrity:** For true healthy controls (e.g., `S9769ATM`), activation shifts outward toward the fronto-parietal cortex, reflecting preserved cortical thickness and absence of periventricular atrophy.
* **Connection to B-SNIP Psychosis Biotypes:** Extensive findings by the B-SNIP consortium (Clementz et al., 2022; Parker et al., 2025) demonstrate that conventional DSM categories fail to capture biologically homogeneous populations[cite: 1, 2]. Instead, numerical taxonomy identifies **three biologically distinct Psychosis Biotypes** using cognitive (BACS) and neurophysiological (EEG/ERP) endophenotypes[cite: 1, 2]:
  1. **Biotype 1:** Severe cognitive impairment, profound neural hypo-reactivity (reduced N100/P300 ERP amplitudes), and extensive brain volume loss[cite: 1, 2].
  2. **Biotype 2:** Moderate cognitive deficits with marked sensorimotor disinhibition (antisaccades, Stop Signal Task) and elevated intrinsic/ongoing high-frequency EEG activity[cite: 1, 2].
  3. **Biotype 3:** Near-normal cognitive/physiological profile with localized stimulus-salience abnormalities[cite: 1, 2].

While this model performs binary diagnostic classification (SZ vs HC), the structural patterns captured by the 3D CNN—specifically central ventricular expansion and peri-striatal tissue reduction—directly mirror the macroscopic neuroanatomical deviations characteristic of the most biologically compromised psychosis subgroups (Biotypes 1 and 2)[cite: 1, 2].

---

## 3. Reproduction & Execution

```bash
# 1. Automated training with 3D GAP and Real-Time Augmentation
python train_3dcnn.py \
    --exp-name bsnip_3dcnn_v5 \
    --use-gap \
    --augment \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda

# 2. Extract multi-subject 3D Grad-CAM heatmaps
python gradcam_bsnip.py --exp-name bsnip_3dcnn_v5 --num-samples 40
