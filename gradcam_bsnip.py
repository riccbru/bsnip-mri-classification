"""gradcam_bsnip.py

Generate 3D Grad-CAM saliency maps for the trained BSNIP HC vs SZ 3D CNN
(best_bsnip_3dcnn.pth), mirroring gradcam_visualize.py / model_3dcnn_gradcam.py.

For a sample of test-set subjects (default 5 HC + 5 SZ, from the same
subject-level split get_bsnip_dataloaders() produces at --seed), this:
    1. Loads BSNIP3DCNN (from train_3dcnn.py) and the trained weights.
    2. Attaches forward/backward hooks to the final conv layer (conv3) to
       capture activations and gradients for Grad-CAM.
    3. Backprops from the predicted-class logit, forms the CAM, normalizes
       it to [0, 1], and trilinearly resizes it to the input volume's native
       shape (121, 145, 121) for BSNIP's SPM12-normalized Grey Matter maps).
    4. Renders Axial / Coronal / Sagittal mid-slices of the volume with the
       heatmap overlaid, and saves one figure per subject to
       gradcam_results_bsnip/.

Orientation note: .npy volumes carry no NIfTI affine/header, so the mapping
from array axis -> anatomical plane can't be read back out; AXIS_VIEW_MAP
below assumes the standard SPM/MNI (X, Y, Z) = (sagittal, coronal, axial)
voxel ordering typical of these dimensions. Adjust it if your data differs.

Usage:
    python gradcam_bsnip.py --num-samples 5 --output-dir gradcam_results_bsnip --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bsnip_dataset import LABEL_NAMES, get_bsnip_dataloaders
from train_3dcnn import NUM_CLASSES, BSNIP3DCNN, infer_input_shape

logger = logging.getLogger("gradcam_bsnip")

DEFAULT_MODEL_PATH = Path("best_bsnip_3dcnn.pth")
DEFAULT_METADATA_CSV = Path("bsnip_preprocessed_npy_metadata.csv")
DEFAULT_OUTPUT_DIR = Path("gradcam_results_bsnip")

# array axis -> (view name, axis index) for a (D, H, W) volume shaped like
# BSNIP's SPM12-normalized (121, 145, 121) Grey Matter maps; see module
# docstring for the orientation caveat.
AXIS_VIEW_MAP: dict[str, int] = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


class GradCAMExtractor:
    """Captures activations/gradients at a target layer via forward+backward hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def compute_cam(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, int, np.ndarray]:
        """Forward + backward on the predicted class; return (raw CAM, pred class, class probs)."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().squeeze(0)
        pred_class = int(np.argmax(probs))

        logits[0, pred_class].backward()

        # Global-average-pool the gradients over (D, H, W) -> per-channel weights.
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, d, h, w]
        cam = F.relu(cam)
        return cam, pred_class, probs


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a CAM tensor to [0, 1]."""
    cam_min, cam_max = cam.min(), cam.max()
    return (cam - cam_min) / (cam_max - cam_min + 1e-8)


def resize_cam(cam: torch.Tensor, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Trilinearly resize a [1, 1, d, h, w] CAM to target_shape and return as numpy [D, H, W]."""
    resized = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def select_samples(df: pd.DataFrame, num_per_class: int, seed: int) -> pd.DataFrame:
    """Sample up to num_per_class subjects per label (0=HC, 1=SZ) from the test split."""
    parts = []
    for label, name in LABEL_NAMES.items():
        available = df[df["label"] == label]
        n = min(num_per_class, len(available))
        if n < num_per_class:
            logger.warning("Only %d %s subjects available in the test split (requested %d)", n, name, num_per_class)
        parts.append(available.sample(n=n, random_state=seed))
    return pd.concat(parts)


def plot_gradcam(
    volume: np.ndarray,
    cam: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Render Axial/Coronal/Sagittal mid-slices of `volume` with `cam` overlaid, and save."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (view_name, axis) in zip(axes, AXIS_VIEW_MAP.items()):
        mid = volume.shape[axis] // 2
        vol_slice = np.rot90(np.take(volume, mid, axis=axis))
        cam_slice = np.rot90(np.take(cam, mid, axis=axis))

        ax.imshow(vol_slice, cmap="gray")
        ax.imshow(cam_slice, cmap="jet", alpha=0.45)
        ax.set_title(view_name)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D Grad-CAM overlays for the BSNIP 3D CNN.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                         help="Path to the trained checkpoint (default: %(default)s)")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV,
                         help="Path to bsnip_preprocessed_npy_metadata.csv (default: %(default)s)")
    parser.add_argument("--num-samples", type=int, default=5,
                         help="Number of subjects per class (HC, SZ) to visualize (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Directory to save Grad-CAM figures into (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for the train/val/test split (must match training) and sample selection (default: %(default)s)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                         help="Device to run on (default: cuda if available, else cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_bsnip_dataloaders(
        metadata_csv=args.metadata_csv, batch_size=1, num_workers=0, random_state=args.seed,
    )
    dataset = test_loader.dataset
    logger.info("Test split size: %d", len(dataset))

    input_shape = infer_input_shape(test_loader)
    logger.info("Inferred volume input shape (D, H, W): %s", input_shape)

    model = BSNIP3DCNN(input_shape=input_shape, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    extractor = GradCAMExtractor(model, model.conv3)

    selected = select_samples(dataset.data, args.num_samples, args.seed)
    logger.info("Selected %d subjects for Grad-CAM visualization", len(selected))

    n_saved = 0
    for idx, row in selected.iterrows():
        subject_id, site, true_label = row["subject_id"], row["site"], int(row["label"])
        try:
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)  # [1, 1, D, H, W]

            cam, pred_class, probs = extractor.compute_cam(img_batch)
            cam = normalize_cam(cam)
            target_shape = tuple(img.shape[-3:])  # native volume shape, e.g. (121, 145, 121)
            cam_resized = resize_cam(cam, target_shape)

            volume = img.squeeze(0).cpu().numpy()  # [D, H, W]

            true_name = LABEL_NAMES[true_label]
            pred_name = LABEL_NAMES[pred_class]
            title = (
                f"{subject_id} ({site}) | true={true_name} pred={pred_name} "
                f"(p_SZ={probs[1]:.3f})"
            )
            out_path = args.output_dir / f"{subject_id}_true-{true_name}_pred-{pred_name}.png"
            plot_gradcam(volume, cam_resized, title, out_path)
            logger.info("Saved Grad-CAM for %s -> %s", subject_id, out_path)
            n_saved += 1
        except Exception as exc:  # noqa: BLE001 - log and continue past a bad subject
            logger.error("Failed to generate Grad-CAM for subject %s: %s", subject_id, exc)

    logger.info("Done: %d/%d Grad-CAM figures saved to %s", n_saved, len(selected), args.output_dir)


if __name__ == "__main__":
    main()
