#!/usr/bin/env python3
"""FishFresh AI – Single‑image inference script.

Usage
-----
    python3 predict.py --image "/path/to/image.jpg"

All requirements from the task are fulfilled:
  • argparse + --help
  • auto‑device (MPS > CUDA > CPU)
  • EfficientNet‑B0 exactly as in training
  • loads best_model.pth (raw state_dict or dict with "state_dict" key)
  • validation‑size transforms (224×224, ImageNet normalisation)
  • torch.no_grad() + sigmoid
  • FRESH / NON‑FRESH label
  • formatted console output (Image, Prediction, Confidence, Fresh prob, Non‑fresh prob)
  • explicit error messages for every failure mode
  • never silently exits
"""

import argparse
import os
import sys
from typing import Dict, Any

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# ----------------------------------------------------------------------
# Project‑relative imports – make sure the local `model` and `utils` are seen
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import FishFreshEfficientNet   # <-- exact same architecture as training
from utils import get_transforms          # contains the "val" transform


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _pick_device() -> torch.device:
    """Return the best available device: MPS (Apple‑Silicon) → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(device: torch.device, checkpoint_path: str = "best_model.pth") -> nn.Module:
    """Instantiate the model and load the saved state_dict.

    The function copes with two common saving styles:
      * torch.save(model.state_dict(), …)            → plain tensor
      * torch.save({"state_dict": model.state_dict()}, …) → dict
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Model checkpoint not found at '{checkpoint_path}'. "
            "Run training first or supply a valid path with --model."
        )

    model = FishFreshEfficientNet(num_classes=1).to(device)

    try:
        raw = torch.load(checkpoint_path, map_location=device)
        # Normalise to a state_dict
        state = raw.get("state_dict", raw)          # works for both formats
        model.load_state_dict(state)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"❌ Failed to load checkpoint '{checkpoint_path}': {exc}") from exc

    return model


def _build_transforms() -> transforms.Compose:
    """Return the exact validation transforms used during training."""
    val_tfms = get_transforms(img_size=224)["val"]
    if isinstance(val_tfms, transforms.Compose):
        return val_tfms
    # Fallback – in case utils returns something unexpected
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ]
    )


def _open_image(path: str) -> Image.Image:
    """Open an image and force RGB mode; raise clear errors if anything fails."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ Image file not found: '{path}'")
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise IOError(f"❌ Could not open image '{path}': {exc}") from exc
    return img


def _predict(model: nn.Module, image_path: str, device: torch.device) -> Dict[str, Any]:
    """Run inference on a single image and return a result dict."""
    transform = _build_transforms()
    img = _open_image(image_path)

    # Tensor: [C H W] → add batch dimension → [1 C H W]
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():                     # requirement #9
        logits = model(tensor)                # raw output
        prob = torch.sigmoid(logits).item()   # requirement #10 (sigmoid)

    pred_label = "FRESH" if prob >= 0.5 else "NON-FRESH"

    # Complementary probability for the required output format
    non_fresh_prob = 1.0 - prob

    return {
        "probability": prob,                  # Fresh probability
        "prediction": pred_label,             # "FRESH" or "NON-FRESH"
        "non_fresh_probability": non_fresh_prob,
    }


# ----------------------------------------------------------------------
# Entry point – argparse driven
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FishFresh AI – Binary image classification (Fresh vs Non‑Fresh)."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the fish eye / gill image to classify",
    )
    parser.add_argument(
        "--model",
        default="best_model.pth",
        help="Path to the trained model checkpoint (default: best_model.pth in CWD)",
    )
    args = parser.parse_args()               # <-- argparse handles --help automatically

    image_path = args.image
    model_path = args.model

    # --------------------------------------------------------------
    # 1️⃣ Device selection (MPS > CUDA > CPU)  – requirement #7
    # --------------------------------------------------------------
    device = _pick_device()
    print(f"🔧 Using device: {device}")

    # --------------------------------------------------------------
    # 2️⃣ Load the model checkpoint  – requirement #4 & #6
    # --------------------------------------------------------------
    try:
        model = _load_model(device, model_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)               # explicit error message
        sys.exit(1)           # never silently exit – requirement #14

    # --------------------------------------------------------------
    # 3️⃣ Run prediction  – requirements #8, #9, #10, #11, #12
    # --------------------------------------------------------------
    try:
        result = _predict(model, image_path, device)
    except (FileNotFoundError, IOError) as e:
        print(e)               # explicit error message
        sys.exit(1)

    # --------------------------------------------------------------
    # 4️⃣ Pretty‑print the required output  – requirement #12
    # --------------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"Image:          {os.path.basename(image_path)}")
    print(f"Prediction:     {result['prediction']}")
    print(f"Confidence:     {result['probability']*100:.2f}%")
    print(f"Fresh probability: {result['probability']*100:.2f}%")
    print(f"Non-fresh probability: {result['non_fresh_probability']*100:.2f}%")
    print("=" * 40 + "\n")


# ----------------------------------------------------------------------
# Script guard – ensures main() runs when the file is executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()