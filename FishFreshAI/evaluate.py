"""Evaluation script – compute metrics on a held-out test set."""

from typing import Dict

import torch
from torch.cuda import amp

from model import FishFreshEfficientNet
from utils import get_dataloaders


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation loop (no gradients)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int()
        correct += (preds.squeeze() == labels.squeeze()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def run_evaluation(
    model_path: str = "best_model.pth",
    data_dir: str = "./data",
    batch_size: int = 32,
    device: str = "auto",
) -> Dict[str, float]:
    """Load the best model and evaluate on the test split."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading model from {model_path} and evaluating on {device}...")

    # Model
    model = FishFreshEfficientNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Data
    loaders, sizes, _ = get_dataloaders(data_dir, batch_size=batch_size)
    test_loader = loaders["test"]

    # Criterion (same structure as training; pos_weight can be omitted for pure eval)
    criterion = torch.nn.BCEWithLogitsLoss()

    metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"Test Results → Loss: {metrics['loss']:.4f} | Accuracy: {metrics['acc']:.4f} "
        f"({sizes['test']} samples)"
    )
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FishFresh AI")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )