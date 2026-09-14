"""Training script for FishFresh AI."""

import argparse
import os
from typing import Dict, Any

import torch
from torch.cuda import amp

from utils import compute_class_weights, get_dataloaders
from model import FishFreshEfficientNet

# Mixed-precision scaler for faster training on GPU
scaler = amp.GradScaler()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
) -> Dict[str, float]:
    """Train for one epoch; returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)  # logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        # binary accuracy @ 0.5 threshold on sigmoid
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int()
        correct += (preds.squeeze() == labels.squeeze()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "acc": epoch_acc}


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
) -> Dict[str, float]:
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


def main(
    data_dir: str = "./data",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 7,
    device: str = "auto",
):
    """Run training with early stopping and best-model checkpoint."""

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # 1. Data loaders
    loaders, sizes, _ = get_dataloaders(data_dir, batch_size=batch_size)

    # 2. Model, loss, optimizer
    model = FishFreshEfficientNet(num_classes=1).to(device)

    # Class-imbalance weight from training set
    pos_weight = compute_class_weights(loaders["train"].dataset, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # LR scheduler – ReduceLROnPlateau based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Early-stopping tracker
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Checkpoint path
    ckpt_path = os.path.join(os.getcwd(), "best_model.pth")

    print(
        f"Training on {sizes['train']} images, "
        f"validating on {sizes['val']}, testing on {sizes['test']}"
    )

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, scaler, device
        )

        # ---- Evaluate ----
        val_metrics = evaluate(model, loaders["val"], criterion, device)

        # ---- Scheduler step ----
        scheduler.step(val_metrics["loss"])

        # ---- Early stopping logic ----
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            # Save best model weights
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ New best model saved (val loss {val_metrics['loss']:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        # ---- Logging ----
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f}"
        )

    # Load best weights at the end
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Best model loaded for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FishFresh AI – Binary classification")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )