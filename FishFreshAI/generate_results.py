"""Generate training graphs and confusion matrix for the report."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import FishFreshEfficientNet
from utils import get_transforms

# ── Training history (from actual training logs) ──────────────────────────────
train_loss = [0.6740, 0.3505, 0.2448, 0.1911, 0.1595, 0.1327, 0.1317, 0.1125,
              0.1024, 0.0843, 0.0828, 0.0758, 0.0848, 0.0670, 0.0655]
train_acc  = [0.8629, 0.9753, 0.9829, 0.9857, 0.9857, 0.9908, 0.9825, 0.9873,
              0.9869, 0.9920, 0.9900, 0.9928, 0.9869, 0.9904, 0.9948]
val_loss   = [0.4077, 0.1992, 0.1166, 0.0847, 0.0630, 0.0541, 0.0414, 0.0374,
              0.0292, 0.0266, 0.0212, 0.0184, 0.0177, 0.0155, 0.0139]
val_acc    = [0.9857, 0.9857, 0.9984, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
              1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
epochs = list(range(1, len(train_loss) + 1))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Training & Validation Accuracy ─────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(epochs, [a*100 for a in train_acc], 'b-o', label='Training Accuracy', markersize=4)
plt.plot(epochs, [a*100 for a in val_acc], 'r-s', label='Validation Accuracy', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Training and Validation Accuracy vs Epochs', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim(85, 101)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_graph.png'), dpi=150)
plt.close()
print('✓ Saved accuracy_graph.png')

# ── 2. Training & Validation Loss ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4)
plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss vs Epochs', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_graph.png'), dpi=150)
plt.close()
print('✓ Saved loss_graph.png')

# ── 3. Confusion Matrix ──────────────────────────────────────────────────────
device = torch.device('cpu')
model = FishFreshEfficientNet(num_classes=1).to(device)
model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'best_model.pth'), map_location=device))
model.eval()

transform = get_transforms(img_size=224)['test']
test_dataset = datasets.ImageFolder(root=os.path.join(PROJECT_ROOT, 'data', 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int().squeeze()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fresh', 'Fresh'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print('✓ Saved confusion_matrix.png')

# ── 4. Classification Report ──────────────────────────────────────────────────
report = classification_report(all_labels, all_preds, target_names=['Non-Fresh', 'Fresh'])
print('\n' + '='*50)
print('CLASSIFICATION REPORT')
print('='*50)
print(report)

# Save report to file
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write('FISH FRESHNESS DETECTION - CLASSIFICATION REPORT\n')
    f.write('='*50 + '\n\n')
    f.write(f'Model: EfficientNet-B0 (Transfer Learning)\n')
    f.write(f'Test Samples: {len(all_labels)}\n')
    f.write(f'Test Accuracy: {(all_preds == all_labels).mean()*100:.2f}%\n\n')
    f.write(report)
    f.write('\n' + '='*50 + '\n')
    f.write('Training Parameters:\n')
    f.write(f'  Epochs: 15\n')
    f.write(f'  Batch Size: 32\n')
    f.write(f'  Learning Rate: 1e-3\n')
    f.write(f'  Optimizer: AdamW\n')
    f.write(f'  Loss Function: BCEWithLogitsLoss\n')
    f.write(f'  LR Scheduler: ReduceLROnPlateau\n')
    f.write(f'  Early Stopping Patience: 7\n')

print('✓ Saved classification_report.txt')
print(f'\nAll results saved to: {OUTPUT_DIR}')
