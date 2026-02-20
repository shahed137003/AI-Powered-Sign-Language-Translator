# File: src/evaluation/eval_tcn.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter

from src.configs.tcn_config import *
from src.data.dataloaders import build_dataloaders
from src.models.sign2text.tcn import TCN
from src.models.losses import SmoothCE

def print_and_save_model_structure(model, save_path):
    lines = []
    lines.append("\n" + "="*50)
    lines.append("        TEMPORAL CONVOLUTIONAL NETWORK")
    lines.append("="*50 + "\n")
    lines.append(f"Input: Shape: (Batch, {FEATURE_DIM}, {TARGET_FRAMES})\n")
    lines.append("Temporal Blocks:")

    for i, block in enumerate(model.tcn):
        conv1 = block.net[0]
        conv2 = block.net[3]
        lines.append(f"\n  ┌─ Block {i+1}")
        lines.append(f"  │  Conv1: {conv1.in_channels} → {conv1.out_channels} | k=3 | dilation={conv1.dilation[0]}")
        lines.append(f"  │  Conv2: {conv2.in_channels} → {conv2.out_channels} | k=3 | dilation={conv2.dilation[0]}")
        lines.append(f"  │  Residual: {'Conv1x1' if isinstance(block.res, torch.nn.Conv1d) else 'Identity'}")
        lines.append(f"  └────────────────────────")

    lines.append("\nMasked Global Average Pooling")
    lines.append(f"  Output: (Batch, 192)")
    lines.append("\nFully Connected:")
    lines.append(f"  Linear: 192 → {model.fc.out_features}")

    total_params = sum(p.numel() for p in model.parameters())
    lines.append("\n" + "-"*50)
    lines.append(f"Total Parameters: {total_params:,}")
    lines.append("="*50 + "\n")

    structure_str = "\n".join(lines)
    print(structure_str)

    # Save to file (UTF-8 to handle box-drawing characters)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(structure_str)
    print(f"Saved model structure to {save_path}")


def main():
    MODEL_NAME = "tcn"  # fixed for this script
    print(f"Evaluating {MODEL_NAME} model using checkpoint: {MODEL_SAVE_PATH}")

    # ---------------------------
    # Load data
    # ---------------------------
    _, _, test_loader, num_classes = build_dataloaders()

    # ---------------------------
    # Load TCN model
    # ---------------------------
    model = TCN(FEATURE_DIM, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # Print and save model structure
    print_and_save_model_structure(model, f"experiments/plots/model_structure_{MODEL_NAME}.txt")

    criterion = SmoothCE(LABEL_SMOOTH)

    # ---------------------------
    # Run evaluation
    # ---------------------------
    all_preds = []
    all_targets = []
    loss_sum = 0
    total = 0
    all_top5_correct = 0

    with torch.no_grad():
        for x, m, y in test_loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            logits = model(x, m)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)

            # Top-1
            preds = probs.argmax(dim=1)

            # Top-5
            top5 = torch.topk(probs, k=5, dim=1).indices
            all_top5_correct += top5.eq(y.view(-1,1)).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

    # ---------------------------
    # Compute metrics
    # ---------------------------
    test_loss = loss_sum / total
    top1_acc = accuracy_score(all_targets, all_preds)
    top5_acc = all_top5_correct / total
    precision_macro = precision_score(all_targets, all_preds, average='macro')
    recall_macro = recall_score(all_targets, all_preds, average='macro')
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    precision_weighted = precision_score(all_targets, all_preds, average='weighted')
    recall_weighted = recall_score(all_targets, all_preds, average='weighted')
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    print("\n==============================")
    print(f"TEST LOSS   : {test_loss:.4f}")
    print(f"TOP-1 ACC   : {top1_acc:.4f}")
    print(f"TOP-5 ACC   : {top5_acc:.4f}")
    print("==============================")
    print("\nMacro Avg:")
    print(f"Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f} | F1: {f1_macro:.4f}")
    print("\nWeighted Avg:")
    print(f"Precision: {precision_weighted:.4f} | Recall: {recall_weighted:.4f} | F1: {f1_weighted:.4f}")

    # ---------------------------
    # Confusion Matrix for top 20 classes
    # ---------------------------
    all_targets_np = np.array(all_targets)
    top20_classes = [cls for cls, _ in Counter(all_targets_np).most_common(20)]

    cm = confusion_matrix(all_targets_np, np.array(all_preds), labels=top20_classes)

    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title(f"Confusion Matrix ({MODEL_NAME}) - Top 20 Classes")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(20), labels=top20_classes, rotation=90)
    plt.yticks(ticks=np.arange(20), labels=top20_classes)
    plt.tight_layout()
    plt.savefig(f"experiments/plots/confusion_matrix_{MODEL_NAME}_top20.png")
    print(f"\nSaved top-20 confusion matrix to experiments/plots/confusion_matrix_{MODEL_NAME}_top20.png")

    # ---------------------------
    # Save metrics
    # ---------------------------
    metrics = {
        "test_loss": test_loss,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }

    Path("experiments/plots").mkdir(parents=True, exist_ok=True)
    with open(f"experiments/plots/test_metrics_{MODEL_NAME}.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to experiments/plots/test_metrics_{MODEL_NAME}.json")


if __name__ == "__main__":
    main()