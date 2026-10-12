"""
Évaluation détaillée du modèle :
  - Accuracy globale et par classe
  - Matrice de confusion
  - Rapport de classification (précision / rappel / F1)
  - Visualisation des erreurs les plus fréquentes
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

LABEL_TO_CHAR = {i: chr(65 + i) for i in range(26)}


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Retourne (y_true, y_pred, proba) pour tout le loader."""
    model.eval()
    all_preds, all_labels, all_proba = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_proba.extend(proba)

    return np.array(all_labels), np.array(all_preds), np.array(all_proba)


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str = "./results",
    dataset_name: str = "CCC",
    class_names: Optional[list] = None,
) -> dict:
    """
    Évalue le modèle et génère :
      1. Rapport texte (précision/rappel/F1 par classe)
      2. Matrice de confusion (PNG)
      3. Graphique top-5 erreurs (PNG)
    
    Retourne un dictionnaire avec les métriques globales.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, proba = get_predictions(model, test_loader, device)

    accuracy = (y_true == y_pred).mean()
    print(f"\n{'='*50}")
    print(f"  Évaluation sur {dataset_name}")
    print(f"{'='*50}")
    print(f"  Accuracy globale : {accuracy:.4f} ({accuracy*100:.2f}%)")

    # --- Rapport par classe ---
    if class_names is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1 if len(y_true) else 1
        class_names = [LABEL_TO_CHAR.get(i, str(i)) for i in range(num_classes)]
    else:
        class_names = list(class_names)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nRapport par classe :\n")
    print(report)

    report_path = save_dir / f"{dataset_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n\n")
        f.write(report)

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, values_format=".2f", cmap="Blues")
    ax.set_title(f"Matrice de confusion – {dataset_name} (normalisée)", fontsize=14)
    plt.tight_layout()
    cm_path = save_dir / f"{dataset_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Matrice de confusion → {cm_path}")

    # --- Top erreurs par classe ---
    per_class_acc = {}
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        per_class_acc[LABEL_TO_CHAR[i]] = (y_true[mask] == y_pred[mask]).mean()

    sorted_classes = sorted(per_class_acc, key=per_class_acc.get)
    worst_5  = sorted_classes[:5]
    best_5   = sorted_classes[-5:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_bad  = ["#d32f2f"] * 5
    colors_good = ["#388e3c"] * 5

    axes[0].barh(worst_5, [per_class_acc[c] for c in worst_5], color=colors_bad)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("5 classes les plus difficiles")
    axes[0].set_xlim(0, 1)
    for i, (c, v) in enumerate(zip(worst_5, [per_class_acc[c] for c in worst_5])):
        axes[0].text(v + 0.01, i, f"{v:.2%}", va="center")

    axes[1].barh(best_5, [per_class_acc[c] for c in best_5], color=colors_good)
    axes[1].set_xlabel("Accuracy")
    axes[1].set_title("5 classes les mieux reconnues")
    axes[1].set_xlim(0, 1)
    for i, (c, v) in enumerate(zip(best_5, [per_class_acc[c] for c in best_5])):
        axes[1].text(v + 0.01, i, f"{v:.2%}", va="center")

    fig.suptitle(f"Performances par classe – {dataset_name}", fontsize=13)
    plt.tight_layout()
    per_class_path = save_dir / f"{dataset_name}_per_class.png"
    plt.savefig(per_class_path, dpi=150)
    plt.close()
    print(f"  Performances par classe → {per_class_path}")

    return {
        "accuracy": float(accuracy),
        "per_class_acc": per_class_acc,
        "report_path": str(report_path),
        "cm_path": str(cm_path),
        "per_class_path": str(per_class_path),
    }


def plot_training_history(history: dict, save_dir: str = "./results"):
    """Trace les courbes loss/accuracy d'entraînement."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history["train_loss"], label="Train", color="#1565c0")
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="#c62828", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Courbe de perte")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", color="#1565c0")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Validation", color="#c62828", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Courbe d'accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    best_val = max(history["val_acc"])
    ax2.axhline(y=best_val * 100, color="green", linestyle=":", alpha=0.6,
                label=f"Best val: {best_val:.2%}")
    ax2.legend()

    plt.suptitle("Historique d'entraînement – CCC", fontsize=13)
    plt.tight_layout()
    path = save_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Historique d'entraînement → {path}")
    return str(path)
