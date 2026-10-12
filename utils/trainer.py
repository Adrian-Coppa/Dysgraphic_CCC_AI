"""
Boucle d'entraînement, validation et sauvegarde du meilleur modèle.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    checkpoint_dir: str = "./models",
    device: Optional[torch.device] = None,
) -> Dict[str, List]:
    """
    Entraîne le modèle avec :
      - Optimiseur Adam + OneCycleLR
      - Early stopping sur la val_loss
      - Sauvegarde du meilleur modèle
      - Mixed precision si CUDA disponible
    
    Retourne l'historique des métriques.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Dispositif : {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    history: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_loss = float("inf")
    no_improve = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | {train_metrics['loss']:>10.4f} | "
            f"{train_metrics['acc']:>8.2%} | {val_metrics['loss']:>8.4f} | "
            f"{val_metrics['acc']:>6.2%} | {elapsed:>5.1f}s"
        )

        # Early stopping & sauvegarde
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_metrics["acc"],
                "val_loss": val_metrics["loss"],
            }, best_path)
            print(f"         ✓ Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[EarlyStopping] Aucune amélioration depuis {patience} epochs. Arrêt.")
                break

    # Sauvegarde de l'historique
    hist_path = checkpoint_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] Historique sauvegardé dans {hist_path}")
    print(f"[Train] Meilleur modèle : {best_path}")

    return history
