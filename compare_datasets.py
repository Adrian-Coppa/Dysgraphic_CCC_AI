"""
compare_datasets.py — Compare les performances du modèle sur :
  1. Le test set CCC
  2. Des images d'écriture dysgraphique personnelle

Ce script est la PHASE 2 du projet.
Il suppose que le modèle a déjà été entraîné (best_model.pt existe).

Structure attendue pour les données dysgraphiques :
    custom_data/
        A/  lettre_a_1.png  lettre_a_2.png …
        B/  lettre_b_1.png …
        …

Utilisation :
    python compare_datasets.py --custom-dir ./custom_data
    python compare_datasets.py --custom-dir ./custom_data --retrain  # fine-tuning
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.cnn_model import build_model
from utils.data_loader import LABEL_TO_CHAR, load_ccc, load_custom_dataset, load_custom_chr
from utils.evaluator import full_evaluation
from utils.trainer import train


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(num_classes=len(LABEL_TO_CHAR))
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    return model.to(device).eval()


def compare(
    ccc_metrics: dict,
    custom_metrics: dict,
    save_dir: str = "./results",
):
    """Génère un graphique comparatif par classe."""
    save_dir = Path(save_dir)
    ccc_acc  = ccc_metrics["per_class_acc"]
    custom_acc  = custom_metrics["per_class_acc"]
    common_cls  = sorted(set(ccc_acc) & set(custom_acc))

    x     = np.arange(len(common_cls))
    width = 0.35
    e_vals = [ccc_acc[c] * 100  for c in common_cls]
    c_vals = [custom_acc[c]  * 100 for c in common_cls]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width/2, e_vals, width, label="CCC standard", color="#1565c0", alpha=0.85)
    ax.bar(x + width/2, c_vals, width, label="Écriture dysgraphique", color="#c62828", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(common_cls)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        "Comparaison de la reconnaissance : CCC vs écriture dysgraphique",
        fontsize=13,
    )
    ax.legend()
    ax.axhline(np.mean(e_vals), color="#1565c0", linestyle="--", alpha=0.5,
               label=f"Moy. CCC {np.mean(e_vals):.1f}%")
    ax.axhline(np.mean(c_vals), color="#c62828", linestyle="--", alpha=0.5,
               label=f"Moy. dysgraphique {np.mean(c_vals):.1f}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = save_dir / "comparison_CCC_vs_dysgraphic.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[Comparaison] Graphique sauvegardé → {path}")

    # Résumé texte
    print(f"\n{'─'*45}")
    print(f"  Accuracy globale CCC           : {ccc_metrics['accuracy']:.2%}")
    print(f"  Accuracy globale Dysgraphique  : {custom_metrics['accuracy']:.2%}")
    delta = custom_metrics['accuracy'] - ccc_metrics['accuracy']
    print(f"  Écart                          : {delta:+.2%}")
    print(f"{'─'*45}")


def parse_args():
    p = argparse.ArgumentParser(description="Comparaison CCC vs écriture dysgraphique")
    p.add_argument("--checkpoint",  default="./models/best_model.pt")
    p.add_argument("--data-dir",    default="./ccc")
    p.add_argument("--custom-dir",  required=True, help="Dossier avec images ou fichiers .chr")
    p.add_argument(
        "--custom-format", choices=("auto", "images", "chr"), default="auto",
        help="Format du dataset personnalisé (défaut: détection automatique)",
    )
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument(
        "--retrain",
        action="store_true",
        help="Fine-tune le modèle sur les données dysgraphiques puis réévalue",
    )
    p.add_argument("--retrain-epochs", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Comparaison des datasets")
    print(f"  Dispositif : {device}")
    print(f"{'='*55}\n")

    # ── Chargement du modèle ──────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Évaluation CCC ────────────────────────────────────────────
    print("[1/3] Évaluation sur CCC…")
    _, _, test_loader = load_ccc(args.data_dir, batch_size=args.batch_size, num_workers=2)
    ccc_metrics = full_evaluation(model, test_loader, device,
                                   save_dir=args.results_dir, dataset_name="CCC",
                                   class_names=list(LABEL_TO_CHAR.values()))

    # ── Évaluation sur données dysgraphiques ─────────────────────
    print("\n[2/3] Évaluation sur l'écriture dysgraphique…")
    custom_root = Path(args.custom_dir)
    has_chr = any(custom_root.rglob("training*.chr")) and any(custom_root.rglob("test*.chr"))
    use_chr = args.custom_format == "chr" or (args.custom_format == "auto" and has_chr)
    if use_chr:
        custom_loader = load_custom_chr(args.custom_dir, batch_size=args.batch_size)
    else:
        custom_loader = load_custom_dataset(args.custom_dir, batch_size=args.batch_size)
    custom_metrics = full_evaluation(model, custom_loader, device,
                                      save_dir=args.results_dir, dataset_name="Dysgraphique",
                                      class_names=list(LABEL_TO_CHAR.values()))

    # ── Fine-tuning optionnel ─────────────────────────────────────
    if args.retrain:
        print("\n[Fine-tuning] Ajustement du modèle sur l'écriture dysgraphique…")
        # On reutilise le loader custom en train/val (80/20)
        from torch.utils.data import random_split, DataLoader
        ds = custom_loader.dataset
        n_val  = max(1, int(0.2 * len(ds)))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        ft_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        ft_val   = DataLoader(val_ds,   batch_size=args.batch_size)

        train(
            model, ft_train, ft_val,
            epochs=args.retrain_epochs,
            lr=1e-4,           # LR bas pour le fine-tuning
            patience=5,
            checkpoint_dir=str(Path(args.results_dir) / "finetuned"),
            device=device,
        )
        ft_path = Path(args.results_dir) / "finetuned" / "best_model.pt"
        model = load_model(str(ft_path), device)

        print("\n[Fine-tuning] Réévaluation après ajustement…")
        ft_metrics = full_evaluation(model, custom_loader, device,
                                      save_dir=args.results_dir, dataset_name="Dysgraphique_FT",
                                      class_names=list(LABEL_TO_CHAR.values()))
        print(f"\n  Accuracy avant fine-tuning : {custom_metrics['accuracy']:.2%}")
        print(f"  Accuracy après fine-tuning : {ft_metrics['accuracy']:.2%}")
        custom_metrics = ft_metrics

    # ── Graphique comparatif ──────────────────────────────────────
    print("\n[3/3] Génération du graphique comparatif…")
    compare(ccc_metrics, custom_metrics, save_dir=args.results_dir)


if __name__ == "__main__":
    main()
