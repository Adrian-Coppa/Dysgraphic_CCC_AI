"""
Point d'entrée principal : entraîne le CNN sur le dataset CCC.

Utilisation :
    python train.py                          # paramètres par défaut
    python train.py --epochs 50 --lr 3e-3   # personnalisé
    python train.py --fast                   # 5 epochs pour tester
"""

import argparse
import json
import sys
import torch
from pathlib import Path

# Ajout du dossier courant au path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.cnn_model import build_model
from utils.data_loader import LABEL_TO_CHAR, load_ccc, load_custom_chr
from utils.trainer import train
from utils.evaluator import full_evaluation, plot_training_history


def parse_args():
    p = argparse.ArgumentParser(description="Entraîne le CNN sur le dataset CCC")
    p.add_argument("--data-dir",    default="./ccc",    help="Dossier contenant training.chr et test.chr")
    p.add_argument("--model-dir",   default="./models",  help="Dossier pour les checkpoints")
    p.add_argument("--results-dir", default="./results", help="Dossier pour les résultats")
    p.add_argument("--epochs",      type=int,   default=30,   help="Nombre d'epochs max")
    p.add_argument("--batch-size",  type=int,   default=128,  help="Taille de batch")
    p.add_argument("--lr",          type=float, default=1e-3, help="Learning rate max")
    p.add_argument("--dropout",     type=float, default=0.4,  help="Taux de dropout")
    p.add_argument("--patience",    type=int,   default=7,    help="Patience early stopping")
    p.add_argument("--workers",     type=int,   default=2,    help="DataLoader workers")
    p.add_argument(
        "--dataset", choices=("ccc", "custom"), default=None,
        help="Dataset d'entraînement (si absent, choix interactif)",
    )
    p.add_argument(
        "--custom-dir", default="./custom_data_processed",
        help="Racine du dataset personnalisé .chr",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Après l'entraînement, évalue le modèle sur CCC et le dataset personnalisé",
    )
    p.add_argument("--fast",        action="store_true",      help="Mode rapide (5 epochs)")
    p.add_argument("--eval-only",   action="store_true",      help="Évaluation seule (charge best_model.pt)")
    return p.parse_args()


def choose_dataset() -> str:
    print("Choisis le dataset d'entraînement :")
    print("  1. CCC")
    print("  2. Dataset personnalisé (.chr)")
    while True:
        choice = input("Choix (1/2) : ").strip().lower()
        if choice in {"1", "ccc"}:
            return "ccc"
        if choice in {"2", "custom", "personnalise", "personnalisé"}:
            return "custom"
        print("Choix invalide. Entre 1 ou 2.")


def choose_comparison() -> bool:
    while True:
        choice = input("Faire une comparaison croisée CCC/personnalisé ? (o/n) : ").strip().lower()
        if choice in {"o", "oui", "y", "yes"}:
            return True
        if choice in {"n", "non", "no"}:
            return False
        print("Réponse invalide. Entre o ou n.")


def main():
    args = parse_args()
    args.dataset = args.dataset or choose_dataset()
    if not args.compare and sys.stdin.isatty() and not args.eval_only:
        args.compare = choose_comparison()

    dataset_label = "CCC" if args.dataset == "ccc" else "Custom"
    checkpoint_dir = (
        Path(args.model_dir) / "custom"
        if args.dataset == "custom"
        else Path(args.model_dir)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.fast:
        args.epochs  = 5
        args.patience = 5
        print("[Mode rapide] 5 epochs seulement.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Reconnaissance de lettres manuscrites – {dataset_label}")
    print(f"  Dispositif : {device}")
    print(f"{'='*55}\n")

    # ── Données ──────────────────────────────────────────────────
    print(f"[1/3] Chargement des données {dataset_label}…")
    loader_fn = load_ccc if args.dataset == "ccc" else load_custom_chr
    loader_root = args.data_dir if args.dataset == "ccc" else args.custom_dir
    train_loader, val_loader, test_loader = loader_fn(
        data_dir=loader_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # ── Modèle ───────────────────────────────────────────────────
    model = build_model(num_classes=len(LABEL_TO_CHAR), dropout=args.dropout)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[2/3] Modèle CNN — {total_params:,} paramètres")

    best_model_path = checkpoint_dir / "best_model.pt"

    if args.eval_only:
        if not best_model_path.exists():
            print(f"[Erreur] Pas de modèle trouvé à {best_model_path}.")
            sys.exit(1)
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[EvalOnly] Modèle chargé (val_acc={ckpt.get('val_acc', 'N/A'):.4f})")
    else:
        # ── Entraînement ──────────────────────────────────────────
        print(f"\n[2/3] Entraînement ({args.epochs} epochs max, patience={args.patience})…\n")
        history = train(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            checkpoint_dir=str(checkpoint_dir),
            device=device,
        )

        # Courbes d'entraînement
        print("\n[3/3] Génération des graphiques…")
        plot_training_history(history, save_dir=args.results_dir)

        # Rechargement du meilleur modèle pour l'évaluation finale
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Best model rechargé (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # ── Évaluation finale sur le test set ────────────────────────
    print("\n[3/3] Évaluation finale sur le test set…")
    metrics = full_evaluation(
        model, test_loader, device,
        save_dir=args.results_dir,
        dataset_name=dataset_label,
        class_names=list(LABEL_TO_CHAR.values()),
    )

    if args.compare:
        print("\n[Comparaison] Évaluation croisée sur les deux datasets…")
        if args.dataset == "ccc":
            other_loader = load_custom_chr(
                args.custom_dir, batch_size=args.batch_size, num_workers=args.workers
            )[2]
            other_name = "Custom"
        else:
            other_loader = load_ccc(
                args.data_dir, batch_size=args.batch_size, num_workers=args.workers
            )[2]
            other_name = "CCC"
        other_metrics = full_evaluation(
            model, other_loader, device,
            save_dir=args.results_dir,
            dataset_name=f"{other_name}_cross",
            class_names=list(LABEL_TO_CHAR.values()),
        )
        print(f"  Accuracy {dataset_label} : {metrics['accuracy']:.2%}")
        print(f"  Accuracy {other_name}    : {other_metrics['accuracy']:.2%}")

    # Résumé final
    print(f"\n{'='*55}")
    print(f"  Accuracy finale sur le test set : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Fichiers générés dans : {args.results_dir}/")
    print(f"{'='*55}\n")

    # Sauvegarde des métriques JSON
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        # Convertir les valeurs numpy pour JSON
        serializable = {
            "accuracy": metrics["accuracy"],
            "per_class_acc": {k: float(v) for k, v in metrics["per_class_acc"].items()},
        }
        json.dump(serializable, f, indent=2)

    return metrics["accuracy"]


if __name__ == "__main__":
    main()
