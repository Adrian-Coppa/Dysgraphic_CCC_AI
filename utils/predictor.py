"""
Module d'inférence : prédit la lettre dans une image unique.
Peut être utilisé en ligne de commande ou importé par un autre script.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Ajoute le dossier parent au path si on exécute ce fichier directement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn_model import build_model
from utils.data_loader import get_inference_transform, LABEL_TO_CHAR


class LetterPredictor:
    """
    Prédit une ou plusieurs lettres à partir d'images.

    Usage :
        predictor = LetterPredictor("models/best_model.pt")
        letter, conf, top5 = predictor.predict_image("ma_lettre.png")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        num_classes: int = 26,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_inference_transform()

        self.model = build_model(num_classes=num_classes)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)   # gère les deux formats
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[Predictor] Modèle chargé depuis {checkpoint_path} sur {self.device}")

    @torch.no_grad()
    def predict_image(
        self,
        image_input,   # str (chemin) | PIL.Image | np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Prédit la lettre dans une image.

        Retourne :
            (lettre_prédite, confiance, top5)
            où top5 = [(lettre, proba), …] triées par proba décroissante
        """
        # Chargement de l'image
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        else:
            raise TypeError(f"Type non supporté : {type(image_input)}")

        # Le modèle attend des traits clairs sur fond sombre.
        img = _auto_invert(img)

        tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1,1,28,28)
        proba  = F.softmax(self.model(tensor), dim=1).squeeze().cpu().numpy()

        top5_idx  = proba.argsort()[::-1][:5]
        top5      = [(LABEL_TO_CHAR[i], float(proba[i])) for i in top5_idx]
        best_char = top5[0][0]
        best_conf = top5[0][1]

        return best_char, best_conf, top5

    def predict_and_show(
        self,
        image_input,
        save_path: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Prédit et génère une figure de visualisation."""
        char, conf, top5 = self.predict_image(image_input)

        # Reconstruire l'image pour l'affichage
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = image_input.convert("L")

        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

        ax_img.imshow(img, cmap="gray")
        ax_img.axis("off")
        ax_img.set_title(f"Prédiction : « {char} »\nConfiance : {conf:.1%}", fontsize=14)

        letters = [t[0] for t in top5]
        probas  = [t[1] * 100 for t in top5]
        colors  = ["#1565c0" if l == char else "#90a4ae" for l in letters]
        bars    = ax_bar.barh(letters[::-1], probas[::-1], color=colors[::-1])
        ax_bar.set_xlabel("Probabilité (%)")
        ax_bar.set_title("Top-5 prédictions")
        ax_bar.set_xlim(0, 105)
        for bar, val in zip(bars, probas[::-1]):
            ax_bar.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[Predictor] Figure sauvegardée → {save_path}")
        plt.close()

        return char, conf


def _auto_invert(img: Image.Image) -> Image.Image:
    """
    Le chargeur du CCC convertit les caractères en traits clairs sur fond
    sombre. On inverse une image externe à fond clair pour l'aligner dessus.
    """
    arr = np.array(img)
    if arr.mean() > 127:          # fond clair → on inverse
        img = ImageOps.invert(img)
    return img


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prédit la lettre dans une image.")
    parser.add_argument("image", help="Chemin vers l'image")
    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pt",
        help="Chemin vers le checkpoint du modèle",
    )
    parser.add_argument("--save", default=None, help="Où sauvegarder la figure")
    args = parser.parse_args()

    predictor = LetterPredictor(args.checkpoint)
    char, conf, top5 = predictor.predict_image(args.image)

    print(f"\nPrédiction : « {char} »  ({conf:.1%} de confiance)")
    print("\nTop-5 :")
    for rank, (l, p) in enumerate(top5, 1):
        print(f"  {rank}. {l} — {p:.2%}")

    if args.save or True:
        save_path = args.save or "results/prediction.png"
        predictor.predict_and_show(args.image, save_path=save_path)
