#!/usr/bin/env python
"""
count_grid_cells.py

Compte les lettres dans une image en la divisant en grille régulière.
Par défaut : 7 colonnes × 20 lignes = 140 cellules.
Pour chaque cellule, on vérifie si elle contient assez de pixels noirs.

Utilisation :
    python count_grid_cells.py                    # demande la lettre
    python count_grid_cells.py --letter A
    python count_grid_cells.py --letter B --cols 7 --rows 20 --visualize
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # ou "Agg"


def count_cells(image_path, cols=9, rows=12, visualize=False, save_vis_dir=None):
    """
    Découpe l'image en cols × rows et compte les cellules non vides.
    Retourne : (nombre_detecté, image_annotée_ou_None)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return -1, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inversion si fond clair
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Seuil binaire pour isoler le tracé
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dimensions des cellules
    cell_w = w // cols
    cell_h = h // rows

    count = 0
    # On stocke les coordonnées des cellules pleines pour la visualisation
    cells_ok = []

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cell = thresh[y1:y2, x1:x2]

            # Proportion de pixels noirs dans la cellule (255 = blanc après inversion)
            # On compte les pixels > 0 (tracé)
            total_pixels = cell.size
            black_pixels = np.count_nonzero(cell > 0)  # en niveau de gris, >0 = trait
            ratio = black_pixels / total_pixels

            # Seuil : on considère qu'il y a une lettre si plus de 2% de la cellule est remplie
            # (ajuster selon la taille de l'écriture)
            if ratio > 0.02:
                count += 1
                cells_ok.append((x1, y1, cell_w, cell_h))

    # Visualisation
    vis_img = None
    if visualize:
        vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for (x, y, cw, ch) in cells_ok:
            cv2.rectangle(vis_img, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            f"Detected: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        if save_vis_dir:
            save_vis_dir = Path(save_vis_dir)
            save_vis_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_vis_dir / f"vis_{image_path.name}"
            cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"   🖼️  Visualisation sauvegardée : {out_path}")

    return count, vis_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", type=str, help="Lettre à vérifier")
    parser.add_argument("--root", default="./custom_data", help="Dossier racine")
    parser.add_argument("--cols", type=int, default=9, help="Nombre de colonnes")
    parser.add_argument("--rows", type=int, default=12, help="Nombre de lignes")
    parser.add_argument("--expected", type=int, default=108, help="Nombre attendu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-dir", default="./results/grid_vis")
    args = parser.parse_args()

    if args.letter is None:
        letter = input("Quelle lettre voulez-vous vérifier ? (ex: A) : ").strip().upper()
    else:
        letter = args.letter.upper()

    letter_dir = Path(args.root) / letter
    if not letter_dir.exists():
        print(f"❌ Dossier {letter_dir} introuvable.")
        return

    extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [f for f in letter_dir.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        print(f"⚠️ Aucune image dans {letter_dir}")
        return

    print(f"\n Dossier : {letter_dir}")
    print(f"Nombre de fichiers : {len(image_files)}")
    print(f"" Grille : {args.cols}×{args.rows} = {args.cols*args.rows} cellules")
    print("-" * 60)

    total_detected = 0
    total_ok = 0
    results = []

    for img_path in sorted(image_files):
        detected, _ = count_cells(
            img_path,
            cols=args.cols,
            rows=args.rows,
            visualize=args.visualize,
            save_vis_dir=args.vis_dir if args.visualize else None,
        )

        if detected == -1:
            print(f"  ❌ {img_path.name} : erreur")
            continue

        status = "✅ OK" if detected == args.expected else "❌ KO"
        diff = detected - args.expected
        print(f"  {status}  {img_path.name} : {detected} (écart: {diff:+d})")
        results.append((img_path.name, detected, diff))
        total_detected += detected
        if detected == args.expected:
            total_ok += 1

    print("-" * 60)
    print(f"📊 Résumé pour {letter} :")
    print(f"   - Fichiers : {len(image_files)}, conformes : {total_ok}/{len(image_files)}")
    if total_ok < len(image_files):
        print("   - Problématiques :")
        for name, det, diff in results:
            if det != args.expected:
                print(f"       • {name} : {det} (diff: {diff:+d})")
    print(f"   - Total détecté : {total_detected}")
    print(f"   - Moyenne : {total_detected/len(image_files):.1f}")

    if args.visualize:
        print(f"\n🖼️  Visualisations dans : {args.vis_dir}/")


if __name__ == "__main__":
    main()