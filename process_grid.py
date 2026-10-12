#!/usr/bin/env python3
"""
process_grid.py
----------------
Traite toutes les photos de /custom_data/[lettre]/*.jpg :
  1. Affiche chaque image, tu cliques les 4 coins de la zone UTILE de la
     grille (celle où les 12x9 cases sont complètement formées).
  2. Redresse la perspective (corrige l'angle d'impression / de prise de photo).
  3. Découpe en 12 lignes x 9 colonnes.
  4. Sauvegarde chaque case dans sa résolution de découpe, sans normalisation
      spécifique à un dataset.
  5. Sauvegarde les cases dans /custom_data_processed/[lettre]/<image>_r{row}_c{col}.png

Les coins cliqués sont mis en cache dans un fichier JSON à côté de chaque
image (mêmes noms + .json) : si tu relances le script, il ne te redemande
pas de recliquer une image déjà traitée (sauf option --force).

Contrôles pendant le clic :
  - Clic gauche : ajouter un point (dans l'ordre : Haut-Gauche, Haut-Droit,
    Bas-Droit, Bas-Gauche)
  - 'r'          : réinitialiser les 4 points sur l'image en cours
  - 'entrée'     : valider les 4 points et passer à l'image suivante
  - 's'          : sauter cette image (ne pas la traiter)
  - 'q'          : quitter le programme

Usage :
    python3 process_grid.py --input /custom_data --output /custom_data_processed
    python3 process_grid.py --input /custom_data --output /custom_data_processed --force
    python3 process_grid.py --input /custom_data --output /custom_data_processed --rows 12 --cols 9
    python3 process_grid.py --input /custom_data --output /custom_data_processed --test-ratio 0.333333
    python3 process_grid.py --input /custom_data --output /custom_data_processed --display-max-dim 0
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROWS_DEFAULT = 12
COLS_DEFAULT = 9
DISPLAY_MAX_DIM = 1800  # taille max d'affichage ; 0 conserve la résolution native


# --------------------------------------------------------------------------- #
# Sélection interactive des 4 coins
# --------------------------------------------------------------------------- #
class CornerPicker:
    def __init__(self, img, max_display_dim=DISPLAY_MAX_DIM):
        self.orig = img
        h, w = img.shape[:2]
        largest_dimension = max(h, w)
        self.scale = (
            1.0
            if max_display_dim <= 0
            else min(1.0, max_display_dim / largest_dimension)
        )
        if self.scale < 1.0:
            display_size = (round(w * self.scale), round(h * self.scale))
            self.display = cv2.resize(img, display_size, interpolation=cv2.INTER_AREA)
        else:
            self.display = img.copy()
        self.points = []
        self.window = "Clique les 4 coins : HG -> HD -> BD -> BG  |  r=reset  entree=valider  s=skip  q=quit"

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            # reconvertit en coordonnées de l'image originale
            self.points.append((x / self.scale, y / self.scale))

    def _redraw(self):
        canvas = self.display.copy()
        labels = ["HG", "HD", "BD", "BG"]
        for i, (px, py) in enumerate(self.points):
            dx, dy = int(px * self.scale), int(py * self.scale)
            cv2.circle(canvas, (dx, dy), 6, (0, 0, 255), -1)
            cv2.putText(canvas, labels[i], (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(self.points) > 1:
            pts = np.array([(int(px * self.scale), int(py * self.scale))
                             for px, py in self.points], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=(len(self.points) == 4),
                          color=(0, 255, 0), thickness=2)
        cv2.imshow(self.window, canvas)

    def pick(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)
        while True:
            self._redraw()
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.points = []
            elif key == ord('s'):
                cv2.destroyWindow(self.window)
                return "skip"
            elif key == ord('q'):
                cv2.destroyWindow(self.window)
                return "quit"
            elif key in (13, 10):  # entrée
                if len(self.points) == 4:
                    cv2.destroyWindow(self.window)
                    return self.points
        # jamais atteint


# --------------------------------------------------------------------------- #
# Redressement + découpe
# --------------------------------------------------------------------------- #
def warp_grid(img, corners, rows, cols, cell_px=200):
    """Redresse la zone définie par 4 coins (HG,HD,BD,BG) vers un rectangle
    de taille (cols*cell_px, rows*cell_px) parfaitement rectangulaire."""
    src = np.array(corners, dtype=np.float32)
    out_w, out_h = cols * cell_px, rows * cell_px
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped


def slice_cells(warped, rows, cols, cell_px=200, margin_ratio=0.08):
    """Découpe la grille redressée en cellules, en rognant une petite marge
    intérieure pour éviter d'inclure les traits de la grille."""
    cells = []
    margin = int(cell_px * margin_ratio)
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y0, y1 = r * cell_px + margin, (r + 1) * cell_px - margin
            x0, x1 = c * cell_px + margin, (c + 1) * cell_px - margin
            row_cells.append(warped[y0:y1, x0:x1])
        cells.append(row_cells)
    return cells


# --------------------------------------------------------------------------- #
# Conversion bitmap pour le format CCC
# --------------------------------------------------------------------------- #
def to_ccc_bitmap(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Le format CCC stocke les pixels de l'encre sous forme de 1.
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    return (binary > 0).astype(np.uint8)


def write_ccc_record(handle, bitmap, label):
    """Écrit un enregistrement au format CCC dans un fichier .chr."""
    h, w = bitmap.shape
    handle.write(f"{w} {h} 0 0 0\n")
    for row in bitmap:
        handle.write("".join("1" if px else "0" for px in row) + "\n")
    handle.write(f"{label}\n")


def is_test_cell(index, total_cells, test_ratio):
    """Répartit régulièrement les cellules d'une image dans le set de test."""
    test_count = round(total_cells * test_ratio)
    before = round(index * test_count / total_cells)
    after = round((index + 1) * test_count / total_cells)
    return after > before


# --------------------------------------------------------------------------- #
# Cache des coins cliqués
# --------------------------------------------------------------------------- #
def cache_path(img_path):
    return os.path.splitext(img_path)[0] + ".corners.json"


def load_cached_corners(img_path):
    p = cache_path(img_path)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None


def list_subdirectories(path):
    try:
        return sorted([
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ])
    except FileNotFoundError:
        return []


def choose_from_list(prompt_text, items):
    print(prompt_text)
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item}")

    while True:
        choice = input("Choix (numéro ou nom) : ").strip()
        if not choice:
            print("Veuillez entrer un numéro ou un nom valide.")
            continue
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(items):
                return items[index]
            print("Numéro invalide. Réessaie.")
            continue
        if choice in items:
            return choice
        print("Valeur invalide. Réessaie.")


def save_cached_corners(img_path, corners):
    with open(cache_path(img_path), "w") as f:
        json.dump(corners, f)


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Redressement + découpe de grilles manuscrites vers un format compatible CCC.")
    parser.add_argument("--input", default="custom_data", help="Dossier racine contenant les sous-dossiers [lettre] (chemin relatif au dossier d'exécution, ou absolu)")
    parser.add_argument("--output", default="custom_data_processed", help="Dossier racine de sortie")
    parser.add_argument("--rows", type=int, default=ROWS_DEFAULT)
    parser.add_argument("--cols", type=int, default=COLS_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Recliquer même si un cache de coins existe déjà")
    parser.add_argument("--ext", default="jpg", help="Extension des photos (jpg, png, ...)")
    parser.add_argument("--test-ratio", type=float, default=1 / 3, help="Proportion de cellules de chaque image à placer dans test.chr (défaut: 1/3)")
    parser.add_argument("--chr-output", default=None, help="Chemin du fichier training.chr. Si absent, il sera créé dans le dossier de sortie.")
    parser.add_argument("--test-chr-output", default=None, help="Chemin du fichier test.chr. Si absent, il sera créé dans le dossier de sortie.")
    parser.add_argument("--display-max-dim", type=int, default=DISPLAY_MAX_DIM, help="Dimension maximale de l'aperçu des images (0 = résolution native, défaut: 1800 px)")
    args = parser.parse_args()

    if not 0 <= args.test_ratio <= 1:
        parser.error("--test-ratio doit être compris entre 0 et 1.")

    persons = list_subdirectories(args.input)
    if not persons:
        print(f"Aucun dossier de personne trouvé dans {args.input}.")
        sys.exit(1)

    chosen_person = choose_from_list("Sélectionne une personne :", persons)
    person_input_dir = os.path.join(args.input, chosen_person)

    letters = list_subdirectories(person_input_dir)
    if not letters:
        print(f"Aucun dossier de lettre trouvé dans {person_input_dir}.")
        sys.exit(1)

    chosen_letter = choose_from_list(f"Sélectionne une lettre pour {chosen_person} :", letters)
    letter_input_dir = os.path.join(person_input_dir, chosen_letter)

    search_pattern = os.path.join(letter_input_dir, f"*.{args.ext}")
    image_paths = sorted(glob.glob(search_pattern))
    if not image_paths:
        print(f"Aucune image trouvée avec le motif : {search_pattern}")
        print(f"Chemin absolu recherché : {os.path.abspath(search_pattern)}")
        print("Vérifie que --input pointe bien vers le dossier contenant les sous-dossiers [person]/[letter],")
        print("et que --ext correspond à l'extension réelle de tes photos (jpg, jpeg, png...).")
        sys.exit(1)

    print(f"{len(image_paths)} image(s) trouvée(s) pour {chosen_person}/{chosen_letter}.")

    out_base_dir = os.path.join(args.output, chosen_person, chosen_letter)
    os.makedirs(out_base_dir, exist_ok=True)

    training_chr_path = args.chr_output or os.path.join(out_base_dir, "training.chr")
    test_chr_path = args.test_chr_output or os.path.join(out_base_dir, "test.chr")
    os.makedirs(os.path.dirname(training_chr_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(test_chr_path) or ".", exist_ok=True)

    with open(training_chr_path, "w", encoding="latin1") as training_handle, open(
        test_chr_path, "w", encoding="latin1"
    ) as test_handle:
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = out_base_dir

            img = cv2.imread(img_path)
            if img is None:
                print(f"  [!] Impossible de lire {img_path}, ignoré.")
                continue

            corners = None if args.force else load_cached_corners(img_path)

            if corners is None:
                print(f"-> {img_path}")
                picker = CornerPicker(img, args.display_max_dim)
                result = picker.pick()
                if result == "quit":
                    print("Arrêt demandé.")
                    sys.exit(0)
                if result == "skip":
                    print("   sauté.")
                    continue
                corners = result
                save_cached_corners(img_path, corners)
            else:
                print(f"-> {img_path} (coins en cache, utilisés directement)")

            warped = warp_grid(img, corners, args.rows, args.cols)
            cells = slice_cells(warped, args.rows, args.cols)

            count = 0
            test_count = 0
            total_cells = args.rows * args.cols
            for r in range(args.rows):
                for c in range(args.cols):
                    cell_index = r * args.cols + c
                    ccc_bitmap = to_ccc_bitmap(cells[r][c])
                    out_name = f"{base_name}_r{r:02d}_c{c:02d}.png"
                    cv2.imwrite(os.path.join(out_dir, out_name), cells[r][c])
                    if is_test_cell(cell_index, total_cells, args.test_ratio):
                        write_ccc_record(test_handle, ccc_bitmap, chosen_letter.upper())
                        test_count += 1
                    else:
                        write_ccc_record(training_handle, ccc_bitmap, chosen_letter.upper())
                        count += 1
            print(f"   {count} cases training et {test_count} cases test sauvegardées dans {out_dir}")

    cv2.destroyAllWindows()
    print(f"Export CCC terminé : {training_chr_path} et {test_chr_path}")


if __name__ == "__main__":
    main()