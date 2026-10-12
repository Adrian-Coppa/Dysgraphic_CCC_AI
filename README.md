# Reconnaissance d'écriture manuscrite — EMNIST Letters

Projet en deux phases :
- **Phase 1** : entraîner un CNN sur EMNIST Letters (26 classes A-Z)
- **Phase 2** : comparer les performances sur une écriture dysgraphique personnelle

---

## Structure du projet

```
emnist_project/
├── train.py                  # ← Script principal d'entraînement
├── compare_datasets.py       # ← Phase 2 : comparaison des datasets
├── requirements.txt
├── models/
│   └── cnn_model.py          # Architecture CNN
├── utils/
│   ├── data_loader.py        # Chargement EMNIST + dataset personnalisé
│   ├── trainer.py            # Boucle d'entraînement + early stopping
│   ├── evaluator.py          # Métriques, matrice de confusion, graphiques
│   └── predictor.py          # Inférence sur une image unique
├── data/                     # Téléchargé automatiquement par torchvision
├── models/                   # Checkpoints sauvegardés ici
└── results/                  # Graphiques et rapports
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Choisir le dataset et entraîner

```bash
# Choix interactif : CCC ou dataset personnalisé
python train.py

# Choix explicite
python train.py --dataset ccc
python train.py --dataset custom --custom-dir ./custom_data_processed/adrian

# Paramètres personnalisés
python train.py --dataset ccc --epochs 50 --lr 3e-3 --batch-size 256

# Test rapide (5 epochs)
python train.py --fast

# Évaluation seule (si best_model.pt existe déjà)
python train.py --dataset ccc --eval-only

# Entraîner puis évaluer le modèle sur CCC et le dataset personnalisé
python train.py --dataset custom --custom-dir ./custom_data_processed/adrian --compare
```

Le dataset personnalisé doit contenir un dossier par lettre, avec `training.chr`
et `test.chr` dans chaque dossier. La racine peut être `custom_data_processed`
ou directement un dossier de personne comme `custom_data_processed/adrian`.
Le modèle CCC reste dans `models/best_model.pt` pour préserver la compatibilité ;
les modèles personnalisés sont sauvegardés dans `models/custom/best_model.pt`.

**Fichiers produits :**
- `models/best_model.pt` — meilleur checkpoint
- `results/training_history.png` — courbes loss/accuracy
- `results/EMNIST_confusion_matrix.png` — matrice de confusion (26×26)
- `results/EMNIST_per_class.png` — accuracy par lettre
- `results/EMNIST_report.txt` — précision / rappel / F1 par classe

---

## Phase 2 — Comparaison avec écriture dysgraphique

### Préparer les données

Créer un dossier `custom_data/` avec une sous-arborescence par lettre :

```
custom_data/
    A/
        a_photo_1.png
        a_photo_2.png
    B/
        b_photo_1.png
    …
```

> **Astuce pour scanner votre écriture :**
> - Écrire chaque lettre sur du papier blanc
> - Scanner ou photographier avec fond blanc
> - Recadrer chaque lettre en une image (environ 100×100 px suffit)
> - Nommage libre, l'extension `.png` ou `.jpg` est acceptée

### Lancer la comparaison

```bash
# Comparaison simple
python compare_datasets.py --custom-dir ./custom_data

# Comparaison avec les fichiers .chr personnalisés
python compare_datasets.py --checkpoint ./models/custom/best_model.pt \
    --custom-dir ./custom_data_processed/adrian --custom-format chr

# Avec fine-tuning sur les données dysgraphiques
python compare_datasets.py --custom-dir ./custom_data --retrain --retrain-epochs 15
```

---

## Prédire une seule lettre

```bash
python utils/predictor.py ma_lettre.png --checkpoint models/best_model.pt
```

Ou en Python :

```python
from utils.predictor import LetterPredictor

predictor = LetterPredictor("models/best_model.pt")
lettre, confiance, top5 = predictor.predict_image("ma_lettre.png")
print(f"Lettre : {lettre}  ({confiance:.1%})")
```

---

## Architecture du CNN

```
Input (1×28×28)
  → Block 1 : Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout
  → Block 2 : Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
  → Block 3 : Conv(128) → BN → ReLU → AdaptiveAvgPool(4×4)
  → FC : Flatten → Linear(256) → ReLU → Dropout → Linear(26)
```

Paramètres : ~600 000  
Performances attendues sur EMNIST Letters : **~91-93% accuracy**

---

## Notes sur EMNIST Letters

- Dataset : 88 800 images train / 14 800 test
- 26 classes (A-Z, insensible à la casse)
- Images 28×28 niveaux de gris, fond noir, trait blanc
- Téléchargement automatique via `torchvision` (~550 Mo)

---

## Piste d'extension : Transfer Learning

Pour améliorer la reconnaissance dysgraphique, deux stratégies :

1. **Fine-tuning** (`--retrain`) : ajuster les poids sur vos données après entraînement EMNIST.
2. **Data augmentation renforcée** : ajouter bruit, déformation élastique pour simuler la dysgraphie lors de l'entraînement initial.
