# Reconnaissance de lettres manuscrites — CCC

Ce projet entraîne un réseau convolutif sur le **Cursive Character Challenge (CCC)**, puis permet d'évaluer le modèle sur des écritures personnelles, notamment dysgraphiques.

## Structure

```text
train.py                  Entraînement et évaluation sur le CCC ou un dataset personnel
compare_datasets.py       Comparaison du CCC avec une écriture personnelle
process_grid.py           Découpage de photos de grilles d'écriture
check_custom_data.py      Vérification des images d'un dataset personnel
ccc/                      Fichiers CCC (*.chr)
models/cnn_model.py       Architecture du réseau convolutif
utils/data_loader.py      Chargement et prétraitement des données
utils/trainer.py          Entraînement, early stopping et checkpoints
utils/evaluator.py        Métriques et graphiques d'évaluation
utils/predictor.py        Prédiction d'une lettre sur une image
custom_data/              Images personnelles, rangées par lettre
custom_data_processed/    Données personnelles transformées
results/                  Rapports et graphiques générés
```

## Préparer l'environnement

Avec PowerShell sous Windows :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloque l'activation, autorisez-la pour la fenêtre actuelle puis relancez la commande d'activation :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Les données CCC doivent être accessibles dans `ccc/` et contenir les fichiers `.chr` d'entraînement et de test. Un autre dossier peut être indiqué avec `--data-dir`.

## Entraîner et évaluer sur le CCC

```bash
# Entraînement avec les paramètres par défaut
python train.py --dataset ccc

# Paramètres personnalisés
python train.py --dataset ccc --epochs 50 --lr 3e-3 --batch-size 256

# Entraînement court (5 epochs)
python train.py --dataset ccc --fast

# Indiquer un autre dossier CCC
python train.py --dataset ccc --data-dir ./ccc

# Évaluer le checkpoint existant sur le jeu de test CCC
python train.py --dataset ccc --eval-only
```

Sans `--dataset`, le script propose un choix interactif entre le CCC et un dataset personnel. L'option `--compare` demande également une comparaison croisée avec les données personnelles.

Le meilleur modèle est écrit dans `models/best_model.pt`. Les fichiers d'évaluation et l'historique d'entraînement sont enregistrés dans `results/` et `models/history.json`.

## Utiliser un dataset personnel

Les fichiers `.chr` personnels peuvent être rangés dans des sous-dossiers nommés par lettre, par exemple :

```text
custom_data_processed/
└── adrian/
    ├── A/
    │   ├── training.chr
    │   └── test.chr
    └── B/
        ├── training.chr
        └── test.chr
```

L'étiquette du dossier donne la lettre de chaque échantillon. L'entraînement accepte aussi des images rangées dans des sous-dossiers par lettre.

```bash
# Entraîner sur les fichiers .chr personnels
python train.py --dataset custom --custom-dir ./custom_data_processed/adrian

# Entraîner sur des images, rangées dans des dossiers A/, B/, etc.
python train.py --dataset custom --custom-dir ./custom_data

# Entraîner sur les données personnelles puis comparer avec le CCC
python train.py --dataset custom --custom-dir ./custom_data_processed/adrian --compare
```

Les checkpoints personnalisés sont sauvegardés dans `models/custom/best_model.pt`.

## Comparer le CCC et une écriture personnelle

La comparaison évalue le checkpoint sur le jeu de test CCC et sur des données personnelles, puis produit un graphique comparatif.

```bash
# Les images sont rangées dans ./custom_data/A/, ./custom_data/B/, etc.
python compare_datasets.py --custom-dir ./custom_data

# Charger un dossier de fichiers .chr personnels
python compare_datasets.py --custom-dir ./custom_data_processed/adrian --custom-format chr

# Affiner le modèle sur les données personnelles avant de les réévaluer
python compare_datasets.py --custom-dir ./custom_data --retrain --retrain-epochs 15
```

Le dossier CCC est `./ccc` par défaut. Pour en utiliser un autre, passer `--data-dir`.

## Prédire une lettre

```bash
python utils/predictor.py ma_lettre.png --checkpoint models/best_model.pt
```

Ou depuis Python :

```python
from utils.predictor import LetterPredictor

predictor = LetterPredictor("models/best_model.pt")
lettre, confiance, top5 = predictor.predict_image("ma_lettre.png")
print(f"Lettre : {lettre} ({confiance:.1%})")
```

Les images personnelles peuvent être placées sur un fond clair : le prétraitement inverse automatiquement les images à fond clair avant la prédiction.

## Architecture du réseau

```text
Entrée (1 × 28 × 28)
  → Bloc 1 : Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout
  → Bloc 2 : Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
  → Bloc 3 : Conv(128) → BN → ReLU → AdaptiveAvgPool(4 × 4)
  → Classificateur : Flatten → Linear(256) → ReLU → Dropout → Linear(26)
```

Les données CCC sont composées de caractères cursifs étiquetés. Le chargeur convertit les étiquettes en majuscules et conserve les 26 classes A–Z.

## Piste d'amélioration

L'option `--retrain` permet d'affiner le modèle entraîné sur le CCC à l'aide de données personnelles. Une augmentation de données plus poussée peut également aider à simuler les variations d'écriture.
