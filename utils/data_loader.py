"""
Chargement et augmentation du dataset CCC (Cursive Character Challenge).

Le format .chr est décrit dans ccc/README.TXT.
Chaque échantillon contient :
- une bitmap binaire 0/1,
- une étiquette de caractère (lettre, parfois en majuscule/minuscule).
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# ── Correspondance label ↔ caractère ───────────────────────────────────────
LABEL_TO_CHAR = {i: chr(65 + i) for i in range(26)}
CHAR_TO_LABEL = {v: k for k, v in LABEL_TO_CHAR.items()}


def _set_label_mapping(labels: List[str]) -> None:
    global LABEL_TO_CHAR, CHAR_TO_LABEL
    LABEL_TO_CHAR.clear()
    LABEL_TO_CHAR.update({i: char for i, char in enumerate(labels)})
    CHAR_TO_LABEL.clear()
    CHAR_TO_LABEL.update({char: idx for idx, char in LABEL_TO_CHAR.items()})


# ── Transformations ─────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmentation légère pour l'entraînement."""
    return transforms.Compose([
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            shear=4,
        ),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_eval_transform() -> transforms.Compose:
    """Transformation minimale pour validation/test/inférence."""
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_inference_transform() -> transforms.Compose:
    """Prétraitement pour une image externe."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


# ── Lecture du format CCC (.chr) ────────────────────────────────────────────

def _resolve_ccc_dir(data_dir: str) -> Path:
    root = Path(data_dir)
    if (root / "training.chr").exists():
        return root
    if (root / "ccc" / "training.chr").exists():
        return root / "ccc"
    if root.name == "ccc" and any(root.glob("*.chr")):
        return root
    raise FileNotFoundError(
        f"Aucun fichier CCC trouvé dans {data_dir}. "
        "Attendez un dossier contenant training.chr et test.chr."
    )


def _parse_chr_file(path: Path) -> List[Tuple[Image.Image, str]]:
    samples: List[Tuple[Image.Image, str]] = []
    with path.open("r", encoding="latin1") as handle:
        lines = [line.rstrip("\n") for line in handle]

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        parts = line.split()
        if len(parts) == 5 and all(part.lstrip("-").isdigit() for part in parts):
            width, height, _, _, _ = map(int, parts)
            idx += 1
            bitmap_rows: List[str] = []
            label = None
            while idx < len(lines):
                row = lines[idx].rstrip("\n")
                if not row.strip():
                    idx += 1
                    continue
                if len(row.strip()) == 1 and row.strip() not in {"0", "1"}:
                    label = row.strip()
                    idx += 1
                    break
                bitmap_rows.append(row.strip())
                idx += 1

            if len(bitmap_rows) != height or label is None:
                continue
            if width <= 0:
                continue

            pixel_rows = []
            for row in bitmap_rows:
                if len(row) != width:
                    row = row[:width]
                pixel_rows.append([1 if ch == "1" else 0 for ch in row])

            bitmap = np.array(pixel_rows, dtype=np.uint8)
            bitmap = np.where(bitmap > 0, 255, 0).astype(np.uint8)
            img = Image.fromarray(bitmap, mode="L")
            # normaliser l'étiquette en majuscule (CCC contient des maj/min)
            samples.append((img, label.upper()))
            continue

        idx += 1

    return samples


class CCCCharacterDataset(Dataset):
    """Dataset basé sur les fichiers .chr du CCC."""

    def __init__(self, samples: List[Tuple[Image.Image, str]], transform=None):
        self.samples = samples
        self.transform = transform or get_eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, label = self.samples[idx]
        img = image.convert("L")
        if self.transform:
            img = self.transform(img)
        return img, CHAR_TO_LABEL[label]


def load_ccc(
    data_dir: str = "./ccc",
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Charge les données d'entraînement, validation et test depuis le format CCC."""
    root = _resolve_ccc_dir(data_dir)
    train_files = sorted(root.glob("training*.chr"))
    test_files = sorted(root.glob("test*.chr"))

    if not train_files or not test_files:
        raise FileNotFoundError(f"Impossible de trouver les fichiers training*.chr/test*.chr dans {root}")

    train_samples: List[Tuple[Image.Image, str]] = []
    for path in train_files:
        train_samples.extend(_parse_chr_file(path))

    test_samples: List[Tuple[Image.Image, str]] = []
    for path in test_files:
        test_samples.extend(_parse_chr_file(path))

    if not train_samples or not test_samples:
        raise FileNotFoundError(f"Aucun échantillon lisible dans {root}")

    # Forcer une mappage stable A-Z (26 classes) et normaliser les labels
    # Le dataset CCC peut contenir majuscules/minuscules ; on convertit tout en majuscules
    def _force_alpha_samples(samples: List[Tuple[Image.Image, str]]):
        out: List[Tuple[Image.Image, str]] = []
        for img, lab in samples:
            L = lab.upper()
            if len(L) == 1 and "A" <= L <= "Z":
                out.append((img, L))
        return out

    train_samples = _force_alpha_samples(train_samples)
    test_samples = _force_alpha_samples(test_samples)

    # fixed mapping A-Z to ensure consistent class ordering and 26 classes
    LABEL_TO_CHAR.clear()
    LABEL_TO_CHAR.update({i: chr(65 + i) for i in range(26)})
    CHAR_TO_LABEL.clear()
    CHAR_TO_LABEL.update({v: k for k, v in LABEL_TO_CHAR.items()})

    train_ds = CCCCharacterDataset(train_samples, transform=get_train_transform())
    val_ds = CCCCharacterDataset(train_samples, transform=get_eval_transform())
    test_ds = CCCCharacterDataset(test_samples, transform=get_eval_transform())

    n = len(train_samples)
    n_val = int(n * val_split)
    n_val = max(1, min(n_val, n - 1))
    indices = torch.randperm(n).tolist()
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[Data] Train: {len(train_subset):,}  Val: {len(val_subset):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def _find_custom_chr_files(root: Path, split: str) -> List[Tuple[Path, str]]:
    """Trouve les fichiers .chr d'un split et déduit la lettre du dossier parent."""
    files: List[Tuple[Path, str]] = []
    for path in sorted(root.rglob(f"{split}*.chr")):
        letter_dir = path.parent
        label = letter_dir.name.upper()
        if len(label) == 1 and "A" <= label <= "Z":
            files.append((path, label))
    return files


def load_custom_chr(
    data_dir: str = "./custom_data_processed",
    batch_size: int = 128,
    num_workers: int = 0,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Charge un dataset personnalisé organisé en dossiers de lettres et fichiers .chr.

    Le dossier fourni peut être la racine (ex. custom_data_processed) ou un
    sous-dossier de personne (ex. custom_data_processed/adrian).
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dossier personnalisé introuvable : {data_dir}")

    train_files = _find_custom_chr_files(root, "training")
    test_files = _find_custom_chr_files(root, "test")
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Impossible de trouver training*.chr et test*.chr dans {data_dir}"
        )

    def read_files(files: List[Tuple[Path, str]]) -> List[Tuple[Image.Image, str]]:
        samples: List[Tuple[Image.Image, str]] = []
        for path, folder_label in files:
            samples.extend((image, folder_label) for image, _ in _parse_chr_file(path))
        return samples

    train_samples = read_files(train_files)
    test_samples = read_files(test_files)
    if not train_samples or not test_samples:
        raise FileNotFoundError(f"Aucun échantillon lisible dans {data_dir}")

    LABEL_TO_CHAR.clear()
    LABEL_TO_CHAR.update({i: chr(65 + i) for i in range(26)})
    CHAR_TO_LABEL.clear()
    CHAR_TO_LABEL.update({v: k for k, v in LABEL_TO_CHAR.items()})

    train_ds = CCCCharacterDataset(train_samples, transform=get_train_transform())
    val_ds = CCCCharacterDataset(train_samples, transform=get_eval_transform())
    test_ds = CCCCharacterDataset(test_samples, transform=get_eval_transform())

    n = len(train_samples)
    if n < 2:
        raise ValueError("Le dataset personnalisé doit contenir au moins 2 exemples d'entraînement.")
    n_val = max(1, min(int(n * val_split), n - 1))
    indices = torch.randperm(n).tolist()
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(Subset(train_ds, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(val_ds, val_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(
        f"[CustomCHR] Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
        f"Test: {len(test_ds):,}"
    )
    return train_loader, val_loader, test_loader


class CustomHandwritingDataset(Dataset):
    """Dataset pour images manuscrites personnelles."""

    def __init__(self, root_dir: str, transform=None):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform or get_inference_transform()

        root = Path(root_dir)
        for letter_dir in sorted(root.iterdir()):
            if not letter_dir.is_dir():
                continue
            letter = letter_dir.name.upper()
            if letter not in CHAR_TO_LABEL:
                continue
            label = CHAR_TO_LABEL[letter]
            for img_path in letter_dir.glob("*"):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    self.samples.append((str(img_path), label))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {root_dir}. "
                "Assurez-vous d'avoir des sous-dossiers A/, B/, …"
            )
        print(f"[CustomDataset] {len(self.samples)} images dans {len(set(label for _, label in self.samples))} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_custom_dataset(root_dir: str, batch_size: int = 32) -> DataLoader:
    ds = CustomHandwritingDataset(root_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
