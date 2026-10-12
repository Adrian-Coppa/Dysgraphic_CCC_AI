"""
CNN pour la reconnaissance de lettres manuscrites cursives.
Architecture : ConvNet à 3 blocs convolutifs + classificateur FC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LetterCNN(nn.Module):
    """
    Réseau convolutif pour les lettres cursives (28x28 pixels, 26 classes A-Z).
    
    Architecture :
        Bloc 1 : Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout
        Bloc 2 : Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout
        Bloc 3 : Conv(64→128) → BN → ReLU → AdaptiveAvgPool
        FC     : 128 → 256 → 26
    """

    def __init__(self, num_classes: int = 26, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # --- Blocs convolutifs ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28×28 → 14×14
            nn.Dropout2d(dropout * 0.5),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14×14 → 7×7
            nn.Dropout2d(dropout * 0.5),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), # → 4×4
        )

        # --- Classificateur ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités softmax."""
        return F.softmax(self(x), dim=1)


def build_model(num_classes: int = 26, dropout: float = 0.4) -> LetterCNN:
    return LetterCNN(num_classes=num_classes, dropout=dropout)
