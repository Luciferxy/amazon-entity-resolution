"""
PyTorch Lightning Neural Entity Matcher
======================================
Deep Residual Tabular Matcher built with PyTorch Lightning (Lightning AI).
Combines multi-head feature embeddings with residual MLP layers to predict
entity match probabilities and optimize the macro F0.5 metric.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Support both unified lightning and pytorch_lightning
try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        L = None


class ERDataset(Dataset):
    """Dataset for entity resolution candidate pair features."""
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class ResBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and Dropout."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        return self.act(out + residual)


class ERLightningModule(L.LightningModule if L is not None else nn.Module):
    """
    Deep Residual Entity Matcher Lightning Module.
    Optimizes candidate pair classification with weighted binary cross entropy / focal loss.
    """
    def __init__(
        self,
        in_features: int = 28,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight

        # Input projection and normalization
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return logits

    def compute_loss(self, logits, y):
        # Weighted BCE to balance precision vs recall for F0.5
        pos_weight = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Approximate batch precision & recall
        tp = (preds * y).sum()
        fp = (preds * (1 - y)).sum()
        fn = ((1 - preds) * y).sum()
        
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec + 1e-7)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f05", f05, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5
        )
        return [optimizer], [scheduler]

    def predict_probs(self, X: np.ndarray, batch_size: int = 4096, device: str = None) -> np.ndarray:
        """Predict match probabilities for a numpy feature array."""
        self.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        dataset = ERDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))
        
        probs_list = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device)
                logits = self.forward(x)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                probs_list.append(probs)

        return np.concatenate(probs_list)
