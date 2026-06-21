from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency until installed
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if nn is not None:
    class _NumericTabTransformer(nn.Module):
        def __init__(
            self,
            n_features: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.value_projection = nn.Linear(1, d_model)
            self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x: Any) -> Any:
            tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self.head(pooled).squeeze(-1)
else:
    _NumericTabTransformer = None


@dataclass
class TabTransformerClassifier:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    val_fraction: float = 0.15
    random_state: int = 42
    device: str = "auto"

    def fit(self, x: Any, y: Any) -> "TabTransformerClassifier":
        if torch is None:
            raise RuntimeError("torch is not installed. Install requirements.txt to run TABTX experiments.")
        torch.manual_seed(self.random_state)
        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        self.mean_ = x_np.mean(axis=0, keepdims=True)
        self.std_ = x_np.std(axis=0, keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        x_np = (x_np - self.mean_) / self.std_

        device = self._device()
        self.model_ = _NumericTabTransformer(
            n_features=x_np.shape[1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)

        stratify = y_np if len(np.unique(y_np)) == 2 and np.bincount(y_np.astype(int)).min() >= 2 else None
        use_validation = False
        if 0 < self.val_fraction < 0.5 and len(y_np) >= 20:
            x_train, x_val, y_train, y_val = train_test_split(
                x_np,
                y_np,
                test_size=self.val_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
            use_validation = bool(y_train.sum() >= 1 and (1.0 - y_train).sum() >= 1)
            if not use_validation:
                x_train, y_train = x_np, y_np
                x_val, y_val = None, None
        else:
            x_train, y_train = x_np, y_np
            x_val, y_val = None, None

        pos = max(float(y_train.sum()), 1.0)
        neg = max(float((1.0 - y_train).sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        val_x = torch.from_numpy(x_val).to(device) if use_validation else None
        val_y = torch.from_numpy(y_val).to(device) if use_validation else None
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        best_loss = float("inf")
        best_state = None
        stale = 0
        self.history_ = []
        self.model_.train()
        for epoch in range(1, self.max_epochs + 1):
            train_loss_sum = 0.0
            train_examples = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                batch_size = int(yb.shape[0])
                train_loss_sum += float(loss.detach().cpu()) * batch_size
                train_examples += batch_size
            train_loss = train_loss_sum / max(train_examples, 1)
            val_loss = float("nan")
            if not use_validation:
                self.history_.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "used_validation": False,
                })
                continue
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(val_x), val_y).detach().cpu())
            self.model_.train()
            self.history_.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "used_validation": True,
            })
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        self.n_epochs_ = len(self.history_)
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch is not installed. Install requirements.txt to run TABTX experiments.")
        x_np = np.asarray(x, dtype=np.float32)
        x_np = (x_np - self.mean_) / self.std_
        device = self._device()
        self.model_.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(x_np), self.batch_size):
                xb = torch.from_numpy(x_np[start:start + self.batch_size]).to(device)
                prob = torch.sigmoid(self.model_(xb)).detach().cpu().numpy()
                probs.append(prob)
        pos = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, x: Any) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
