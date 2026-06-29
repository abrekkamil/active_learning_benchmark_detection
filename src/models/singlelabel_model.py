"""
Single-label image classification model for folder-level datasets such as
StructDamage and SDNET2018.

Use this for:
    - binary classification with class ids 0/1
    - multi-class classification with one class id per image

Loss: CrossEntropyLoss over raw logits.
Primary metric: macro_f1.
"""

import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from .base_model import BaseModel
from .utils import _ensure_rgb, _batched
from ..data_modules.sample_utils import unpack_batch


class ResNetSingleLabel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
            return feat.view(feat.size(0), -1)


class SingleLabelClassificationModel(BaseModel):
    def __init__(self, num_classes: int, device: torch.device, config):
        super().__init__(num_classes, device, config)
        self.task_type = "classification"

        pretrained = getattr(config, "pretrained", True)
        self.model = ResNetSingleLabel(num_classes, pretrained=pretrained).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-4),
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 32),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        total_loss = 0.0
        start = time.time()
        pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)

        for batch in pbar:
            images, labels = unpack_batch(batch)
            images = images.to(self.device)
            labels = labels.to(self.device).long().view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "train_loss": float(total_loss / max(len(loader), 1)),
            "training_time": float(time.time() - start),
        }

    def evaluate(self, dataset) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 4),
            pin_memory=True,
        )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                images, labels = unpack_batch(batch)
                images = images.to(self.device)
                labels = labels.long().view(-1)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        probs = F.softmax(all_logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

        macro_f1 = float(f1_score(all_labels, preds, average="macro", zero_division=0))
        micro_f1 = float(f1_score(all_labels, preds, average="micro", zero_division=0))
        acc = float(accuracy_score(all_labels, preds))
        bal_acc = float(balanced_accuracy_score(all_labels, preds))
        precision_macro = float(precision_score(all_labels, preds, average="macro", zero_division=0))
        recall_macro = float(recall_score(all_labels, preds, average="macro", zero_division=0))

        if self.num_classes == 2:
            f2 = float(fbeta_score(all_labels, preds, beta=2, average="binary", zero_division=0))
        else:
            f2 = float(fbeta_score(all_labels, preds, beta=2, average="macro", zero_division=0))

        return {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "f1": macro_f1,
            "f2": f2,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
        }

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            logits = self.model(batch)
            return logits.argmax(dim=1).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """Softmax entropy. Higher means more uncertain."""
        self.model.eval()
        scores = []
        with torch.no_grad():
            for img in images:
                img = _ensure_rgb(img).unsqueeze(0).to(self.device)
                logits = self.model(img)
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                scores.append(float(entropy.item()))
        return np.array(scores, dtype=np.float32)

    def get_bottleneck_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            return self.model.get_bottleneck_features(batch).cpu()

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_classes": self.num_classes,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
