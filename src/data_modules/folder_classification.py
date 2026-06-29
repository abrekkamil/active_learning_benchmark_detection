"""
Generic image-folder classification datasets for StructDamage, SDNET2018,
and other folder-level civil damage datasets.

Supported layouts:

1) Explicit split folders:
   root/train/class_a/*.jpg
   root/val/class_a/*.jpg
   root/test/class_a/*.jpg

2) No split folders:
   root/class_a/*.jpg
   root/class_b/*.jpg
   A deterministic stratified train/val/pool split is created internally.

3) SDNET2018 binary layout:
   root/D/CD/*.jpg, root/D/UD/*.jpg, root/P/CP/*.jpg, ...
   Labels are mapped to: uncracked=0, cracked=1.
"""

from pathlib import Path
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FolderClassificationDataset(Dataset):
    def __init__(self, args, data_dir, split="train", label_mode="folder"):
        self.args = args
        self.root = Path(data_dir)
        self.split = "val" if split == "valid" else split
        self.label_mode = label_mode
        self.seed = int(getattr(args, "seed", 42))
        self.val_fraction = float(getattr(args, "val_fraction", 0.15))
        self.pool_fraction = float(getattr(args, "pool_fraction", 0.0))

        size = int(getattr(args, "scale_size", getattr(args, "img_size", 224)))

        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = train_tf if self.split == "train" else eval_tf

        samples, class_names = self._build_samples()
        if len(samples) == 0:
            raise RuntimeError(f"No images found for split='{self.split}' under {self.root}")

        self.classes = class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = samples
        self.imgPaths = [str(p.relative_to(self.root)) for p, _ in self.samples]
        self.targets = [int(y) for _, y in self.samples]
        self.labels = self.targets
        self.num_classes = len(self.classes)

    def _all_images(self, root: Path) -> List[Path]:
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

    def _find_explicit_split_root(self):
        aliases = {
            "train": ["train", "Train", "training", "Training"],
            "val": ["val", "Val", "valid", "Valid", "validation", "Validation", "test", "Test"],
            "pool": ["pool", "Pool", "unlabelled", "unlabeled", "Unlabeled"],
            "test": ["test", "Test"],
        }
        for name in aliases.get(self.split, [self.split]):
            candidate = self.root / name
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _sdnet_label_from_path(self, path):
        """
        Infer binary SDNET2018 label from the image path.

        Supports both common SDNET2018 layouts:

        1) Decks/Cracked/image.jpg
        Decks/Non-cracked/image.jpg

        2) D/CD/image.jpg
        D/UD/image.jpg
        P/CP/image.jpg
        P/UP/image.jpg
        W/CW/image.jpg
        W/UW/image.jpg

        Returns:
            1 = cracked
            0 = non-cracked
        """
        parts = [p.lower() for p in path.parts]

        cracked_tokens = {
            "cracked",
            "crack",
            "cd",
            "cp",
            "cw",
        }

        non_cracked_tokens = {
            "non-cracked",
            "non_cracked",
            "noncracked",
            "uncracked",
            "no-crack",
            "no_crack",
            "nocrack",
            "ud",
            "up",
            "uw",
        }

        for part in parts:
            if part in non_cracked_tokens:
                return 0
            if part in cracked_tokens:
                return 1

        raise ValueError(f"Could not infer SDNET cracked/uncracked label from path: {path}")

    def _folder_label_from_path(self, base: Path, path: Path) -> str:
        rel_parts = path.relative_to(base).parts
        if len(rel_parts) < 2:
            raise ValueError(f"Expected at least class/image under {base}, got {path}")
        return rel_parts[0]

    def _build_samples_from_base(self, base: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
        images = self._all_images(base)
        if self.label_mode == "sdnet_binary":
            classes = ["uncracked", "cracked"]
            samples = [(p, self._sdnet_label_from_path(p)) for p in images]
            return samples, classes

        class_names = sorted({self._folder_label_from_path(base, p) for p in images})
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        samples = [(p, class_to_idx[self._folder_label_from_path(base, p)]) for p in images]
        return samples, class_names

    def _split_without_explicit_folders(self, samples: List[Tuple[Path, int]]):
        rng = random.Random(self.seed)
        by_class = {}
        for p, y in samples:
            by_class.setdefault(y, []).append((p, y))

        selected = []
        for _, items in by_class.items():
            items = list(items)
            rng.shuffle(items)
            n = len(items)
            n_val = int(round(self.val_fraction * n))
            n_pool = int(round(self.pool_fraction * n))
            n_train = max(0, n - n_val - n_pool)

            if self.split == "train":
                part = items[:n_train]
            elif self.split == "val":
                part = items[n_train:n_train + n_val]
            elif self.split == "pool":
                part = items[n_train + n_val:]
            elif self.split == "test":
                part = items[n_train:n_train + n_val]
            else:
                raise ValueError(f"Unsupported split: {self.split}")
            selected.extend(part)

        return sorted(selected, key=lambda x: str(x[0]))

    def _build_samples(self):
        explicit_root = self._find_explicit_split_root()
        if explicit_root is not None:
            return self._build_samples_from_base(explicit_root)

        all_samples, class_names = self._build_samples_from_base(self.root)
        samples = self._split_without_explicit_folders(all_samples)
        return samples, class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "labels": torch.tensor(label, dtype=torch.long),
            "imageIDs": str(path.relative_to(self.root)),
            "image_path": str(path),
        }
