from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RDDDetectionDataset(Dataset):
    """
    RDD2022 / Pascal VOC object-detection dataset.

    Expected target format for torchvision Faster R-CNN:
        image: Tensor [3, H, W]
        target:
            boxes: Tensor [N, 4] in xyxy format
            labels: Tensor [N]
            image_id: Tensor[1]
            area: Tensor[N]
            iscrowd: Tensor[N]

    Class ids:
        0 = background
        1 = D00
        2 = D10
        3 = D20
        4 = D40
    """

    DEFAULT_CLASSES = ["D00", "D10", "D20", "D40"]

    def __init__(self, args, data_dir, split: str = "train"):
        self.args = args
        self.root = Path(data_dir)
        self.split = "val" if split in ("valid", "validation") else split
        self.seed = int(getattr(args, "seed", 42))
        self.val_fraction = float(getattr(args, "val_fraction", 0.15))

        classes = list(getattr(args, "rdd_classes", self.DEFAULT_CLASSES))

        self.classes = ["background"] + classes
        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}

        # Some old/variant RDD labels may appear.
        self.label_aliases = {
            "D01": "D00",
            "D11": "D10",
        }

        self.transform = transforms.ToTensor()

        self.records = self._build_records()

        if len(self.records) == 0:
            raise RuntimeError(
                f"No RDD/VOC XML records found for split='{self.split}' under {self.root}"
            )

        self.ids = [r["image_id"] for r in self.records]
        self.images = [str(r["image_path"]) for r in self.records]
        self.imgPaths = self.images
        self.num_classes = len(self.classes)

        # Needed for COCO AP evaluation.
        self.coco = self._build_coco_gt()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        img = Image.open(rec["image_path"]).convert("RGB")
        image = self.transform(img)

        boxes = torch.as_tensor(rec["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(rec["labels"], dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (
                (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
                * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(rec["image_id"], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        return image, target

    def _build_records(self) -> List[Dict]:
        xml_files = sorted(self.root.rglob("*.xml"))

        # Do not accidentally use test annotations if a test folder exists.
        xml_files = [
            p for p in xml_files
            if "test" not in {part.lower() for part in p.parts}
        ]

        if not xml_files:
            raise FileNotFoundError(
                f"No Pascal VOC .xml annotations found under {self.root}. "
                "Check if the dataset is fully unzipped. "
                "If this Kaggle version only has YOLO .txt labels, we need a YOLO loader/converter instead."
            )

        parsed = []

        for xml_path in xml_files:
            try:
                rec = self._parse_voc_xml(xml_path)
            except Exception as e:
                print(f"[RDDDetectionDataset] skipping bad XML {xml_path}: {e}")
                continue

            if rec is not None:
                parsed.append(rec)

        explicit_val = [
            r for r in parsed
            if self._path_has_part(r["xml_path"], {"val", "valid", "validation"})
        ]

        explicit_train = [
            r for r in parsed
            if self._path_has_part(r["xml_path"], {"train", "training"})
        ]

        if explicit_val and explicit_train:
            chosen = explicit_val if self.split == "val" else explicit_train
        else:
            rng = random.Random(self.seed)
            items = list(parsed)
            rng.shuffle(items)

            n_val = max(1, int(round(self.val_fraction * len(items))))

            val_items = items[:n_val]
            train_items = items[n_val:]

            chosen = val_items if self.split == "val" else train_items

        chosen = sorted(chosen, key=lambda r: str(r["image_path"]))

        for i, r in enumerate(chosen, start=1):
            r["image_id"] = i

        return chosen

    @staticmethod
    def _path_has_part(path: Path, names: set) -> bool:
        return any(part.lower() in names for part in path.parts)

    def _parse_voc_xml(self, xml_path: Path) -> Dict:
        root = ET.parse(xml_path).getroot()

        filename_node = root.find("filename")
        filename = (
            filename_node.text.strip()
            if filename_node is not None and filename_node.text
            else None
        )

        path_node = root.find("path")
        image_path = None

        if path_node is not None and path_node.text:
            p = Path(path_node.text)
            if p.exists():
                image_path = p

        if image_path is None:
            if filename is None:
                filename = xml_path.with_suffix(".jpg").name

            image_path = self._find_image_for_xml(xml_path, filename)

        size = root.find("size")

        if size is not None:
            width = int(float(size.findtext("width", default="0")))
            height = int(float(size.findtext("height", default="0")))
        else:
            with Image.open(image_path) as im:
                width, height = im.size

        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            name = self.label_aliases.get(name, name)

            if name not in self.class_to_idx:
                continue

            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            xmin = float(bnd.findtext("xmin", default="0"))
            ymin = float(bnd.findtext("ymin", default="0"))
            xmax = float(bnd.findtext("xmax", default="0"))
            ymax = float(bnd.findtext("ymax", default="0"))

            xmin = max(0.0, min(xmin, width - 1))
            ymin = max(0.0, min(ymin, height - 1))
            xmax = max(0.0, min(xmax, width - 1))
            ymax = max(0.0, min(ymax, height - 1))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

        return {
            "xml_path": xml_path,
            "image_path": image_path,
            "width": width,
            "height": height,
            "boxes": boxes,
            "labels": labels,
        }

    def _find_image_for_xml(self, xml_path: Path, filename: str) -> Path:
        candidates = [
            xml_path.parent / filename,
            xml_path.parent.parent / filename,
            xml_path.parent.parent / "images" / filename,
            xml_path.parent.parent.parent / "images" / filename,
            xml_path.parent.parent.parent / "JPEGImages" / filename,
            xml_path.parent.parent.parent / "images" / "train" / filename,
        ]

        for c in candidates:
            if c.exists():
                return c

        stem = Path(filename).stem

        local_parents = [
            xml_path.parent,
            xml_path.parent.parent,
            xml_path.parent.parent.parent,
            self.root,
        ]

        for parent in local_parents:
            if not parent.exists():
                continue

            for ext in IMG_EXTS:
                p = parent / f"{stem}{ext}"
                if p.exists():
                    return p

                p_upper = parent / f"{stem}{ext.upper()}"
                if p_upper.exists():
                    return p_upper

        matches = list(self.root.rglob(filename))
        if matches:
            return matches[0]

        stem_matches = [
            p for p in self.root.rglob(f"{stem}.*")
            if p.suffix.lower() in IMG_EXTS
        ]

        if stem_matches:
            return stem_matches[0]

        raise FileNotFoundError(
            f"Could not find image '{filename}' for XML {xml_path}"
        )

    def _build_coco_gt(self) -> COCO:
        dataset = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": idx, "name": name}
                for name, idx in self.class_to_idx.items()
            ],
        }

        ann_id = 1

        for rec in self.records:
            image_id = rec["image_id"]

            dataset["images"].append(
                {
                    "id": image_id,
                    "file_name": Path(rec["image_path"]).name,
                    "width": rec["width"],
                    "height": rec["height"],
                }
            )

            for box, label in zip(rec["boxes"], rec["labels"]):
                x1, y1, x2, y2 = box
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                dataset["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )

                ann_id += 1

        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()

        return coco