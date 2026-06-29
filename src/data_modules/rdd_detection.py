from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RDDDetectionDataset(Dataset):
    """
    RDD2022 detection dataset.

    Supports:
        1. Pascal VOC XML annotations
        2. YOLO TXT annotations

    Expected output for torchvision Faster R-CNN:
        image: Tensor [3, H, W]
        target:
            boxes: Tensor [N, 4] in xyxy format
            labels: Tensor [N]
            image_id: Tensor
            area: Tensor[N]
            iscrowd: Tensor[N]

    Class ids:
        0 = background
        1 = D00
        2 = D10
        3 = D20
        4 = D40

    For YOLO txt:
        original class ids are assumed to be:
            0 = D00
            1 = D10
            2 = D20
            3 = D40

        They are converted to Faster R-CNN labels:
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

        self.label_aliases = {
            "D01": "D00",
            "D11": "D10",
        }

        self.transform = transforms.ToTensor()

        self.records = self._build_records()

        if len(self.records) == 0:
            raise RuntimeError(
                f"No usable detection records found for split='{self.split}' under {self.root}"
            )

        self.ids = [r["image_id"] for r in self.records]
        self.images = [str(r["image_path"]) for r in self.records]
        self.imgPaths = self.images
        self.num_classes = len(self.classes)

        self.coco = self._build_coco_gt()

        print(
            f"[RDDDetectionDataset] split={self.split} | "
            f"records={len(self.records)} | "
            f"classes={self.classes}"
        )

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

    # ==========================================================
    # Build records
    # ==========================================================
    def _build_records(self) -> List[Dict]:
        split_dir = self._find_split_dir()

        if split_dir is not None:
            xml_files = sorted(split_dir.rglob("*.xml"))
            txt_files = sorted(split_dir.rglob("*.txt"))

            # Exclude YAML/class-list txt files if present.
            txt_files = [
                p for p in txt_files
                if "label" in {part.lower() for part in p.parts}
                or p.parent.name.lower() in {"labels", "label"}
            ]

            if xml_files:
                records = self._build_from_voc_xml(xml_files)
            elif txt_files:
                records = self._build_from_yolo_split(split_dir)
            else:
                # Some YOLO datasets can have images with missing/empty labels.
                # Try image-only records.
                records = self._build_from_images_with_optional_yolo(split_dir)

        else:
            # Fallback for non-split VOC folders.
            xml_files = sorted(self.root.rglob("*.xml"))

            if xml_files:
                records_all = self._build_from_voc_xml(xml_files)
                records = self._split_records_if_needed(records_all)
            else:
                records = self._build_from_yolo_split(self.root)

        records = sorted(records, key=lambda r: str(r["image_path"]))

        for i, r in enumerate(records, start=1):
            r["image_id"] = i

        return records

    def _find_split_dir(self) -> Optional[Path]:
        candidates = [
            self.root / self.split,
            self.root / ("valid" if self.split == "val" else self.split),
            self.root / ("validation" if self.split == "val" else self.split),
            self.root / "RDD_SPLIT" / self.split,
            self.root / "RDD_SPLIT" / ("valid" if self.split == "val" else self.split),
            self.root / "RDD_SPLIT" / ("validation" if self.split == "val" else self.split),
        ]

        for c in candidates:
            if c.exists() and c.is_dir():
                return c

        return None

    # ==========================================================
    # YOLO support
    # ==========================================================
    def _build_from_yolo_split(self, split_dir: Path) -> List[Dict]:
        image_files = self._find_images(split_dir)

        if len(image_files) == 0:
            raise FileNotFoundError(f"No images found under YOLO split dir: {split_dir}")

        records = []

        for image_path in image_files:
            label_path = self._find_yolo_label_for_image(split_dir, image_path)

            with Image.open(image_path) as im:
                width, height = im.size

            boxes, labels = self._parse_yolo_txt(
                label_path=label_path,
                width=width,
                height=height,
            )

            records.append(
                {
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        return records

    def _build_from_images_with_optional_yolo(self, split_dir: Path) -> List[Dict]:
        return self._build_from_yolo_split(split_dir)

    def _find_images(self, split_dir: Path) -> List[Path]:
        preferred_dirs = [
            split_dir / "images",
            split_dir / "Images",
            split_dir / "JPEGImages",
            split_dir,
        ]

        image_files = []

        for d in preferred_dirs:
            if d.exists():
                for ext in IMG_EXTS:
                    image_files.extend(d.rglob(f"*{ext}"))
                    image_files.extend(d.rglob(f"*{ext.upper()}"))

        # remove duplicates while keeping order
        seen = set()
        unique = []

        for p in image_files:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                unique.append(p)

        # Do not include label/annotation folders accidentally.
        unique = [
            p for p in unique
            if "label" not in {part.lower() for part in p.parts}
            and "annotation" not in {part.lower() for part in p.parts}
        ]

        return sorted(unique)

    def _find_yolo_label_for_image(self, split_dir: Path, image_path: Path) -> Optional[Path]:
        stem = image_path.stem

        candidates = [
            split_dir / "labels" / f"{stem}.txt",
            split_dir / "label" / f"{stem}.txt",
            split_dir / "Labels" / f"{stem}.txt",
            split_dir / "annotations" / f"{stem}.txt",
            split_dir / "Annotations" / f"{stem}.txt",
            image_path.with_suffix(".txt"),
        ]

        # Common YOLO structure:
        # train/images/img.jpg -> train/labels/img.txt
        parts = list(image_path.parts)

        if "images" in parts:
            idx = parts.index("images")
            new_parts = parts[:idx] + ["labels"] + parts[idx + 1:]
            candidates.append(Path(*new_parts).with_suffix(".txt"))

        if "Images" in parts:
            idx = parts.index("Images")
            new_parts = parts[:idx] + ["Labels"] + parts[idx + 1:]
            candidates.append(Path(*new_parts).with_suffix(".txt"))

        for c in candidates:
            if c.exists():
                return c

        # No label file means no objects.
        return None

    def _parse_yolo_txt(
        self,
        label_path: Optional[Path],
        width: int,
        height: int,
    ) -> Tuple[List[List[float]], List[int]]:
        boxes: List[List[float]] = []
        labels: List[int] = []

        if label_path is None or not label_path.exists():
            return boxes, labels

        text = label_path.read_text().strip()

        if not text:
            return boxes, labels

        for line in text.splitlines():
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
            except ValueError:
                continue

            # YOLO normalized coordinates.
            x_center *= width
            y_center *= height
            box_w *= width
            box_h *= height

            xmin = x_center - box_w / 2.0
            ymin = y_center - box_h / 2.0
            xmax = x_center + box_w / 2.0
            ymax = y_center + box_h / 2.0

            xmin = max(0.0, min(xmin, width - 1))
            ymin = max(0.0, min(ymin, height - 1))
            xmax = max(0.0, min(xmax, width - 1))
            ymax = max(0.0, min(ymax, height - 1))

            if xmax <= xmin or ymax <= ymin:
                continue

            # YOLO labels are 0-based. Faster R-CNN labels need background=0,
            # so foreground starts from 1.
            frcnn_label = class_id + 1

            if frcnn_label < 1 or frcnn_label >= len(self.classes):
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(frcnn_label)

        return boxes, labels

    # ==========================================================
    # Pascal VOC support
    # ==========================================================
    def _build_from_voc_xml(self, xml_files: List[Path]) -> List[Dict]:
        parsed = []

        for xml_path in xml_files:
            try:
                rec = self._parse_voc_xml(xml_path)
            except Exception as e:
                print(f"[RDDDetectionDataset] skipping bad XML {xml_path}: {e}")
                continue

            if rec is not None:
                parsed.append(rec)

        return parsed

    def _split_records_if_needed(self, records_all: List[Dict]) -> List[Dict]:
        explicit_val = [
            r for r in records_all
            if self._path_has_part(r["xml_path"], {"val", "valid", "validation"})
        ]

        explicit_train = [
            r for r in records_all
            if self._path_has_part(r["xml_path"], {"train", "training"})
        ]

        if explicit_val and explicit_train:
            return explicit_val if self.split == "val" else explicit_train

        rng = random.Random(self.seed)
        items = list(records_all)
        rng.shuffle(items)

        n_val = max(1, int(round(self.val_fraction * len(items))))

        val_items = items[:n_val]
        train_items = items[n_val:]

        return val_items if self.split == "val" else train_items

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

        for parent in [xml_path.parent, xml_path.parent.parent, xml_path.parent.parent.parent, self.root]:
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

    # ==========================================================
    # COCO ground truth for evaluation
    # ==========================================================
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