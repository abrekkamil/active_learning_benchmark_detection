import time
from typing import Dict, List

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import torch.nn.functional as F
from .base_model import BaseModel
from .utils import _ensure_rgb


class FasterRCNNModel(BaseModel):
    """
    Torchvision Faster R-CNN wrapper for bounding-box object detection.

    Dataset must return:
        image  : Tensor [3, H, W]
        target : dict with boxes, labels, image_id, area, iscrowd

    Label convention:
        0 = background, never used in target labels
        1..K = foreground classes
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        super().__init__(num_classes, device, config)

        self.task_type = "detection"

        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        pretrained = bool(getattr(config, "pretrained", True))

        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            model = fasterrcnn_resnet50_fpn(weights=weights)

        except Exception:
            # Older torchvision fallback.
            model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes,
        )

        self.model = model.to(device)
        self.model.get_bottleneck_features = self.get_bottleneck_features
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=getattr(config, "lr", 0.005),
            momentum=getattr(config, "momentum", 0.9),
            weight_decay=getattr(config, "weight_decay", 0.0005),
        )


    def get_bottleneck_features(self, images):
        """
        Return one feature vector per image for RL state construction.

        This makes Faster R-CNN behave like the classification ResNet model,
        which already exposes get_bottleneck_features().

        Input:
            images can be:
                - Tensor [B, C, H, W]
                - Tensor [C, H, W]
                - list of Tensor [C, H, W]

        Output:
            Tensor [B, D]
        """

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():

            if torch.is_tensor(images):
                if images.dim() == 3:
                    images = [images.to(self.device)]
                elif images.dim() == 4:
                    images = [img.to(self.device) for img in images]
                else:
                    raise ValueError(
                        f"Expected image tensor with 3 or 4 dimensions, got {images.dim()}"
                    )

            elif isinstance(images, (list, tuple)):
                images = [img.to(self.device) for img in images]

            else:
                raise TypeError(
                    "images must be a tensor or a list/tuple of image tensors"
                )

            # Faster R-CNN expects a list of images and applies its own transform.
            image_list, _ = self.model.transform(images, None)

            features = self.model.backbone(image_list.tensors)

            # Faster R-CNN FPN returns an OrderedDict of feature maps.
            if isinstance(features, dict):
                pooled_levels = []

                for fmap in features.values():
                    # fmap: [B, C, H, W] -> [B, C]
                    pooled = F.adaptive_avg_pool2d(
                        fmap,
                        output_size=(1, 1),
                    ).flatten(1)

                    pooled_levels.append(pooled)

                # Most FPN levels have same channel size, usually 256.
                # Average across pyramid levels to keep feature dim compact.
                feats = torch.stack(pooled_levels, dim=0).mean(dim=0)

            else:
                feats = F.adaptive_avg_pool2d(
                    features,
                    output_size=(1, 1),
                ).flatten(1)

        if was_training:
            self.model.train()

        return feats
    def train_epoch(self, dataset, epoch: int, total_epochs: int = 1) -> Dict[str, float]:
        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 2),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
        )

        self.model.train()

        start = time.time()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=False,
        )

        for images, targets_raw in pbar:
            images = [
                _ensure_rgb(img).to(self.device)
                for img in images
            ]

            targets = [
                {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in t.items()
                }
                for t in targets_raw
            ]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad(set_to_none=True)
            losses.backward()
            self.optimizer.step()

            total_loss += float(losses.item())
            n_batches += 1

            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        return {
            "train_loss": float(total_loss / max(n_batches, 1)),
            "training_time": float(time.time() - start),
        }

    def evaluate(self, dataset) -> Dict[str, float]:
        if not hasattr(dataset, "coco"):
            raise ValueError(
                "FasterRCNNModel.evaluate requires the validation dataset "
                "to expose a `coco` ground-truth object."
            )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
        )

        self.model.eval()

        results = []
        score_thresh = float(getattr(self.config, "eval_score_threshold", 0.001))

        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Validation", leave=False):
                images = [
                    _ensure_rgb(img).to(self.device)
                    for img in images
                ]

                outputs = self.model(images)

                image_id = int(targets[0]["image_id"].item())
                output = outputs[0]

                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if float(score) < score_thresh:
                        continue

                    x1, y1, x2, y2 = box

                    w = max(0.0, float(x2 - x1))
                    h = max(0.0, float(y2 - y1))

                    if w <= 0 or h <= 0:
                        continue

                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), w, h],
                            "score": float(score),
                        }
                    )

        if len(results) == 0:
            coco_gt = dataset.coco
            cat_ids = coco_gt.getCatIds()
            per_class = {
                coco_gt.cats[cat_id].get("name", str(cat_id)): {
                    "AP50": 0.0,
                    "AP50_95": 0.0,
                }
                for cat_id in cat_ids
            }

            return {
                "bbox_AP": 0.0,
                "bbox_AP50": 0.0,
                "bbox_AP75": 0.0,
                "mask_AP": 0.0,
                "per_class": per_class,
            }

        coco_gt = dataset.coco
        coco_dt = coco_gt.loadRes(results)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        cat_ids = coco_eval.params.catIds
        class_names = [
            coco_gt.cats[cat_id].get("name", str(cat_id))
            for cat_id in cat_ids
        ]

        per_class = self.summarize_coco_per_class(coco_eval, class_names)

        return {
            "bbox_AP": float(coco_eval.stats[0]),
            "bbox_AP50": float(coco_eval.stats[1]),
            "bbox_AP75": float(coco_eval.stats[2]),
            "mask_AP": 0.0,
            "per_class": per_class,
        }

    def summarize_coco_per_class(self, coco_eval, class_names):
        """
        Extract per-class AP50 and AP50-95 from COCOeval.

        COCOeval precision shape:
            [iou_thresholds, recall_thresholds, classes, area_ranges, max_dets]
        """

        precisions = coco_eval.eval["precision"]

        # precision: [T, R, K, A, M]
        # T = IoU thresholds
        # R = recall thresholds
        # K = classes
        # A = area range, 0 is usually "all"
        # M = max detections, -1 is usually maxDets[-1]

        per_class = {}

        for class_idx, class_name in enumerate(class_names):
            precision_all = precisions[:, :, class_idx, 0, -1]

            valid = precision_all[precision_all > -1]

            if valid.size == 0:
                ap_5095 = 0.0
            else:
                ap_5095 = float(valid.mean())

            # IoU threshold 0.50 is index 0 in standard COCOeval
            precision_50 = precisions[0, :, class_idx, 0, -1]
            valid_50 = precision_50[precision_50 > -1]

            if valid_50.size == 0:
                ap50 = 0.0
            else:
                ap50 = float(valid_50.mean())

            per_class[class_name] = {
                "AP50": ap50,
                "AP50_95": ap_5095,
            }

        return per_class

    def predict(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        self.model.eval()

        imgs = [
            _ensure_rgb(img).to(self.device)
            for img in images
        ]

        with torch.no_grad():
            outputs = self.model(imgs)

        return [
            {
                k: v.detach().cpu()
                for k, v in out.items()
            }
            for out in outputs
        ]

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Simple detection uncertainty:

        - no detections => high uncertainty
        - otherwise uncertainty = 1 - mean top-k confidence
        """

        preds = self.predict(images)
        scores = []

        for pred in preds:
            if "scores" not in pred or len(pred["scores"]) == 0:
                scores.append(1.0)
                continue

            conf = pred["scores"].float()

            k = min(len(conf), 5)
            topk_mean = torch.topk(conf, k).values.mean().item()

            uncertainty = 1.0 - topk_mean
            uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

            scores.append(uncertainty)

        return np.asarray(scores, dtype=np.float32)

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