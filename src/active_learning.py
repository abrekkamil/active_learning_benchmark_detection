import torch
import numpy as np
import time
import os
import json
import glob
import re
from torch.utils.data import Subset
from typing import List, Tuple, Dict, Any, Optional
import datetime
from .cold_start_strategies import ColdStartStrategies
from .query_strategies import QueryStrategies
from .utils import setup_logging, save_checkpoint, load_checkpoint
from .data_modules.factory import load_dataset
from .models import build_model

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_train, dataset_pool, samples):
        self.dataset_train = dataset_train
        self.dataset_pool = dataset_pool
        self.samples = samples  # list of ("train"/"pool", idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        source, idx = self.samples[i]
        if source == "train":
            return self.dataset_train[idx]
        elif source == "pool":
            return self.dataset_pool[idx]
        else:
            raise ValueError(f"Unknown source: {source}")

class ActiveLearningSystem:
    """
    Main active learning system combining cold start and query strategies.
    
    This class orchestrates the entire active learning process:
    1. Initialize with a cold start strategy
    2. Train model on initial labeled set
    3. Iteratively query new samples and retrain
    """
    
    def __init__(self, config, skip_cold_start=False):
        """Initialize active learning system."""
        self.system_start_time = datetime.datetime.now()

        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        
        # Setup logging
        self.logger = setup_logging(config.experiment_name)
        self._init_results_path()
        # Load datasets
        self.num_classes = config.num_classes
        self.dataset_train = load_dataset(config, split="train")
        self.dataset_val   = load_dataset(config, split="val")
        self.dataset_pool = None
        if self.config.pool:
            self.dataset_pool  = load_dataset(config, split="pool")

        # Let folder datasets define the number of classes automatically.
        # This is important for StructDamage, because the class count is
        # discovered from the folders.
        if hasattr(self.dataset_train, "num_classes"):
            self.config.num_classes = int(self.dataset_train.num_classes)
            self.num_classes = self.config.num_classes

        # Optional class weights for Sewer-ML multi-label classification.
        if hasattr(self.dataset_train, "class_weights"):
            self.config.class_weights = self.dataset_train.class_weights
        
        # Initialize strategies
        self.cold_start = ColdStartStrategies(self.dataset_train, config)
        self.query_strategy = QueryStrategies(config)
        
        # Initialize model
        self.model = build_model(
            config.model_name,
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )
        # if config.model_name == "maskrcnn":
        #     from .models import MaskRCNNModel
        #     self.model = MaskRCNNModel(config.num_classes, self.device, config)
        # elif config.model_name == "unet":
        #     from .models import UNetModel
        #     self.model = UNetModel(config.num_classes, self.device,config)
        # elif config.model_name == "Deeplabv3":
        #     from .models import DeepLabV3Model
        #     self.model = DeepLabV3Model(config.num_classes, self.device, config)
        # else:
        #     raise ValueError("Unknown task")
        
        # Initialize pools
        if not skip_cold_start:
            self._init_pools()
        
        # Tracking
        self.cycle = 0
        self.best_score = float("-inf")
        self.history = {}
        
        print(f"Active Learning System initialized with:")
        print(f"  Device: {self.device}")
        print(f"  Cold Start Strategy: {config.cold_start_strategy}")
        print(f"  Query Strategy: {config.query_strategy}")
        if not skip_cold_start:
            print(f"  Initial labeled: {len(self.labeled_indices)} samples")
        else:
            print(f"  Skipped cold start. Full dataset will be used for training.")
            self.labeled_indices = [("train", i) for i in range(len(self.dataset_train))]
            if self.config.pool:
                self.labeled_indices += [("pool", i) for i in range(len(self.dataset_pool))]
            self.unlabeled_indices = []
    
    def _load_datasets(self):
        """Load training and validation datasets."""
        # Using pytorch_mask_rcnn or custom dataset loader
        import pytorch_mask_rcnn as pmr
        
        self.dataset_train = pmr.datasets(
            self.config.dataset, 
            self.config.data_dir, 
            "train", 
            train=True
        )
        self.dataset_val = pmr.datasets(
            self.config.dataset,
            self.config.data_dir,
            "valid",
            train=True
        )
        self.num_classes = max(self.dataset_train.classes) + 1
    
    def _init_pools(self):
        train_indices = list(range(len(self.dataset_train)))
        pool_indices = list(range(len(self.dataset_pool))) if self.config.pool else []

        all_samples = [("train", i) for i in train_indices] + \
                    [("pool", i) for i in pool_indices]

        # initial labeled only from original train split
        if 0 <= self.config.initial_labeled <= 1:
            n_labeled = int(self.config.initial_labeled * len(train_indices))
        else:
            n_labeled = min(int(self.config.initial_labeled), len(train_indices))

        labeled_train = self.cold_start.apply(
            strategy_name=self.config.cold_start_strategy,
            n_samples=n_labeled,
            all_indices=train_indices
        )

        self.labeled_indices = [("train", i) for i in labeled_train]
        self.unlabeled_indices = [s for s in all_samples if s not in self.labeled_indices]

        self.logger.info(
            f"Initialized with {len(self.labeled_indices)} labeled "
            f"and {len(self.unlabeled_indices)} unlabeled samples"
        )

    def get_labeled_dataset(self):
        return MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.labeled_indices
        )
    
    def get_unlabeled_dataset(self):
        return MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.unlabeled_indices
        )
    
    def set_labeled_indices(self, labeled_indices):
        self.labeled_indices = list(labeled_indices)

        all_samples = [("train", i) for i in range(len(self.dataset_train))]
        if self.config.pool:
            all_samples += [("pool", i) for i in range(len(self.dataset_pool))]

        self.unlabeled_indices = [
            s for s in all_samples if s not in self.labeled_indices
        ]

        self.logger.info(
            f"Manually set labeled pool: "
            f"{len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )

    def _get_sample_name(self, source, idx):

        dataset = self.dataset_train if source == "train" else self.dataset_pool

        # COCO-style datasets
        if hasattr(dataset, "coco"):
            img_id = dataset.ids[idx]
            return dataset.coco.imgs[img_id]["file_name"]

        # generic fallback
        if hasattr(dataset, "images"):
            return dataset.images[idx]

        return f"{source}_{idx}"
    
    def train(self, epochs: Optional[int] = None):
        if epochs is None:
            epochs = self.config.epochs_per_cycle

        labeled_dataset = self.get_labeled_dataset()

        print(f"\nTraining cycle {self.cycle} with {len(self.labeled_indices)} samples")

        cycle_metrics = []

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.model.train_epoch(
                labeled_dataset,
                epoch + self.cycle * epochs,
                epochs
            )

            # Evaluate
            eval_metrics = self.model.evaluate(self.dataset_val)

            epoch_time = time.time() - epoch_start

            # ✅ LOG AFTER EACH EPOCH
            self._log_metrics(
                epoch=epoch,
                train_time=epoch_time,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )
            self.save_results()
            cycle_metrics.append(eval_metrics)

            # Save checkpoint
            current_score = self.get_primary_metric(self.config.task, eval_metrics)
            if current_score > self.best_score:
                self.best_score = current_score
                save_checkpoint(
                    model=self.model,
                    cycle=self.cycle,
                    epoch=epoch,
                    score=current_score,
                    is_best=True,
                    config=self.config
                )

        self.cycle += 1
        return cycle_metrics
    
    def query(self, query_size: Optional[int] = None):
        self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB before querying")
        if query_size is None:
            if self.config.query_size <= 1:
                query_size = int(self.config.query_size * len(self.unlabeled_indices))
            else:
                query_size = self.config.query_size

        if len(self.unlabeled_indices) == 0:
            self.logger.warning("No unlabeled samples left!")
            return []

        unlabeled_dataset = self.get_unlabeled_dataset()
        local_indices = list(range(len(unlabeled_dataset)))
        self.logger.info(f"unlabeled pool size: {len(unlabeled_dataset)} samples, querying {query_size} samples")
        uncertainties = self.query_strategy.calculate_uncertainty(
            model=self.model,
            dataset=unlabeled_dataset,
            indices=local_indices,
            device=self.device
        )
        self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB after uncertainty calculation")
        selected_indices = self.query_strategy.select_samples(
            strategy_name=self.config.query_strategy,
            uncertainties=uncertainties,
            dataset=unlabeled_dataset,
            indices=local_indices,
            query_size=query_size
        )
        self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB after selection")
        selected_global_indices = [self.unlabeled_indices[i] for i in selected_indices]
        selected_samples_info = []

        for i in selected_indices:

            source, idx = self.unlabeled_indices[i]

            name = self._get_sample_name(source, idx)

            score = float(uncertainties[i])

            selected_samples_info.append({
                "source": source,
                "index": idx,
                "name": name,
                "uncertainty": score
            })
        self.labeled_indices.extend(selected_global_indices)
        self.unlabeled_indices = [
            idx for i, idx in enumerate(self.unlabeled_indices)
            if i not in selected_indices
        ]

        n_train = sum(1 for s, _ in selected_global_indices if s == "train")
        n_pool = sum(1 for s, _ in selected_global_indices if s == "pool")

        self.logger.info(
            f"Selected {len(selected_global_indices)} new samples "
            f"({n_train} from train, {n_pool} from pool). "
            f"Now {len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
        self.logger.info("Selected sample names:")

        for s in selected_samples_info[:10]:  # avoid huge logs
            self.logger.info(f"{s['source']} | {s['name']}")
        self.history.setdefault("selected_train_count", []).append(n_train)
        self.history.setdefault("selected_pool_count", []).append(n_pool)
        self.history.setdefault("selected_samples", []).append(selected_samples_info)
        return selected_global_indices
    
    def run(self):
        """Run complete active learning process."""
        self.logger.info("Starting active learning process...")
        run_start_time = datetime.datetime.now()
        all_metrics = []
        
        # Initial training
        initial_metrics = self.train(epochs=self.config.initial_training_epoch)
        all_metrics.append(initial_metrics)
        
        # Active learning cycles
        for cycle in range(self.config.al_cycles):
            self.logger.info(f"\n=== AL Cycle {cycle + 1}/{self.config.al_cycles} ===")
            
            # Query new samples
            self.query()
            
            # Train on expanded dataset
            cycle_metrics = self.train()
            all_metrics.append(cycle_metrics)
        
        self.logger.info("Active learning completed!")
        run_time = str(datetime.datetime.now() - run_start_time)
        self.logger.info(f"RL Active Learning completed in {run_time}")
        self.history["run_time"] = run_time
        system_time = str(datetime.datetime.now() - self.system_start_time)
        self.logger.info(f"Total system time: {system_time}")
        self.history["system_time"] = system_time
        self.save_results()
        return all_metrics
    
    def _log_metrics(self, epoch, train_time, train_metrics, eval_metrics):
        global_epoch = epoch + self.cycle * self.config.epochs_per_cycle

        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("global_epoch", []).append(global_epoch)
        self.history.setdefault("cycle", []).append(self.cycle)

        self.history.setdefault("train_loss", []).append(train_metrics.get("train_loss", 0))
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))
        self.history.setdefault("train_time", []).append(train_time)

        # ----------------------------
        # Semantic segmentation models
        # ----------------------------
        if self.config.task == "segmentation":

            dice = eval_metrics.get("dice", 0)
            f1 = eval_metrics.get("f1", 0)
            miou = eval_metrics.get("mean_iou", 0)

            self.history.setdefault("val_dice", []).append(dice)
            self.history.setdefault("val_F1", []).append(f1)
            self.history.setdefault("val_mean_iou", []).append(miou)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics.get('train_loss',0):.4f} | "
                f"F1: {f1:.4f} | "
                f"Dice: {dice:.4f} | "
                f"Mean IoU: {miou:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if self.config.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics.get("train_loss",0),
                        "val_dice": dice,
                        "val_iou": miou,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )

        # ---------------------------------
        # Instance segmentation / detection
        # ---------------------------------
        elif self.config.task in ["instance_segmentation", "detection"]:

            bbox_ap = eval_metrics.get("bbox_AP", eval_metrics.get("bbox_mAP50_95", 0))
            bbox_ap50 = eval_metrics.get("bbox_AP50", eval_metrics.get("bbox_mAP50", 0))

            # Keep old key so old graph/explorer code does not break
            self.history.setdefault("val_bbox_AP", []).append(bbox_ap)

            # Clearer names for future plots
            self.history.setdefault("val_bbox_AP50", []).append(bbox_ap50)
            self.history.setdefault("val_bbox_AP50_95", []).append(bbox_ap)

            # Only meaningful for Mask R-CNN / instance segmentation
            if self.config.task == "instance_segmentation":
                mask_ap = eval_metrics.get("mask_AP", 0)
                self.history.setdefault("val_mask_AP", []).append(mask_ap)
            else:
                mask_ap = None
                # Optional: keep old key to avoid explorer errors
                self.history.setdefault("val_mask_AP", []).append(0.0)

            # Optional per-class detection metrics
            per_class = eval_metrics.get("per_class", {})

            for class_name, class_metrics in per_class.items():
                safe_name = (
                    class_name
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("-", "_")
                )

                ap50 = class_metrics.get("AP50", 0)
                ap5095 = class_metrics.get("AP50_95", class_metrics.get("AP", 0))

                self.history.setdefault(
                    f"val_class_{safe_name}_AP50", []
                ).append(ap50)

                self.history.setdefault(
                    f"val_class_{safe_name}_AP50_95", []
                ).append(ap5095)

            msg = (
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
                f"BBox AP50: {bbox_ap50:.4f} | "
                f"BBox AP50-95: {bbox_ap:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if mask_ap is not None:
                msg += f" | Mask AP: {mask_ap:.4f}"

            self.logger.info(msg)

            if self.config.use_wandb:
                wandb_log = {
                    "epoch": epoch + 1,
                    "global_epoch": global_epoch,
                    "cycle": self.cycle,
                    "train_loss": train_metrics["train_loss"],
                    "val_bbox_AP": bbox_ap,
                    "val_bbox_AP50": bbox_ap50,
                    "val_bbox_AP50_95": bbox_ap,
                    "labeled_count": len(self.labeled_indices),
                }

                if mask_ap is not None:
                    wandb_log["val_mask_AP"] = mask_ap

                for class_name, class_metrics in per_class.items():
                    safe_name = (
                        class_name
                        .replace(" ", "_")
                        .replace("/", "_")
                        .replace("-", "_")
                    )

                    wandb_log[f"val_class_{safe_name}_AP50"] = class_metrics.get("AP50", 0)
                    wandb_log[f"val_class_{safe_name}_AP50_95"] = class_metrics.get(
                        "AP50_95",
                        class_metrics.get("AP", 0)
                    )

                log_to_wandb(wandb_log, step=global_epoch)

        # ---------------------------------
        # Multi-label classification
        # ---------------------------------
        elif self.config.task == "multilabel_classification":

            macro_f1 = eval_metrics.get("macro_f1", 0)
            micro_f1 = eval_metrics.get("micro_f1", 0)
            map_score = eval_metrics.get("map", 0)
            hamming = eval_metrics.get("hamming_loss", 0)

            self.history.setdefault("val_macro_f1", []).append(macro_f1)
            self.history.setdefault("val_micro_f1", []).append(micro_f1)
            self.history.setdefault("val_map", []).append(map_score)
            self.history.setdefault("val_hamming_loss", []).append(hamming)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics.get('train_loss',0):.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Micro-F1: {micro_f1:.4f} | "
                f"mAP: {map_score:.4f} | "
                f"Hamming: {hamming:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

        # ---------------------------------
        # Single-label classification
        # ---------------------------------
        elif self.config.task in ["classification", "multiclass_classification", "binary_classification"]:

            macro_f1 = eval_metrics.get("macro_f1", eval_metrics.get("f1", 0))
            micro_f1 = eval_metrics.get("micro_f1", 0)
            acc = eval_metrics.get("accuracy", 0)
            bal_acc = eval_metrics.get("balanced_accuracy", 0)
            f2 = eval_metrics.get("f2", 0)

            self.history.setdefault("val_macro_f1", []).append(macro_f1)
            self.history.setdefault("val_micro_f1", []).append(micro_f1)
            self.history.setdefault("val_accuracy", []).append(acc)
            self.history.setdefault("val_balanced_accuracy", []).append(bal_acc)
            self.history.setdefault("val_f2", []).append(f2)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics.get('train_loss',0):.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Micro-F1: {micro_f1:.4f} | "
                f"Acc: {acc:.4f} | "
                f"Balanced Acc: {bal_acc:.4f} | "
                f"F2: {f2:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

    def save_results(self):
        results = {
            "config": self._config_to_dict(),
            "history": self.history,
        }

        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

    def _init_results_path(self):
        date_folder = datetime.datetime.now().strftime("%m_%d")
        results_dir = os.path.join(self.config.results_dir, date_folder)
        os.makedirs(results_dir, exist_ok=True)

        time_stamp = datetime.datetime.now().strftime("%H%M")

        self.results_path = os.path.join(
            results_dir,
            f"{self.config.experiment_name}_"
            f"{self.config.dataset_type}_"
            f"{self.config.cold_start_strategy}_"
            f"{self.config.query_strategy}_"
            f"{time_stamp}.json"
        )

        self.logger.info(f"the results will be saved in: {self.results_path}")

    def _config_to_dict(self):
        # works for argparse.Namespace or simple config objects, while making
        # tensors JSON-serializable.
        out = {}
        for k, v in vars(self.config).items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().tolist()
            else:
                out[k] = v
        return out
    
    def plot_results(self):
        """Save a lightweight learning-curve plot if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            metric_candidates = [
                "val_macro_f1", "val_F1", "val_dice",
                "val_mask_AP", "val_bbox_AP", "val_accuracy"
            ]
            metric_name = next((m for m in metric_candidates if m in self.history), None)
            if metric_name is None or "labeled_count" not in self.history:
                return None

            x = self.history["labeled_count"]
            y = self.history[metric_name]

            plt.figure(figsize=(7, 5))
            plt.plot(x, y, marker="o")
            plt.xlabel("Labeled samples")
            plt.ylabel(metric_name)
            plt.title(self.config.experiment_name)
            plt.grid(True, alpha=0.3)
            out_path = self.results_path.replace(".json", f"_{metric_name}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            self.logger.info(f"Saved plot to: {out_path}")
            return out_path
        except Exception as e:
            self.logger.warning(f"Could not plot results: {e}")
            return None

    def get_primary_metric(self, task, metrics):

        if task == "segmentation":
            return metrics.get("f1", metrics.get("dice", 0))

        if task == "instance_segmentation":
            return metrics.get("mask_AP", 0)

        if task == "detection":
            return metrics.get("bbox_AP", 0)
        if task == "multilabel_classification":
            return metrics.get("macro_f1", metrics.get("map", 0))

        if task in ["classification", "multiclass_classification", "binary_classification"]:
            return metrics.get("macro_f1", metrics.get("accuracy", 0))

        return 0
    