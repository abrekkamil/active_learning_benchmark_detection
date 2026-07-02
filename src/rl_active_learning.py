import torch
import torch.nn.functional as F
import numpy as np
import time
import datetime
from typing import List, Dict, Tuple, Optional
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from .models import UNetModel, PolicyNet
from .utils import setup_logging, set_seed, universal_collate, save_checkpoint
from .data_modules.factory import load_dataset
from .data_modules.sample_utils import unpack_sample
from .cold_start_strategies import ColdStartStrategies
from .models import build_model
from .models.utils import _ensure_rgb

import wandb

def log_to_wandb(metrics, step=None):
    wandb.log(metrics, step=step)

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
        

class ActiveLearningSystemRL:
    """
    Reinforcement Learning–based Active Learning system
    Compatible with ActiveLearningConfig and existing datasets.
    """

    def __init__(self, config,  skip_cold_start: bool = False):
        self.system_start_time = datetime.datetime.now()
        self.config = config
        set_seed(config.seed)
        self.cycle = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        self.prev_score = None

        self.logger = setup_logging(f"{config.experiment_name}_RL")
        self._init_results_path()
        # --------------------
        # Datasets
        # --------------------
        self.logger.info("Loading train dataset...")
        self.dataset_train = load_dataset(config, split="train")
        self.logger.info(f"Loaded train dataset: {len(self.dataset_train)} samples")
        self.dataset_val   = load_dataset(config, split="val")
        self.logger.info(f"Loaded val dataset: {len(self.dataset_val)} samples")
        self.dataset_pool = None
        if config.pool:
            self.logger.info("Loading pool dataset...")
            self.dataset_pool = load_dataset(config, split="pool")
            self.logger.info(f"Loaded pool dataset: {len(self.dataset_pool)} samples")

        if hasattr(self.dataset_train, "num_classes"):
            self.config.num_classes = int(self.dataset_train.num_classes)

        if hasattr(self.dataset_train, "class_weights"):
            self.config.class_weights = self.dataset_train.class_weights
        # --------------------
        # Models
        # --------------------
        self.oracle_model = build_model(
            config.model_name,
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )
        self.main_model = build_model(
            config.model_name,
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )


        # --------------------
        # RL policy
        # --------------------
        # Infer bottleneck dimension dynamically
        sample_img, _ = unpack_sample(self.dataset_train[0])
        sample_img = _ensure_rgb(sample_img).to(self.device)

        if self.config.task in ["detection", "instance_segmentation"]:
            sample_input = [sample_img]
        else:
            sample_input = sample_img.unsqueeze(0)

        with torch.no_grad():
            feat = self.oracle_model.model.get_bottleneck_features(sample_input)
        feat = self._ensure_2d_tensor(feat, name="sample bottleneck features")
        bottleneck_dim = feat.shape[1]

        self.state_dim = bottleneck_dim + 3  # +3 for uncertainty features
        budget_options = getattr(
            self.config,
            "budget_options",
            [getattr(self.config, "query_size", 1)]
        )

        budget_mode = getattr(self.config, "budget_mode", "discrete").lower()

        if budget_mode == "continuous":
            num_budget_options = 1

        elif budget_mode == "discrete":
            budget_options = getattr(
                self.config,
                "budget_options",
                [getattr(self.config, "query_size", 1)]
            )
            budget_options = [max(1, int(float(b))) for b in budget_options]
            self.config.budget_options = budget_options
            num_budget_options = len(budget_options)

        else:
            raise ValueError(
                f"Unknown budget_mode={budget_mode}. "
                "Use 'discrete' or 'continuous'."
            )

        self.policy = PolicyNet(
            self.state_dim,
            hidden_dim=self.config.policy_hidden,
            num_budget_options=num_budget_options,
        ).to(self.device)


        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=getattr(config, "policy_lr", 1e-4),
        )

        self.entropy_beta = getattr(config, "entropy_beta", 1e-3)
        self.policy_temp = getattr(config, "policy_temp", 1.0)

        # --------------------
        # Pools
        # --------------------
        train_indices = list(range(len(self.dataset_train)))
        pool_indices = list(range(len(self.dataset_pool))) if config.pool else []
        all_samples = [("train", i) for i in train_indices] + \
              [("pool", i) for i in pool_indices]
        
        self.total_samples = len(all_samples)
        
        if skip_cold_start:
            # FULL DATASET (upper bound)
            self.labeled_indices = all_samples
            self.unlabeled_indices = []

            self.logger.info(
                "RL AL initialized with FULL dataset (skip cold start)"
            )

        else:
            n_init = (
                int(config.initial_labeled * len(train_indices))
                if config.initial_labeled <= 1
                else int(config.initial_labeled)
            )

            cold_start = ColdStartStrategies(self.dataset_train, config)
            self.logger.info(
                f"Applying cold start strategy={config.cold_start_strategy}, n_init={n_init}"
            )
            labeled_train = cold_start.apply(
                strategy_name=config.cold_start_strategy,
                n_samples=n_init,
                all_indices=train_indices
            )
            self.labeled_indices = [("train", i) for i in labeled_train]

            self.unlabeled_indices = [
                s for s in all_samples if s not in self.labeled_indices
            ]

            self.logger.info(
                f"RL AL initialized with {len(self.labeled_indices)} labeled "
                f"and {len(self.unlabeled_indices)} unlabeled samples"
            )


        # --------------------
        # Tracking
        # --------------------
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9
        self.history = {}
        # Best validation score for saving the best RAL main model
        self.best_score = float("-inf")
        self.logger.info(
            f"RL AL initialized with {len(self.labeled_indices)} labeled samples"
        )
        # --------------------
        # Train oracle model (ONCE)
        # --------------------
        self.logger.info("Training oracle model on initial labeled set")

        oracle_dataset = MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.labeled_indices
        )
        for ep in range(self.config.oracle_epochs):
            self.oracle_model.train_epoch(
                oracle_dataset, ep, self.config.oracle_epochs
            )

        self.oracle_model.eval()
        for p in self.oracle_model.model.parameters():
            p.requires_grad = False
    # ==========================================================
    # Feature + uncertainty → state
    # ==========================================================
    def set_labeled_indices(self, labeled_indices: List[Tuple[str, int]]):
        """
        Manually set the initial labeled pool (override cold start).
        Useful for cold start experiments and ablations.
        """
        all_samples = [("train", i) for i in range(len(self.dataset_train))]
        if self.config.pool:
            all_samples += [("pool", i) for i in range(len(self.dataset_pool))]

        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = [
            i for i in all_samples if i not in self.labeled_indices
        ]

        self.logger.info(
            f"Manually set labeled pool: "
            f"{len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
    def get_primary_metric(self, task, metrics):

        if task == "segmentation":
            return metrics.get("f1", 0)

        if task == "instance_segmentation":
            return metrics.get("mask_AP", 0)

        if task == "detection":
            return metrics.get("bbox_AP", 0)
        
        if task == "multilabel_classification":
            return metrics.get("macro_f1", metrics.get("map", 0))

        if task in ["classification", "multiclass_classification", "binary_classification"]:
            return metrics.get("macro_f1", metrics.get("accuracy", 0))
        

        return 0

    def _get_sample_name(self, source, idx):

        dataset = self.dataset_train if source == "train" else self.dataset_pool

        if hasattr(dataset, "coco"):
            img_id = dataset.ids[idx]
            return dataset.coco.imgs[img_id]["file_name"]

        return f"{source}_{idx}"   
    
    def _get_cost_denominator(self):
        """
        Denominator used for dynamic-query cost penalty.

        cost_denom_mode:
            total     -> budget / total number of samples
            unlabeled -> budget / remaining unlabeled pool
            labeled   -> budget / current labeled set
        """

        mode = getattr(self.config, "cost_denom_mode", "total")

        if mode == "total":
            return max(1, int(self.total_samples))

        if mode == "unlabeled":
            return max(1, len(self.unlabeled_indices))

        if mode == "labeled":
            return max(1, len(self.labeled_indices))

        raise ValueError(
            f"Unknown cost_denom_mode: {mode}. "
            "Use one of: 'total', 'unlabeled', 'labeled'."
        )

    def forward_model(self, images):
        """
        Forward pass for oracle model.

        Detection models such as Faster R-CNN require:
            images = list[Tensor[C, H, W]]
        and every tensor must be on the same device as the model.
        """

        if self.config.task in ["detection", "instance_segmentation"]:

            if torch.is_tensor(images):
                if images.dim() == 3:
                    images = [_ensure_rgb(images).to(self.device)]
                elif images.dim() == 4:
                    images = [_ensure_rgb(img).to(self.device) for img in images]
                else:
                    raise ValueError(f"Unexpected detection image tensor shape: {images.shape}")

            else:
                images = [_ensure_rgb(img).to(self.device) for img in images]

            return self.oracle_model.model(images)

        try:
            # HuggingFace-style, for example SegFormer
            return self.oracle_model.model(pixel_values=images)

        except TypeError:
            # Torchvision / U-Net / classification style
            if torch.is_tensor(images):
                images = images.to(self.device)

            return self.oracle_model.model(images)

    def _ensure_2d_tensor(self, x, name="tensor"):
        """
        Convert model features/logits to a 2D tensor [B, D].
        This prevents crashes when bottleneck features come as:
        [D]
        [B, D]
        [B, C, H, W]
        list of tensors
        """
        if isinstance(x, (list, tuple)):
            xs = []

            for item in x:
                if not torch.is_tensor(item):
                    item = torch.as_tensor(item, dtype=torch.float32, device=self.device)
                else:
                    item = item.to(self.device, dtype=torch.float32)

                if item.ndim == 1:
                    item = item.unsqueeze(0)
                elif item.ndim > 2:
                    item = item.flatten(start_dim=1)

                xs.append(item)

            x = torch.cat(xs, dim=0)

        else:
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            else:
                x = x.to(self.device, dtype=torch.float32)

            if x.ndim == 1:
                x = x.unsqueeze(0)
            elif x.ndim > 2:
                x = x.flatten(start_dim=1)

        if x.ndim != 2:
            raise ValueError(f"{name} should be 2D [B, D], got shape {tuple(x.shape)}")

        return x


    def _extract_logits(self, outputs):
        """
        Robustly extract logits from different model output formats.
        Works for:
        tensor
        dict with 'out'
        dict with 'logits'
        HuggingFace-style output.logits
        tuple/list whose first item is logits
        """
        if isinstance(outputs, dict):
            if "out" in outputs:
                return outputs["out"]
            if "logits" in outputs:
                return outputs["logits"]
            raise ValueError(f"Could not find logits in output dict keys: {outputs.keys()}")

        if hasattr(outputs, "logits"):
            return outputs.logits

        if torch.is_tensor(outputs):
            return outputs

        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            if torch.is_tensor(outputs[0]):
                return outputs[0]

        raise ValueError(f"Unknown model output format: {type(outputs)}")

    def _compute_state(self, images):

        with torch.no_grad():

            feats = self.oracle_model.model.get_bottleneck_features(images).detach()
            feats = self._ensure_2d_tensor(feats, name="bottleneck features")

            outputs = self.forward_model(images)

            # ======================================================
            # DETECTION / INSTANCE SEGMENTATION
            # ======================================================
            if self.config.task in ["detection", "instance_segmentation"]:

                entropy_list = []
                confidence_list = []
                margin_list = []

                for out in outputs:

                    scores = out.get("scores", torch.empty(0, device=self.device))

                    if scores.numel() == 0:
                        entropy_list.append(torch.tensor(1.0, device=self.device))
                        confidence_list.append(torch.tensor(0.0, device=self.device))
                        margin_list.append(torch.tensor(0.0, device=self.device))
                        continue

                    scores = scores.to(self.device).clamp(1e-6, 1.0 - 1e-6)

                    entropy = -(
                        scores * torch.log(scores)
                        + (1.0 - scores) * torch.log(1.0 - scores)
                    ).mean()

                    entropy = entropy / np.log(2.0)

                    confidence = scores.max()

                    if scores.numel() > 1:
                        top2 = torch.topk(scores, k=2).values
                        margin = top2[0] - top2[1]
                    else:
                        margin = scores[0]

                    entropy_list.append(entropy)
                    confidence_list.append(confidence)
                    margin_list.append(margin)

                entropy = torch.stack(entropy_list)
                confidence = torch.stack(confidence_list)
                margin = torch.stack(margin_list)

            # ======================================================
            # MULTILABEL CLASSIFICATION
            # ======================================================
            elif self.config.task == "multilabel_classification":

                logits = self._extract_logits(outputs)

                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)

                probs = torch.sigmoid(logits)
                eps = 1e-8

                entropy = -(
                    probs * torch.log(probs + eps)
                    + (1.0 - probs) * torch.log(1.0 - probs + eps)
                ).mean(dim=1)

                entropy = entropy / np.log(2.0)

                confidence = torch.max(probs, 1.0 - probs).mean(dim=1)
                margin = torch.abs(probs - 0.5).mul(2.0).mean(dim=1)

            # ======================================================
            # SINGLE-LABEL CLASSIFICATION
            # ======================================================
            elif self.config.task in [
                "classification",
                "multiclass_classification",
                "binary_classification",
            ]:

                logits = self._extract_logits(outputs)

                # Binary model with one logit: [B] or [B, 1]
                if logits.ndim == 1 or logits.shape[1] == 1:
                    logits = logits.reshape(-1)
                    probs_pos = torch.sigmoid(logits).clamp(1e-8, 1.0 - 1e-8)

                    entropy = -(
                        probs_pos * torch.log(probs_pos)
                        + (1.0 - probs_pos) * torch.log(1.0 - probs_pos)
                    )

                    entropy = entropy / np.log(2.0)

                    confidence = torch.max(probs_pos, 1.0 - probs_pos)
                    margin = torch.abs(probs_pos - 0.5).mul(2.0)

                # Multiclass or binary with two logits: [B, C]
                else:
                    probs = F.softmax(logits, dim=1).clamp(1e-8, 1.0)
                    num_classes = probs.shape[1]

                    entropy = -(probs * torch.log(probs)).sum(dim=1)

                    if num_classes > 1:
                        entropy = entropy / np.log(float(num_classes))

                    confidence = probs.max(dim=1).values

                    top2 = torch.topk(probs, k=min(2, num_classes), dim=1).values

                    if top2.shape[1] == 1:
                        margin = top2[:, 0]
                    else:
                        margin = top2[:, 0] - top2[:, 1]

            # ======================================================
            # SEMANTIC SEGMENTATION
            # ======================================================
            elif self.config.task == "segmentation":

                logits = self._extract_logits(outputs)

                if logits.ndim != 4:
                    raise ValueError(
                        f"Segmentation logits should be [B, C, H, W], "
                        f"got shape {tuple(logits.shape)}"
                    )

                # Binary segmentation with one output channel: [B, 1, H, W]
                if logits.shape[1] == 1:
                    probs_pos = torch.sigmoid(logits[:, 0]).clamp(1e-8, 1.0 - 1e-8)

                    entropy = -(
                        probs_pos * torch.log(probs_pos)
                        + (1.0 - probs_pos) * torch.log(1.0 - probs_pos)
                    ).mean(dim=[1, 2])

                    entropy = entropy / np.log(2.0)

                    confidence = torch.max(probs_pos, 1.0 - probs_pos).mean(dim=[1, 2])
                    margin = torch.abs(probs_pos - 0.5).mul(2.0).mean(dim=[1, 2])

                # Multiclass segmentation: [B, C, H, W]
                else:
                    probs = F.softmax(logits, dim=1).clamp(1e-8, 1.0)
                    num_classes = probs.shape[1]

                    entropy = -(probs * torch.log(probs)).sum(dim=1).mean(dim=[1, 2])

                    if num_classes > 1:
                        entropy = entropy / np.log(float(num_classes))

                    confidence = probs.max(dim=1).values.mean(dim=[1, 2])

                    top2 = torch.topk(probs, k=min(2, num_classes), dim=1).values

                    if top2.shape[1] == 1:
                        margin = top2[:, 0].mean(dim=[1, 2])
                    else:
                        margin = (top2[:, 0] - top2[:, 1]).mean(dim=[1, 2])

            else:
                raise ValueError(f"Unsupported task type: {self.config.task}")

            uncertainty = torch.stack(
                [
                    entropy,
                    1.0 - confidence,
                    1.0 - margin,
                ],
                dim=1,
            )

            if uncertainty.shape[0] != feats.shape[0]:
                raise ValueError(
                    f"Feature/uncertainty batch mismatch: "
                    f"features={feats.shape}, uncertainty={uncertainty.shape}"
                )

            state = torch.cat([feats, uncertainty], dim=1)

            if state.shape[1] != self.state_dim:
                raise ValueError(
                    f"State dimension mismatch inside _compute_state: "
                    f"got {state.shape[1]}, expected {self.state_dim}"
                )

            return state

    # ==========================================================
    # RL query step
    # ==========================================================
    def query(self):

        if len(self.unlabeled_indices) == 0:
            return [], None, None, None

        # ==========================================================
        # Build pool
        # ==========================================================
        unlabeled_dataset = MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.unlabeled_indices
        )

        loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=universal_collate
        )

        state_batches = []

        # ==========================================================
        # Compute RL states
        # ==========================================================
        self.oracle_model.eval()

        with torch.no_grad():

            for images, _ in tqdm(loader, desc="Computing RL states", leave=False):

                if self.config.task in ["detection", "instance_segmentation"]:
                    images = [_ensure_rgb(img).to(self.device) for img in images]
                    batch_states = self._compute_state(images)

                else:
                    images = [_ensure_rgb(img) for img in images]
                    images = torch.stack(images).to(self.device)
                    batch_states = self._compute_state(images)

                # --------------------------------------------------
                # Make every batch state safely [B, state_dim]
                # --------------------------------------------------
                if not torch.is_tensor(batch_states):
                    batch_states = torch.as_tensor(
                        batch_states,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    batch_states = batch_states.detach().to(
                        device=self.device,
                        dtype=torch.float32
                    )

                if batch_states.ndim == 1:
                    batch_states = batch_states.unsqueeze(0)

                if batch_states.ndim != 2:
                    raise ValueError(
                        f"Expected batch_states to be 2D [B, state_dim], "
                        f"got shape {tuple(batch_states.shape)}"
                    )

                state_batches.append(batch_states)

        if len(state_batches) == 0:
            return [], None, None, None

        states = torch.cat(state_batches, dim=0)

        if states.ndim != 2:
            raise ValueError(
                f"Expected states to be 2D [N, state_dim], "
                f"got shape {tuple(states.shape)}"
            )

        if states.shape[1] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: got {states.shape[1]}, "
                f"expected {self.state_dim}"
            )

        if states.shape[0] != len(self.unlabeled_indices):
            raise ValueError(
                f"Number of states does not match unlabeled pool: "
                f"states={states.shape[0]}, "
                f"unlabeled={len(self.unlabeled_indices)}. "
                f"This means _compute_state() is not returning one state per image."
            )

        self.logger.info(f"RL states shape: {tuple(states.shape)}")

        # ==========================================================
        # Candidate Filtering
        # ==========================================================
        entropy_scores = states[:, -3]

        candidate_ratio = getattr(self.config, "candidate_ratio", 0.2)
        top_k = int(candidate_ratio * len(entropy_scores))

        # If discrete mode is used, make sure candidate pool is at least as large
        # as the largest possible selected budget.
        if getattr(self.config, "dynamic_query_size", False):
            budget_mode = getattr(self.config, "budget_mode", "discrete").lower()

            if budget_mode == "discrete":
                budget_options = getattr(
                    self.config,
                    "budget_options",
                    [getattr(self.config, "query_size", 1)]
                )
                budget_options = [max(1, int(float(b))) for b in budget_options]
                top_k = max(top_k, max(budget_options))

        else:
            query_size = getattr(self.config, "query_size", 1)

            if query_size <= 1:
                min_query = max(1, int(query_size * self.total_samples))
            else:
                min_query = int(query_size)

            top_k = max(top_k, min_query)

        top_k = max(1, min(top_k, len(entropy_scores)))

        _, candidate_idx = torch.topk(
            entropy_scores,
            k=top_k,
            largest=True,
            sorted=False
        )

        candidate_states = states.index_select(0, candidate_idx)

        candidate_idx_list = candidate_idx.detach().cpu().tolist()

        candidate_pool = [
            self.unlabeled_indices[i] for i in candidate_idx_list
        ]

        if len(candidate_pool) == 0:
            return [], None, None, None

        # ==========================================================
        # Policy Forward
        # ==========================================================
        global_state = candidate_states.mean(dim=0)

        image_logits, budget_logits = self.policy(
            candidate_states,
            global_state
        )

        # ==========================================================
        # Query size
        # ==========================================================
        if getattr(self.config, "dynamic_query_size", False):

            budget_mode = getattr(self.config, "budget_mode", "discrete").lower()

            # ======================================================
            # Continuous / old scalar behaviour
            # ======================================================
            if budget_mode == "continuous":

                budget_logits = budget_logits.reshape(-1)

                if budget_logits.numel() != 1:
                    raise ValueError(
                        f"Continuous budget mode expects 1 budget logit, "
                        f"but got {budget_logits.numel()}."
                    )

                budget_ratio_min = getattr(self.config, "budget_ratio_min", 0.01)
                budget_ratio_max = getattr(self.config, "budget_ratio_max", 0.15)

                budget_ratio = torch.sigmoid(budget_logits[0])
                budget_ratio = torch.clamp(
                    budget_ratio,
                    min=budget_ratio_min,
                    max=budget_ratio_max
                )

                budget = int(budget_ratio.item() * len(candidate_pool))
                budget = max(1, min(budget, len(candidate_pool)))

                log_prob_budget = torch.log(budget_ratio + 1e-12)

                p = budget_ratio
                entropy_budget = -(
                    p * torch.log(p + 1e-12)
                    + (1.0 - p) * torch.log(1.0 - p + 1e-12)
                )

                self.history.setdefault("selected_budget_ratio", []).append(
                    float(budget_ratio.detach().cpu().item())
                )

                self.logger.info(
                    f"Continuous dynamic budget selected: "
                    f"ratio={budget_ratio.item():.4f}, "
                    f"budget={budget}, "
                    f"candidate_pool={len(candidate_pool)}"
                )

            # ======================================================
            # Discrete / new categorical behaviour
            # ======================================================
            elif budget_mode == "discrete":

                budget_options = getattr(
                    self.config,
                    "budget_options",
                    [250, 500, 750, 1000]
                )
                budget_options = [max(1, int(float(b))) for b in budget_options]

                budget_logits = budget_logits.reshape(-1)

                if budget_logits.numel() != len(budget_options):
                    raise ValueError(
                        f"Discrete budget mode mismatch: "
                        f"budget_logits={budget_logits.numel()}, "
                        f"budget_options={len(budget_options)}"
                    )

                budget_probs = F.softmax(
                    budget_logits / self.policy_temp,
                    dim=0
                )

                budget_probs = torch.nan_to_num(
                    budget_probs,
                    nan=1.0 / budget_probs.numel(),
                    posinf=1.0 / budget_probs.numel(),
                    neginf=0.0
                )

                budget_probs = budget_probs.clamp_min(1e-12)
                budget_probs = budget_probs / budget_probs.sum()

                budget_dist = torch.distributions.Categorical(probs=budget_probs)
                budget_action = budget_dist.sample()

                selected_budget_option = budget_options[int(budget_action.item())]

                budget = min(selected_budget_option, len(candidate_pool))
                budget = max(1, int(budget))

                log_prob_budget = budget_dist.log_prob(budget_action)
                entropy_budget = budget_dist.entropy()

                self.history.setdefault("selected_budget_option", []).append(
                    int(selected_budget_option)
                )

                self.logger.info(
                    f"Discrete dynamic budget selected: {selected_budget_option} "
                    f"(effective budget after clamp: {budget}) | "
                    f"budget options: {budget_options}"
                )

            else:
                raise ValueError(
                    f"Unknown budget_mode={budget_mode}. "
                    "Use 'discrete' or 'continuous'."
                )

        else:
            query_size = getattr(self.config, "query_size", 1)

            if query_size <= 1:
                budget = int(query_size * self.total_samples)
            else:
                budget = int(query_size)

            budget = max(1, min(budget, len(candidate_pool)))

            log_prob_budget = torch.tensor(0.0, device=self.device)
            entropy_budget = torch.tensor(0.0, device=self.device)       
        # ==========================================================
        # Image Sampling
        # ==========================================================
        budget = int(budget)

        image_logits = image_logits.reshape(-1)

        if image_logits.numel() != len(candidate_pool):
            raise ValueError(
                f"Image logits/candidate pool mismatch: "
                f"image_logits={image_logits.numel()}, "
                f"candidate_pool={len(candidate_pool)}"
            )

        image_probs = F.softmax(
            image_logits / self.policy_temp,
            dim=0
        )

        image_probs = torch.nan_to_num(
            image_probs,
            nan=1.0 / image_probs.numel(),
            posinf=1.0 / image_probs.numel(),
            neginf=0.0
        )

        image_probs = image_probs.clamp_min(1e-12)
        image_probs = image_probs / image_probs.sum()

        selected_pos = torch.multinomial(
            image_probs,
            num_samples=budget,
            replacement=False
        )

        log_prob_images = torch.log(image_probs[selected_pos]).sum()
        log_prob_sum = log_prob_images + log_prob_budget

        entropy_images = -(image_probs * torch.log(image_probs)).sum()
        entropy = entropy_images + entropy_budget

        selected_indices = [
            candidate_pool[int(i)] for i in selected_pos.detach().cpu().tolist()
        ]

        selected_samples_info = []

        for source, idx in selected_indices:
            name = self._get_sample_name(source, idx)

            selected_samples_info.append(
                {
                    "source": source,
                    "index": idx,
                    "name": name
                }
            )

        self.logger.info(
            f"Selected {len(selected_indices)} samples with budget {budget}"
        )

        self.logger.info(f"Selected samples: {selected_samples_info}")

        self.history.setdefault("selected_samples", []).append(selected_samples_info)

        n_train = sum(1 for s, _ in selected_indices if s == "train")
        n_pool = sum(1 for s, _ in selected_indices if s == "pool")

        self.logger.info(
            f"Selected {len(selected_indices)} samples "
            f"({n_train} train, {n_pool} pool)"
        )

        self.history.setdefault("selected_train_count", []).append(n_train)
        self.history.setdefault("selected_pool_count", []).append(n_pool)

        return selected_indices, log_prob_sum, entropy, budget
    
    def _maybe_save_best_checkpoint(self, eval_metrics, epoch):
        """
        Save the best RAL main model checkpoint based on the primary validation metric.

        For SDNET classification, this is macro_f1.
        For detection, this is bbox_AP.
        For segmentation, this is f1.
        """

        current_score = self.get_primary_metric(self.config.task, eval_metrics)

        if current_score is None:
            return

        current_score = float(current_score)

        if current_score > self.best_score:
            self.best_score = current_score

            save_checkpoint(
                model=self.main_model,
                cycle=self.cycle,
                epoch=epoch,
                score=current_score,
                is_best=True,
                config=self.config,
                additional_info={
                    "ral": True,
                    "policy_state_dict": self.policy.state_dict(),
                    "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                    "reward_baseline": self.reward_baseline,
                    "prev_score": self.prev_score,
                },
            )

            self.logger.info(
                f"Saved new best RAL checkpoint | "
                f"score={current_score:.4f} | "
                f"cycle={self.cycle} | epoch={epoch}"
            )
    # ==========================================================
    # One AL cycle
    # ==========================================================
    def run_cycle(self):
        # Query
        policy_temp_start = getattr(self.config, "policy_temp_start", self.policy_temp)
        policy_temp_end = getattr(self.config, "policy_temp_end", self.policy_temp)

        self.policy_temp = max(
            policy_temp_end,
            policy_temp_start * (0.95 ** self.cycle)
        )
            
        new_indices, log_prob_sum, entropy, budget = self.query()
        if len(new_indices) == 0:
            self.logger.info("No samples selected this cycle.")
            self.cycle += 1
            return
        self.labeled_indices.extend(new_indices)
        new_set = set(new_indices)
        self.unlabeled_indices = [
            s for s in self.unlabeled_indices if s not in new_set
        ]

        # TrainSubset
        labeled_dataset = MixedDataset(
                self.dataset_train,
                self.dataset_pool,
                self.labeled_indices
            )
        for ep in range(self.config.epochs_per_cycle):
            epoch_start = time.time()
            train_metrics = self.main_model.train_epoch(labeled_dataset, ep, self.config.epochs_per_cycle)
            eval_metrics = self.main_model.evaluate(self.dataset_val)
            epoch_time = time.time() - epoch_start

            self._log_metrics(
                epoch=ep,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                epoch_time=epoch_time,
            )

            self.save_results()
            self._maybe_save_best_checkpoint(eval_metrics, ep)
        
        reward_metric = getattr(self.config, "reward_metric", None)
        if reward_metric is not None:
            score = eval_metrics.get(reward_metric, 0)
        else:
            score = self.get_primary_metric(self.config.task, eval_metrics)
        
        if self.prev_score is None:
            reward = 0.0
        else:
            reward = score - self.prev_score
            reward = float(np.clip(reward, -0.1, 0.1))
        self.prev_score = score 
        # Cost penalty
        if getattr(self.config, "dynamic_query_size", False):
            cost_lambda = getattr(self.config, "cost_lambda", 0.0)
            denom = self._get_cost_denominator()
            cost_penalty = cost_lambda * (budget / denom)
            reward = reward - cost_penalty
            
            self.history.setdefault("selected_budget", []).append(int(budget))
            self.history.setdefault("cost_penalty", []).append(float(cost_penalty))
            self.history.setdefault("cost_denom", []).append(int(denom))
            self.history.setdefault("cost_denom_mode", []).append(
                getattr(self.config, "cost_denom_mode", "total")
            )

            self.logger.info(
                f"Cost penalty: {cost_penalty:.6f} | "
                f"budget={budget} | denom={denom} | "
                f"mode={getattr(self.config, 'cost_denom_mode', 'total')}"
            )
        # Policy update ONLY if a query actually happened
        advantage_value = 0.0
        advantage = torch.tensor(0.0, device=self.device)

        if log_prob_sum is not None:
            advantage_value = float(reward - self.reward_baseline)

            advantage = torch.tensor(
                advantage_value,
                dtype=torch.float32,
                device=self.device
            )

            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline
                + (1 - self.baseline_momentum) * reward
            )

            loss = -(advantage * log_prob_sum) - self.entropy_beta * entropy

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()

        else:
            self.logger.info("No policy update (no query this cycle)")

        self.logger.info(
            "Reward: {:.4f} | Baseline: {:.4f} | Advantage: {:.4f}".format(
                float(reward),
                float(self.reward_baseline),
                float(advantage_value)
            )
        )
        
        self.cycle += 1

    # ==========================================================
    # Full run
    # ==========================================================
    def run(self):
        run_start_time = datetime.datetime.now()
        self.logger.info("Starting RL Active Learning")

        # Warm-up
        labeled_dataset = MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.labeled_indices
        )
        for ep in range(self.config.initial_training_epoch):
            epoch_start = time.time()
            train_metrics = self.main_model.train_epoch(labeled_dataset, ep, self.config.initial_training_epoch)

            eval_metrics = self.main_model.evaluate(self.dataset_val)
            epoch_time = time.time() - epoch_start
            self._log_metrics(
                epoch=ep,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                epoch_time=epoch_time,
            )

            self.save_results()
            self._maybe_save_best_checkpoint(eval_metrics, ep)
        self.cycle += 1

        self.prev_score = self.get_primary_metric(self.config.task, eval_metrics)
        for cycle in range(self.config.al_cycles):
            self.logger.info(f"\n=== Reinforcement AL Cycle {cycle + 1}/{self.config.al_cycles} ===")
            self.run_cycle()

        run_time = str(datetime.datetime.now() - run_start_time)
        self.logger.info(f"RL Active Learning completed in {run_time}")
        self.history["run_time"] = run_time
        system_time = str(datetime.datetime.now() - self.system_start_time)
        self.logger.info(f"Total system time: {system_time}")
        self.history["system_time"] = system_time
        self.save_results()
        return self.history


    def _log_reward(self, reward=None):
        self.history.setdefault("Reward", []).append(reward)
        self.logger.info(f"=== Reward {reward} ===")

    def _log_metrics(self, epoch, train_metrics, eval_metrics, epoch_time):

        global_epoch = epoch + self.cycle * self.config.epochs_per_cycle

        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("global_epoch", []).append(global_epoch)
        self.history.setdefault("cycle", []).append(self.cycle)
        self.history.setdefault("epoch_time", []).append(epoch_time)
        self.history.setdefault("train_loss", []).append(train_metrics["train_loss"])
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))


        # ==========================================
        # Semantic segmentation
        # ==========================================
        if self.config.task == "segmentation":

            f1 = eval_metrics.get("f1", 0)
            dice = eval_metrics.get("dice", 0)
            miou = eval_metrics.get("mean_iou", 0)

            self.history.setdefault("val_F1", []).append(f1)
            self.history.setdefault("val_dice", []).append(dice)
            self.history.setdefault("val_mean_iou", []).append(miou)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
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
                        "train_loss": train_metrics["train_loss"],
                        "val_F1": f1,
                        "val_dice": dice,
                        "val_mean_iou": miou,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )


        # ==========================================
        # Instance segmentation
        # ==========================================
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
        # ==========================================
        # Multi-label classification
        # ==========================================
        elif self.config.task == "multilabel_classification":
 
            macro_f1 = eval_metrics.get("macro_f1", 0)
            micro_f1 = eval_metrics.get("micro_f1", 0)
            map_score = eval_metrics.get("map", 0)
            hamming  = eval_metrics.get("hamming_loss", 0)
 
            self.history.setdefault("val_macro_f1", []).append(macro_f1)
            self.history.setdefault("val_micro_f1", []).append(micro_f1)
            self.history.setdefault("val_map", []).append(map_score)
            self.history.setdefault("val_hamming_loss", []).append(hamming)
 
            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Micro-F1: {micro_f1:.4f} | "
                f"mAP: {map_score:.4f} | "
                f"Hamming: {hamming:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )
 
            if self.config.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics["train_loss"],
                        "val_macro_f1": macro_f1,
                        "val_micro_f1": micro_f1,
                        "val_map": map_score,
                        "val_hamming_loss": hamming,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )

        # ==========================================
        # Single-label classification
        # ==========================================
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
                f"Loss: {train_metrics['train_loss']:.4f} | "
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
    
