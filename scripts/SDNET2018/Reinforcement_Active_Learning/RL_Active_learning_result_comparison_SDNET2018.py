import sys
from pathlib import Path
import gc
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)

from config.config import ActiveLearningConfig
from src.rl_active_learning import ActiveLearningSystemRL


def empty():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


config_path = SCRIPT_DIR / "rl_active_learning_sdnet2018.yaml"

# =========================
# RAL experiment settings
# =========================

initial_labeled_sizes = [0.1, 0.15, 0.2]
query_sizes = [500, 1000]
cold_start_strategies = [
    "random",
    "self_supervised",
]
# Start with these stable cold-start strategies.
# Add "self_supervised" later only if it works well for classification.
# cold_start_strategies = [
#     "random",
#     "simple_diversity",
#     "entropy_based_uncertainty",
#     "weak_supervision",
#     "self_supervised",
#     "diversity",
# ]

for initial_labeled_size in initial_labeled_sizes:
    for query_size in query_sizes:
        for cold_start in cold_start_strategies:

            experiment_parameters = (
                f"i_{int(initial_labeled_size * 100)}pct_"
                f"q_{query_size}"
            )

            config = ActiveLearningConfig.from_yaml(str(config_path))

            config.initial_labeled = initial_labeled_size
            config.query_size = query_size
            config.cold_start_strategy = cold_start
            config.query_strategy = "rl_policy"

            config.use_wandb = False
            config.dynamic_query_size = False
            config.pool = False

            config.model_name = "resnet50_classification"
            config.task = "binary_classification"

            config.al_cycles = 10
            config.initial_training_epoch = 15
            config.epochs_per_cycle = 10
            config.oracle_epochs = 25

            config.experiment_name = (
                f"RAL_resnet50_SDNET2018_"
                f"{cold_start}_rl_policy_"
                f"{experiment_parameters}"
            )

            print("=" * 80)
            print("Starting Reinforcement Active Learning experiment")
            print("Dataset:", config.dataset)
            print("Data dir:", config.data_dir)
            print("Experiment:", config.experiment_name)
            print("Cold start:", config.cold_start_strategy)
            print("Query strategy:", config.query_strategy)
            print("Initial labeled:", config.initial_labeled)
            print("Query size:", config.query_size)
            print("AL cycles:", config.al_cycles)
            print("Initial training epochs:", config.initial_training_epoch)
            print("Epochs per cycle:", config.epochs_per_cycle)
            print("Oracle epochs:", config.oracle_epochs)
            print("=" * 80)

            ral_system = ActiveLearningSystemRL(config)
            history_RAL = ral_system.run()

            ral_system.save_results()

            print("Finished:", config.experiment_name)
            print(history_RAL)

            del ral_system
            empty()