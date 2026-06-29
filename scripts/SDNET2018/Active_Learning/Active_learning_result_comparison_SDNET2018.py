import sys
from pathlib import Path
import gc
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ActiveLearningConfig
from src.active_learning import ActiveLearningSystem


def empty():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


config_path = SCRIPT_DIR / "Active_Learning_classification_sdnet2018.yaml"

initial_labeled_sizes = [0.05]
query_sizes = [1000]

cold_start_strategies = [    
    "diversity",
]
# cold_start_strategies = [
#     "random",
#     "simple_diversity",
#     "entropy_based_uncertainty",
#     "weak_supervision",
#     "self_supervised",
#     "diversity",
# ]

query_strategies = [
    "uncertainty",
]

for initial_labeled_size in initial_labeled_sizes:
    for query_size in query_sizes:
        for cold_start in cold_start_strategies:
            for query_strategy in query_strategies:

                config = ActiveLearningConfig.from_yaml(str(config_path))

                config.initial_labeled = initial_labeled_size
                config.query_size = query_size
                config.cold_start_strategy = cold_start
                config.query_strategy = query_strategy

                config.task = "binary_classification"
                config.model_name = "resnet50_classification"
                config.use_wandb = False

                config.al_cycles = 10
                config.initial_training_epoch = 15
                config.epochs_per_cycle = 10

                config.experiment_name = (
                    f"AL_resnet50_SDNET2018_c10_e10_ie15"
                    f"{cold_start}_{query_strategy}_"
                    f"i{int(initial_labeled_size * 100)}_q{query_size}"
                )

                print("=" * 80)
                print("Starting Active Learning experiment")
                print("Experiment:", config.experiment_name)
                print("Cold start:", config.cold_start_strategy)
                print("Query strategy:", config.query_strategy)
                print("Initial labeled:", config.initial_labeled)
                print("Query size:", config.query_size)
                print("=" * 80)

                al_system = ActiveLearningSystem(config)
                history = al_system.run()

                al_system.save_results()
                al_system.plot_results()

                print("Finished:", config.experiment_name)
                print(history)

                del al_system
                empty()