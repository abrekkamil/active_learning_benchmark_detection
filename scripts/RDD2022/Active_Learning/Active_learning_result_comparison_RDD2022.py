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


config_path = SCRIPT_DIR / "Active_Learning_detection_rdd2022.yaml"


initial_labeled_sizes = [0.05]
query_sizes = [1000]

cold_start_strategies = [
    "random",
    # "diversity",
]

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

                config.task = "detection"
                config.model_name = "fasterrcnn"
                config.use_wandb = False

                config.experiment_name = (
                    f"AL_fasterrcnn_RDD2022_"
                    f"{cold_start}_{query_strategy}_"
                    f"i{int(initial_labeled_size * 100)}_q{query_size}_"
                    f"seed{config.seed}"
                )

                print("=" * 80)
                print("Starting RDD2022 detection Active Learning experiment")
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