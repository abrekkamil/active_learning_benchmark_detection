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


config_path = SCRIPT_DIR / "dynamic_rl_active_learning_sdnet2018.yaml"

# Start small first. Add more cold starts after this works.
initial_labeled_sizes = [0.01, 0.05]

# query_size is still used as the reference/default size.
# The dynamic policy chooses from budget_options in the YAML.
query_sizes = [0]


cold_start_strategies = [
    "self_supervised",
    "random",
    "weak_supervision",
]

# cold_start_strategies = [
#     "self_supervised",
#     "random",
#     "simple_diversity",
#     "entropy_based_uncertainty",
#     "weak_supervision",
#     "diversity",
# ]

for initial_labeled_size in initial_labeled_sizes:
    for query_size in query_sizes:
        for cold_start in cold_start_strategies:

            experiment_parameters = (
                f"i_{int(initial_labeled_size * 100)}pct_"
                f"dynamic_q"
            )

            config = ActiveLearningConfig.from_yaml(str(config_path))

            config.initial_labeled = initial_labeled_size
            config.query_size = query_size
            config.cold_start_strategy = cold_start
            config.query_strategy = "rl_policy"

            config.use_wandb = False
            config.pool = False

            config.model_name = "resnet50_classification"
            config.task = "binary_classification"

            config.al_cycles = 30
            config.initial_training_epoch = 15
            config.epochs_per_cycle = 5
            config.oracle_epochs = 25

            # Dynamic query-size settings
            config.dynamic_query_size = True
            config.budget_options = [125, 250, 500, 750]
            config.cost_lambda = 0.001

            config.experiment_name = (
                f"Dynamic_RAL_resnet50_SDNET2018_c30_e5_ie15"
                f"{experiment_parameters}"
                f"_cost{config.cost_lambda}"
            )

            print("=" * 80)
            print("Starting Dynamic Reinforcement Active Learning experiment")
            print("Dataset:", config.dataset)
            print("Data dir:", config.data_dir)
            print("Experiment:", config.experiment_name)
            print("Cold start:", config.cold_start_strategy)
            print("Query strategy:", config.query_strategy)
            print("Initial labeled:", config.initial_labeled)
            print("Reference query size:", config.query_size)
            print("Dynamic query size:", config.dynamic_query_size)
            print("Budget options:", config.budget_options)
            print("Cost lambda:", config.cost_lambda)
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