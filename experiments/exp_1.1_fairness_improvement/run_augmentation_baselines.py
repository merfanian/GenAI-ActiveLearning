import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import (
    data_service,
    model_service,
    fairness_service,
)
from utils.attribute_mappings import (
    ADIENCE_GENDER_MAPPING,
    ADIENCE_GENDER_TARGET,
    UTKFACE_RACE_MAPPING,
    UTKFACE_RACE_TARGET,
    FRUITS_LABEL_MAPPING,
    FRUITS_LABEL_TARGET,
    FFHQ_GENDER_MAPPING,
    FFHQ_GENDER_TARGET,
    ANIMALS_LABEL_MAPPING,
    ANIMALS_LABEL_TARGET,
)

# --- Configuration ---
RESULTS_DIR = Path(
    "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/exp_1.1_fairness_improvement/results"
)
RESULTS_DIR.mkdir(exist_ok=True)
ARCHITECTURE = "resnet"
NUM_RUNS = 3

BENCHMARKS = [
    {
        "name": "adience",
        "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/metadata_with_ethnicity_1500.csv",
        "target_attribute": "label",
        "fairness_attribute": "ethnicity",
    },
    {
        "name": "fruits",
        "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/fruits/dataset/",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/fruits/filtered.csv",
        "target_attribute": "label",
        "fairness_attribute": "label",
    },
    {
        "name": "animals",
        "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/animals",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/animals/metadata_4000.csv",
        "target_attribute": "label",
        "fairness_attribute": "label",
    },
    {
        "name": "ffhq",
        "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/ffhq/images/",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/ffhq/metadata_with_ethnicity_2000.csv",
        "target_attribute": "gender",
        "fairness_attribute": "ethnicity",
    },
]

AUGMENTATION_BASELINES = ["mixup", "cutout", "cutmix", "groupdro"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_augmentation_benchmark(config: dict, run_number: int, aug_method: str):
    benchmark_name = config["name"]
    logger = logging.getLogger(f"{benchmark_name}_run_{run_number}_{aug_method}")

    data_service.load_and_validate_dataset(
        config["image_dir_path"],
        config["metadata_csv_path"],
        config["target_attribute"],
        config["fairness_attribute"],
    )

    train_paths, train_labels, train_groups = data_service.get_train_val_image_paths_and_labels(
        include_augmented=False, fairness_attribute=config["fairness_attribute"]
    )
    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    # Train model with the specified augmentation
    logger.info(f"--- Training {aug_method} model for {benchmark_name} (Run {run_number}) ---")
    if aug_method == 'groupdro':
        model_path = model_service.train_model(
            train_paths,
            train_labels,
            groups=train_groups,
            architecture=ARCHITECTURE,
            updated_model_path=f"{benchmark_name}_run_{run_number}_{aug_method}.pth",
            use_group_dro=True,
        )
    else:
        model_path = model_service.train_model(
            train_paths,
            train_labels,
            architecture=ARCHITECTURE,
            updated_model_path=f"{benchmark_name}_run_{run_number}_{aug_method}.pth",
            augmentation_method=aug_method,
        )

    # Evaluate the model
    gp_raw = fairness_service.calculate_group_performances(
        model_path,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    worst_accuracy = worst_raw.get("accuracy", 0.0)
    overall_accuracy = fairness_service.calculate_overall_accuracy(
        model_path, test_paths, test_labels
    )

    results = {
        "benchmark_name": benchmark_name,
        "run": run_number,
        "augmentation_method": aug_method,
        f"{aug_method}_overall": overall_accuracy,
        f"{aug_method}_worst": worst_accuracy,
    }

    results_path = RESULTS_DIR / f"{benchmark_name}_run_{run_number}_{aug_method}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved results for {aug_method} to {results_path}")


def main():
    for run in range(1, NUM_RUNS + 1):
        for benchmark_config in BENCHMARKS:
            for aug_method in AUGMENTATION_BASELINES:
                run_augmentation_benchmark(benchmark_config, run, aug_method)


if __name__ == "__main__":
    main()
