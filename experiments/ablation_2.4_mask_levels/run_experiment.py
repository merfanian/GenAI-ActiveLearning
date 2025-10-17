import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import (
    data_service,
    model_service,
    fairness_service,
    llm_augmentation_service,
)
from utils.config import AUGMENTED_IMAGES_DIR
from utils.attribute_mappings import (
    ADIENCE_GENDER_MAPPING,
    ADIENCE_GENDER_TARGET,
    ANIMALS_LABEL_TARGET,
    ANIMALS_LABEL_MAPPING,
    FFHQ_GENDER_MAPPING,
    FFHQ_GENDER_TARGET
)

# --- Configuration ---
RESULTS_DIR = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/ablation_2.5_mask_levels/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
ARCHITECTURE = "resnet"
NUM_ITERATIONS = 10
AUGMENTATION_BATCH_SIZE = 10
ACCURACY_THRESHOLD = 0.95
NUM_RUNS = 3
MASK_LEVELS = ["accurate", "moderate", "imprecise"]

BENCHMARKS = [
    # {
    #     "name": "adience",
    #     "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/",
    #     "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/metadata_with_ethnicity_1500.csv",
    #     "target_attribute": "label",
    #     "fairness_attribute": "ethnicity",
    #     "attribute_mapping": ADIENCE_GENDER_MAPPING,
    #     "target_label_mapping": ADIENCE_GENDER_TARGET,
    # },
    # {
    #     "name": "fruits",
    #     "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/fruits/dataset/",
    #     "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/fruits/filtered.csv",
    #     "target_attribute": "label",
    #     "fairness_attribute": "label",
    #     "attribute_mapping": FRUITS_LABEL_MAPPING,
    #     "target_label_mapping": FRUITS_LABEL_TARGET,
    # },
    {
        "name": "animals",
        "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/animals",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/animals/metadata_4000.csv",
        "target_attribute": "label",
        "fairness_attribute": "label",
        "attribute_mapping": ANIMALS_LABEL_MAPPING,
        "target_label_mapping": ANIMALS_LABEL_TARGET,
    },
    # {
    #     "name": "ffhq",
    #     "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/ffhq/images/",
    #     "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/ffhq/metadata_with_ethnicity_2000.csv",
    #     "target_attribute": "gender",
    #     "fairness_attribute": "ethnicity",
    #     "attribute_mapping": FFHQ_GENDER_MAPPING,
    #     "target_label_mapping": FFHQ_GENDER_TARGET,
    # },
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_mask_level_experiment(config: dict, run_number: int, logger, baseline_model_path: str, mask_level: str):
    logger.info(f"--- Running Scenario: Mask Level '{mask_level}' ---")

    progression = []
    scenario_model_path = baseline_model_path
    model_service.set_current_model_path(scenario_model_path)

    benchmark_augmented_dir = (
            AUGMENTED_IMAGES_DIR / f"{config['name']}_run_{run_number}_{mask_level}"
    )
    if benchmark_augmented_dir.exists():
        shutil.rmtree(benchmark_augmented_dir)
    benchmark_augmented_dir.mkdir(parents=True)

    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    total_generated = 0
    total_rejected = 0

    for i in range(NUM_ITERATIONS):
        gp_raw = fairness_service.calculate_group_performances(
            scenario_model_path,
            test_df,
            test_paths,
            test_labels,
            config["fairness_attribute"],
        )
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        worst_acc_before = worst_raw.get("accuracy", 0.0)
        progression.append({"iteration": i, "worst_group_accuracy": worst_acc_before})

        if worst_acc_before >= ACCURACY_THRESHOLD:
            logger.info("Accuracy threshold met. Stopping.")
            break

        # Track rejections in this batch
        generated_this_batch = 0

        # We need to over-generate to account for rejections
        num_to_generate = AUGMENTATION_BATCH_SIZE
        generated, num_rejected = llm_augmentation_service.generate_image_and_get_label(
            worst_raw["attributes"],
            num_images=num_to_generate,
            augmented_data_dir=benchmark_augmented_dir,
            attribute_mapping=config["attribute_mapping"],
            target_label_mapping=config["target_label_mapping"],
            validate_quality=True,
            mask_level=mask_level,
            sampling_strategy="random",
            exploration_mode="exploitation_only"
        )

        total_generated += len(generated) + num_rejected
        total_rejected += num_rejected

        for item in generated:
            data_service.add_augmented_data(
                item["filename"],
                item["attributes_used"],
                item["llm_acquired_label"],
                augmented_data_dir=benchmark_augmented_dir,
            )
            generated_this_batch += 1

        if generated_this_batch == 0:
            logger.warning("No images were generated in this batch. Skipping training.")
            continue

        train_paths_aug, train_labels_aug, _ = (
            data_service.get_train_val_image_paths_and_labels(
                include_augmented=True, augmented_data_dir=benchmark_augmented_dir
            )
        )
        new_model_path = model_service.train_model(
            train_paths_aug,
            train_labels_aug,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_run_{run_number}_{mask_level}_iter_{i + 1}.pth",
        )

        new_gp_raw = fairness_service.calculate_group_performances(
            new_model_path,
            test_df,
            test_paths,
            test_labels,
            config["fairness_attribute"],
        )
        new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
        worst_acc_after = new_worst_raw.get("accuracy", 0.0)

        if worst_acc_after < worst_acc_before:
            data_service.remove_last_augmented_batch(
                generated_this_batch, augmented_data_dir=benchmark_augmented_dir
            )
        else:
            scenario_model_path = new_model_path
            model_service.set_current_model_path(scenario_model_path)

    final_gp_raw = fairness_service.calculate_group_performances(
        scenario_model_path,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    final_worst_raw = fairness_service.find_worst_performing_group(final_gp_raw)
    final_overall = fairness_service.calculate_overall_accuracy(
        scenario_model_path, test_paths, test_labels
    )

    rejection_rate = total_rejected / total_generated if total_generated > 0 else 0

    return final_worst_raw.get("accuracy", 0.0), final_overall, rejection_rate, progression


def run_benchmark(config: dict, run_number: int):
    benchmark_name = config["name"]
    logger = logging.getLogger(f"{benchmark_name}_run_{run_number}")

    message = data_service.load_and_validate_dataset(
        config["image_dir_path"],
        config["metadata_csv_path"],
        config["target_attribute"],
        config["fairness_attribute"],
    )
    if "error" in message:
        logger.error(f"Data loading/validation failed: {message['error']}")
        return

    train_paths, train_labels, _ = data_service.get_train_val_image_paths_and_labels(
        include_augmented=False
    )
    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    baseline_model_path = model_service.train_model(
        train_paths,
        train_labels,
        architecture=ARCHITECTURE,
        updated_model_path=f"{benchmark_name}_run_{run_number}_baseline.pth",
    )

    results = {
        "benchmark_name": benchmark_name,
        "run": run_number,
        "mask_levels": {}
    }

    for mask_level in MASK_LEVELS:
        final_worst, final_overall, rejection_rate, progression = run_mask_level_experiment(
            config, run_number, logger, baseline_model_path, mask_level
        )
        results["mask_levels"][mask_level] = {
            "worst_group_accuracy": final_worst,
            "overall_accuracy": final_overall,
            "rejection_rate": rejection_rate,
            "progression": progression
        }

    results_path = RESULTS_DIR / f"{benchmark_name}_run_{run_number}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    for run in range(1, NUM_RUNS + 1):
        for benchmark_config in BENCHMARKS:
            run_benchmark(benchmark_config, run)


if __name__ == "__main__":
    main()
