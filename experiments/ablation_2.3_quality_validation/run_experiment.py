import sys
import os
import json
import logging
import shutil
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import (
    data_service,
    model_service,
    fairness_service,
    llm_augmentation_service,
)
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR
from utils.attribute_mappings import ADIENCE_GENDER_MAPPING, ADIENCE_GENDER_TARGET, ANIMALS_LABEL_MAPPING, \
    ANIMALS_LABEL_TARGET, FFHQ_GENDER_MAPPING, FFHQ_GENDER_TARGET, FRUITS_LABEL_MAPPING, FRUITS_LABEL_TARGET

# --- Configuration ---
RESULTS_DIR = Path(
    "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/ablation_2.3_quality_validation/results"
)
RESULTS_DIR.mkdir(exist_ok=True)
ARCHITECTURE = "resnet"
NUM_ITERATIONS = 10
AUGMENTATION_BATCH_SIZE = 10
ACCURACY_THRESHOLD = 0.95
NUM_RUNS = 3

BENCHMARKS = [
    {
        "name": "adience_quality_ablation",
        "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/metadata_with_ethnicity_1500.csv",
        "target_attribute": "label",
        "fairness_attribute": "ethnicity",
        "attribute_mapping": ADIENCE_GENDER_MAPPING,
        "target_label_mapping": ADIENCE_GENDER_TARGET,
    },
    # {
    #     "name": "animals",
    #     "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/animals",
    #     "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/animals/metadata_4000.csv",
    #     "target_attribute": "label",
    #     "fairness_attribute": "label",
    #     "attribute_mapping": ANIMALS_LABEL_MAPPING,
    #     "target_label_mapping": ANIMALS_LABEL_TARGET,
    # },
    # {
    #     "name": "ffhq",
    #     "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/ffhq/images/",
    #     "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/ffhq/metadata_with_ethnicity_2000.csv",
    #     "target_attribute": "gender",
    #     "fairness_attribute": "ethnicity",
    #     "attribute_mapping": FFHQ_GENDER_MAPPING,
    #     "target_label_mapping": FFHQ_GENDER_TARGET,
    # },
    # {
    #     "name": "fruits",
    #     "image_dir_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/fruits/dataset/",
    #     "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/fruits/filtered.csv",
    #     "target_attribute": "label",
    #     "fairness_attribute": "label",
    #     "attribute_mapping": FRUITS_LABEL_MAPPING,
    #     "target_label_mapping": FRUITS_LABEL_TARGET,
    # }
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_scenario(
        config: dict,
        run_number: int,
        logger: logging.Logger,
        baseline_model_path: str,
        validate_quality: bool,
):
    """
    Runs a full augmentation scenario for a given configuration for a single run.
    """
    scenario_name = "with_quality_filter" if validate_quality else "without_quality_filter"
    logger.info(f"--- Running Scenario: {scenario_name} ---")

    progression = []
    scenario_model_path = baseline_model_path
    model_service.set_current_model_path(scenario_model_path)

    scenario_augmented_dir = (
            AUGMENTED_IMAGES_DIR / f"{config['name']}_{scenario_name}_run_{run_number}"
    )
    if scenario_augmented_dir.exists():
        shutil.rmtree(scenario_augmented_dir)
    scenario_augmented_dir.mkdir(parents=True)

    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    for i in range(NUM_ITERATIONS):
        logger.info(f"--- Iteration {i + 1}/{NUM_ITERATIONS} for {scenario_name} ---")

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
            logger.info("Accuracy threshold met. Stopping scenario.")
            break

        generated, _ = llm_augmentation_service.generate_image_and_get_label(
            worst_raw["attributes"],
            AUGMENTATION_BATCH_SIZE,
            augmented_data_dir=scenario_augmented_dir,
            attribute_mapping=config["attribute_mapping"],
            target_label_mapping=config["target_label_mapping"],
            validate_quality=validate_quality,
            exploration_mode="exploitation_only",
        )
        for item in generated:
            data_service.add_augmented_data(
                item["filename"],
                item["attributes_used"],
                item["llm_acquired_label"],
                augmented_data_dir=scenario_augmented_dir,
            )

        train_paths_aug, train_labels_aug, _ = (
            data_service.get_train_val_image_paths_and_labels(
                include_augmented=True, augmented_data_dir=scenario_augmented_dir
            )
        )
        new_model_path = model_service.train_model(
            train_paths_aug,
            train_labels_aug,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_{scenario_name}_run_{run_number}_iter_{i + 1}.pth",
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
            logger.warning(
                f"Performance degraded after augmentation (before: {worst_acc_before:.4f}, after: {worst_acc_after:.4f}). "
                f"Discarding model and rolling back data."
            )
            data_service.remove_last_augmented_batch(
                len(generated), augmented_data_dir=scenario_augmented_dir
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
    final_worst_accuracy = final_worst_raw.get("accuracy", 0.0)
    progression.append({"iteration": NUM_ITERATIONS, "worst_group_accuracy": final_worst_accuracy})

    final_overall_accuracy = fairness_service.calculate_overall_accuracy(
        scenario_model_path, test_paths, test_labels
    )

    return final_worst_accuracy, final_overall_accuracy, progression


def run_benchmark(config: dict, run_number: int):
    benchmark_name = config["name"]
    logger = logging.getLogger(f"{benchmark_name}_run_{run_number}")
    log_file = RESULTS_DIR / f"{benchmark_name}_run_{run_number}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    logger.info(f"--- Starting Benchmark: {benchmark_name}, Run: {run_number} ---")

    message = data_service.load_and_validate_dataset(
        config["image_dir_path"],
        config["metadata_csv_path"],
        config["target_attribute"],
        config["fairness_attribute"],
    )
    if "error" in message:
        logger.error(f"Data loading failed: {message['error']}")
        return

    train_paths, train_labels, _ = data_service.get_train_val_image_paths_and_labels(
        include_augmented=False
    )
    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    baseline_model_path = TRAINED_MODELS_DIR / f"{benchmark_name}_run_{run_number}_baseline.pth"
    if not baseline_model_path.exists():
        logger.info("Training initial baseline model...")
        model_service.train_model(
            train_paths,
            train_labels,
            architecture=ARCHITECTURE,
            updated_model_path=str(baseline_model_path.name),
        )
    else:
        logger.info(f"Using existing baseline model: {baseline_model_path}")

    gp_raw = fairness_service.calculate_group_performances(
        str(baseline_model_path),
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    initial_worst = worst_raw.get("accuracy", 0.0)
    initial_overall = fairness_service.calculate_overall_accuracy(
        str(baseline_model_path), test_paths, test_labels
    )

    # --- Run Scenarios ---
    (
        final_worst_with_q,
        final_overall_with_q,
        progression_with_q,
    ) = run_scenario(config, run_number, logger, str(baseline_model_path), validate_quality=True)

    (
        final_worst_without_q,
        final_overall_without_q,
        progression_without_q,
    ) = run_scenario(config, run_number, logger, str(baseline_model_path), validate_quality=False)

    # --- Save Results ---
    results = {
        "benchmark_name": benchmark_name,
        "run": run_number,
        "initial_overall_accuracy": initial_overall,
        "initial_worst_group_accuracy": initial_worst,
        "with_quality_filter": {
            "final_overall_accuracy": final_overall_with_q,
            "final_worst_group_accuracy": final_worst_with_q,
            "progression": progression_with_q,
        },
        "without_quality_filter": {
            "final_overall_accuracy": final_overall_without_q,
            "final_worst_group_accuracy": final_worst_without_q,
            "progression": progression_without_q,
        },
    }

    results_path = RESULTS_DIR / f"{benchmark_name}_run_{run_number}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Benchmark run complete. Results saved to {results_path}")


def main():
    """
    Main function to set up and run the ablation study.
    """
    for run in range(1, NUM_RUNS + 1):
        for benchmark_config in BENCHMARKS:
            run_benchmark(benchmark_config, run)

    logging.info("--- All runs for all benchmarks complete ---")


if __name__ == "__main__":
    main()
