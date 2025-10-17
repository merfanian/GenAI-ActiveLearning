import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import logging
from pathlib import Path
import numpy as np

from dotenv import load_dotenv

load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import (
    data_service,
    model_service,
    fairness_service,
)

AUGMENTED_IMAGES_DIR = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/exp_1.1_fairness_improvement/augmented_images/")
# --- Configuration ---
RESULTS_DIR = Path(
    "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/exp_1.2_generalizability/results"
)
RESULTS_DIR.mkdir(exist_ok=True)
ARCHITECTURES = ["resnet", "mobilenet", "densenet"]
NUM_RUNS = 2

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_dataset(
    train_paths: list[str],
    train_labels: list[str],
    architecture: str,
    test_df: pd.DataFrame,
    test_paths: list[str],
    test_labels: list[str],
    fairness_attribute: str,
    model_name: str,
    logger,
) -> dict:
    """Trains a model on a given dataset and evaluates its fairness and overall accuracy."""
    logger.info(f"Training model '{model_name}' with architecture '{architecture}'...")
    
    model_path = model_service.train_model(
        train_paths,
        train_labels,
        architecture=architecture,
        updated_model_path=f"{model_name}.pth",
    )
    
    logger.info(f"Evaluating model '{model_name}'...")
    gp_raw = fairness_service.calculate_group_performances(
        model_path, test_df, test_paths, test_labels, fairness_attribute
    )
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    overall_acc = fairness_service.calculate_overall_accuracy(model_path, test_paths, test_labels)
    
    results = {
        "worst_group_accuracy": worst_raw.get("accuracy", 0.0),
        "overall_accuracy": overall_acc,
    }
    
    logger.info(f"--- Results for {model_name} ({architecture}) ---")
    logger.info(f"  Worst-Group Accuracy: {results['worst_group_accuracy']:.4f}")
    logger.info(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info("-" * (30 + len(model_name) + len(architecture)))

    return results

def run_generalizability_for_benchmark(config: dict, run_number: int):
    benchmark_name = config["name"]
    logger = logging.getLogger(f"{benchmark_name}_run_{run_number}")
    logger.info(f"========== Starting Generalizability Benchmark: {benchmark_name}, Run: {run_number} ==========")

    message = data_service.load_and_validate_dataset(
        config["image_dir_path"],
        config["metadata_csv_path"],
        config["target_attribute"],
        config["fairness_attribute"],
    )
    if "error" in message:
        logger.error(f"Data loading/validation failed: {message['error']}")
        return

    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    all_eval_results = {}

    # --- 1. Evaluate Original Data ---
    logger.info("--- Phase 1: Evaluating Original (Un-Augmented) Data ---")
    original_train_paths, original_train_labels = data_service.get_train_val_image_paths_and_labels(include_augmented=False)
    all_eval_results["original"] = {}
    for arch in ARCHITECTURES:
        model_name = f"gen_{benchmark_name}_run_{run_number}_original"
        results = evaluate_dataset(
            original_train_paths, original_train_labels, arch, test_df, test_paths, test_labels,
            config["fairness_attribute"], model_name, logger
        )
        all_eval_results["original"][arch] = results

    # --- 2. Find and Evaluate Augmented Datasets ---
    logger.info("--- Phase 2: Evaluating Augmented Data ---")
    augmented_dirs = list(AUGMENTED_IMAGES_DIR.glob(f"{benchmark_name}_run_{run_number}*"))
    if not augmented_dirs:
        logger.warning(f"No augmented data directories found for {benchmark_name} run {run_number}. Skipping augmented evaluation.")
    
    for aug_dir in augmented_dirs:
        dir_name = aug_dir.name
        if dir_name.endswith("_reweigh"):
            method_name = "our_method_reweigh"
        elif dir_name.endswith("_smote"):
            method_name = "our_method_smote"
        else:
            method_name = "our_method"
        
        logger.info(f"--- Evaluating augmentation method: {method_name} from {aug_dir} ---")
        all_eval_results[method_name] = {}
        
        aug_paths, aug_labels = data_service.get_train_val_image_paths_and_labels(
            include_augmented=True, augmented_data_dir=aug_dir
        )
        
        for arch in ARCHITECTURES:
            model_name = f"gen_{benchmark_name}_run_{run_number}_{method_name}"
            results = evaluate_dataset(
                aug_paths, aug_labels, arch, test_df, test_paths, test_labels,
                config["fairness_attribute"], model_name, logger
            )
            all_eval_results[method_name][arch] = results

    # --- 3. Determine Best Method ---
    logger.info("--- Phase 3: Determining Best Augmentation Method ---")
    method_scores = {}
    aug_methods = [k for k in all_eval_results if k != "original"]

    if not aug_methods:
        logger.error("No augmented methods were evaluated. Cannot determine the best method.")
        best_method_name = "none"
    else:
        for method in aug_methods:
            accuracies = [all_eval_results[method][arch]["worst_group_accuracy"] for arch in ARCHITECTURES]
            method_scores[method] = np.mean(accuracies)
            logger.info(f"  Average Worst-Group Accuracy for '{method}': {method_scores[method]:.4f}")

        best_method_name = max(method_scores, key=method_scores.get)
        logger.info(f"--- Best augmentation method for {benchmark_name} run {run_number}: {best_method_name} ---")

    # --- 4. Save Final Results ---
    final_results = {
        "benchmark_name": benchmark_name,
        "run": run_number,
        "best_augmentation_method": best_method_name,
        "architectures": {}
    }

    for arch in ARCHITECTURES:
        original_results = all_eval_results.get("original", {}).get(arch, {"overall_accuracy": 0.0, "worst_group_accuracy": 0.0})
        best_aug_results = all_eval_results.get(best_method_name, {}).get(arch, {"overall_accuracy": 0.0, "worst_group_accuracy": 0.0})

        final_results["architectures"][arch] = {
            "initial_overall": original_results["overall_accuracy"],
            "initial_worst": original_results["worst_group_accuracy"],
            "augmented_overall": best_aug_results["overall_accuracy"],
            "augmented_worst": best_aug_results["worst_group_accuracy"],
        }
    
    results_path = RESULTS_DIR / f"{benchmark_name}_run_{run_number}_generalizability_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    logger.info(f"Final generalizability results saved to {results_path}")
    logger.info(f"========== Finished Generalizability Benchmark: {benchmark_name}, Run: {run_number} ==========\n")


def main():
    for run in range(1, NUM_RUNS + 1):
        for benchmark_config in BENCHMARKS:
            run_generalizability_for_benchmark(benchmark_config, run)


if __name__ == "__main__":
    main()