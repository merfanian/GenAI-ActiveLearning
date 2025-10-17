import json
import logging
import shutil
from pathlib import Path

import numpy as np
from services import (
    data_service,
    model_service,
    fairness_service,
    llm_augmentation_service,
)
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR
from utils.logging_config import LOGGING_CONFIG

RESULTS_DIR = Path("ablation_results")
RESULTS_DIR.mkdir(exist_ok=True)


# --- User-configurable parameters ---
IMAGE_DIR_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/"
METADATA_CSV_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/adience_sample_1000.csv"
TARGET_ATTRIBUTE = "label"
FAIRNESS_ATTRIBUTE = "label"
INITIAL_MODEL_NAME = "model.pth"
ARCHITECTURE = "resnet"
NUM_ITERATIONS = 10
AUGMENTATION_BATCH_SIZE = 10
ACCURACY_THRESHOLD = 0.95


def run_ablation_study():
    # --- 1. Setup Dataset ---
    logging.info("Setting up the dataset...")
    data_service.load_and_validate_dataset(
        IMAGE_DIR_PATH, METADATA_CSV_PATH, TARGET_ATTRIBUTE, FAIRNESS_ATTRIBUTE
    )

    # --- 2. Train Initial Model ---
    logging.info("Checking for an existing initial model...")
    initial_model_path = TRAINED_MODELS_DIR / INITIAL_MODEL_NAME
    if initial_model_path.exists():
        logging.info(
            f"Found existing model at {initial_model_path}, skipping training."
        )
    else:
        logging.info("No existing model found, training a new one...")
        image_paths, labels = data_service.get_train_val_image_paths_and_labels(
            include_augmented=False
        )
        initial_model_path = model_service.train_model(
            image_paths, labels, architecture=ARCHITECTURE, updated_model_path=INITIAL_MODEL_NAME
        )
        logging.info(f"Initial model trained and saved to {initial_model_path}")

    model_service.set_current_model_path(str(initial_model_path))

    # --- 3. Define Ablation Scenarios ---
    scenarios = {
        "exploitation_only": {"exploration_probability": 0.0},
        "exploration_only": {"exploration_probability": 1.0},
        "balanced_sigmoid": {"exploration_steepness": 10.0},
    }

    # --- 4. Run Ablation Study ---
    for scenario_name, params in scenarios.items():
        logging.info(f"--- Running Scenario: {scenario_name} ---")

        # Reset to the initial model for each scenario
        scenario_model_path = f"{TRAINED_MODELS_DIR}/{scenario_name}_model.pth"
        shutil.copy(initial_model_path, scenario_model_path)
        model_service.set_current_model_path(scenario_model_path)

        scenario_results = []

        for i in range(NUM_ITERATIONS):
            logging.info(
                f"--- Iteration {i + 1}/{NUM_ITERATIONS} for {scenario_name} ---"
            )

            # Evaluate fairness
            df = data_service.get_test_metadata_df()
            image_paths, labels = data_service.get_test_image_paths_and_labels()
            gp_raw = fairness_service.calculate_group_performances(
                scenario_model_path, df, image_paths, labels, FAIRNESS_ATTRIBUTE
            )
            worst_raw = fairness_service.find_worst_performing_group(gp_raw)
            worst_acc = worst_raw.get("accuracy", 0.0)

            if worst_acc >= ACCURACY_THRESHOLD:
                logging.info("Accuracy threshold met. Stopping scenario.")
                break

            # Augment
            if "exploration_probability" in params:
                use_guide_image = None
                if params["exploration_probability"] == 0.0:
                    use_guide_image = True
                elif params["exploration_probability"] == 1.0:
                    use_guide_image = False
                generated = llm_augmentation_service.generate_image_and_get_label(
                    worst_raw["attributes"], AUGMENTATION_BATCH_SIZE, use_guide_image=use_guide_image
                )
            else:
                generated = llm_augmentation_service.generate_image_and_get_label(
                    worst_raw["attributes"],
                    AUGMENTATION_BATCH_SIZE,
                    exploration_steepness=params["exploration_steepness"],
                )

            for item in generated:
                data_service.add_augmented_data(
                    item["filename"],
                    item["attributes_used"],
                    item["llm_acquired_label"],
                )

            # Retrain
            all_image_paths, all_labels = (
                data_service.get_train_val_image_paths_and_labels(
                    include_augmented=True
                )
            )
            new_model_path = model_service.train_model(
                all_image_paths,
                all_labels,
                existing_model_path=scenario_model_path,
                updated_model_path=f"{scenario_name}_model_iter_{i + 1}.pth",
            )

            # Evaluate the new model
            new_gp_raw = fairness_service.calculate_group_performances(
                new_model_path, df, image_paths, labels, FAIRNESS_ATTRIBUTE
            )
            new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
            new_worst_acc = new_worst_raw.get("accuracy", 0.0)

            # Check for performance degradation
            if new_worst_acc < worst_acc:
                logging.warning(
                    f"Performance degraded after augmentation (before: {worst_acc:.4f}, after: {new_worst_acc:.4f}). "
                    f"Discarding model and rolling back data."
                )
                data_service.remove_last_augmented_batch(len(generated))
                # The scenario_model_path remains the same, and we proceed to the next iteration.
                new_worst_raw = worst_raw  # Keep the old results for logging
            else:
                model_service.set_current_model_path(new_model_path)
                scenario_model_path = new_model_path

            # Log results
            overall_accuracy = fairness_service.calculate_overall_accuracy(
                scenario_model_path, image_paths, labels
            )
            logging.info(
                f"Scenario: {scenario_name}, Iteration: {i + 1}, "
                f"Worst Group Accuracy: {new_worst_raw.get('accuracy', 0.0):.4f}, "
                f"Overall Accuracy: {overall_accuracy:.4f}"
            )

            scenario_results.append(
                {
                    "iteration": i + 1,
                    "worst_group_before_aug": worst_raw,
                    "worst_group_after_aug": new_worst_raw,
                    "overall_accuracy": overall_accuracy,
                    "augmented_images_generated": len(generated),
                    "model_path": scenario_model_path,  # Log the model that is kept
                }
            )

        # --- Calculate Price of Fairness, Save and Cleanup after scenario ---
        logging.info(
            f"--- Calculating Price of Fairness for scenario: {scenario_name} ---"
        )
        initial_test_image_paths, initial_test_labels = (
            data_service.get_test_image_paths_and_labels()
        )
        initial_accuracy = fairness_service.calculate_overall_accuracy(
            initial_model_path, initial_test_image_paths, initial_test_labels
        )
        final_accuracy = fairness_service.calculate_overall_accuracy(
            scenario_model_path, initial_test_image_paths, initial_test_labels
        )
        price_of_fairness = initial_accuracy - final_accuracy
        logging.info(f"Price of Fairness for {scenario_name}: {price_of_fairness:.4f}")

        scenario_summary = {
            "scenario_name": scenario_name,
            "parameters": params,
            "initial_model_path": str(initial_model_path),
            "final_model_path": str(scenario_model_path),
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "price_of_fairness": price_of_fairness,
            "iterations": scenario_results,
        }

        results_path = RESULTS_DIR / f"{scenario_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(scenario_summary, f, indent=4)
        logging.info(f"Saved results for scenario '{scenario_name}' to {results_path}")

        logging.info(f"--- Cleaning up after scenario: {scenario_name} ---")
        if AUGMENTED_IMAGES_DIR.exists():
            shutil.rmtree(AUGMENTED_IMAGES_DIR)
            logging.info(f"Removed augmented images directory: {AUGMENTED_IMAGES_DIR}")


if __name__ == "__main__":
    run_ablation_study()
