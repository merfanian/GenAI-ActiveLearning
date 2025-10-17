import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")
import json
import logging
import shutil
from pathlib import Path

from services import data_service, model_service, fairness_service, llm_augmentation_service
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR

from utils.attribute_mappings import ADIENCE_GENDER_MAPPING, ADIENCE_GENDER_TARGET

# --- Configuration ---
RESULTS_DIR = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/ablation_2.2_ccds_sampling/guide_image_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Using Adience dataset for demographic fairness - better for demonstrating CCDS advantages
IMAGE_DIR_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/"
METADATA_CSV_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/metadata_with_ethnicity_5000.csv"
TARGET_ATTRIBUTE = "label"  # Gender classification
FAIRNESS_ATTRIBUTE = "ethnicity"  # Fairness across ethnic groups
INITIAL_MODEL_NAME = "model.pth"
ARCHITECTURE = "resnet"
NUM_ITERATIONS = 5  # Fewer iterations for guide image analysis
AUGMENTATION_BATCH_SIZE = 10
NUM_RUNS = 2  # Fewer runs for testing

ATTRIBUTE_MAPPING = ADIENCE_GENDER_MAPPING
TARGET_LABEL_MAPPING = ADIENCE_GENDER_TARGET

def run_guide_image_scenario(scenario_name, params, run_number):
    """Run experiment using guide images instead of generated images"""
    # Configure logging for each specific run
    log_file = RESULTS_DIR / f"{scenario_name}_guide_run_{run_number}.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    
    logging.info(f"--- Running Guide Image Scenario: {scenario_name}, Run: {run_number} ---")
    
    # Create a unique directory for this scenario run
    scenario_augmented_dir = Path(f"guide_images_{scenario_name}_run_{run_number}")
    if scenario_augmented_dir.exists():
        shutil.rmtree(scenario_augmented_dir)
    scenario_augmented_dir.mkdir(exist_ok=True)

    # Each run needs to load the dataset
    data_service.load_and_validate_dataset(IMAGE_DIR_PATH, METADATA_CSV_PATH, TARGET_ATTRIBUTE, FAIRNESS_ATTRIBUTE)
    initial_model_path = TRAINED_MODELS_DIR / INITIAL_MODEL_NAME
    
    # Reset to the initial model for each scenario run
    scenario_model_path = TRAINED_MODELS_DIR / f"{scenario_name}_guide_run_{run_number}_model.pth"
    shutil.copy(initial_model_path, scenario_model_path)
    model_service.set_current_model_path(str(scenario_model_path))
    
    scenario_results = []
    
    # Evaluate initial state
    test_df = data_service.get_test_metadata_df()
    image_paths, labels = data_service.get_test_image_paths_and_labels()
    gp_raw = fairness_service.calculate_group_performances(str(scenario_model_path), test_df, image_paths, labels, FAIRNESS_ATTRIBUTE)
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    overall_accuracy = fairness_service.calculate_overall_accuracy(str(scenario_model_path), image_paths, labels)
    scenario_results.append({
        "iteration": 0,
        "worst_group_accuracy": worst_raw.get("accuracy", 0.0),
        "overall_accuracy": overall_accuracy
    })

    for i in range(NUM_ITERATIONS):
        logging.info(f"--- Iteration {i + 1}/{NUM_ITERATIONS} for {scenario_name}, Run {run_number} ---")
        gp_raw = fairness_service.calculate_group_performances(str(scenario_model_path), test_df, image_paths, labels, FAIRNESS_ATTRIBUTE)
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        worst_acc_before = worst_raw.get("accuracy", 0.0)
        overall_acc_before = fairness_service.calculate_overall_accuracy(str(scenario_model_path), image_paths, labels)

        # Use guide images instead of generating new ones
        generated = llm_augmentation_service.generate_image_and_get_label(
            worst_raw["attributes"], AUGMENTATION_BATCH_SIZE, 
            sampling_strategy=params["sampling_strategy"], 
            augmented_data_dir=scenario_augmented_dir,
            attribute_mapping=ATTRIBUTE_MAPPING,
            target_label_mapping=TARGET_LABEL_MAPPING, 
            validate_quality=False,
            alpha=1.0,
            exploration_mode="exploitation_only",
            return_guide_images=True  # This is the key difference!
        )
        
        # Add guide images to training data
        for item in generated:
            data_service.add_augmented_data(
                item["filename"], item["attributes_used"], item["llm_acquired_label"],
                augmented_data_dir=scenario_augmented_dir
            )

        all_image_paths, all_labels, _ = data_service.get_train_val_image_paths_and_labels(
            include_augmented=True, augmented_data_dir=scenario_augmented_dir
        )
        new_model_path = model_service.train_model(
            all_image_paths, all_labels, architecture=ARCHITECTURE, 
            existing_model_path=str(scenario_model_path), 
            updated_model_path=f"{scenario_name}_guide_run_{run_number}_iter_{i + 1}.pth"
        )
        
        new_gp_raw = fairness_service.calculate_group_performances(new_model_path, test_df, image_paths, labels, FAIRNESS_ATTRIBUTE)
        new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
        worst_acc_after = new_worst_raw.get("accuracy", 0.0)
        overall_acc_after = fairness_service.calculate_overall_accuracy(new_model_path, image_paths, labels)

        if worst_acc_after < worst_acc_before:
            logging.warning(f"Performance degraded. Rolling back.")
            data_service.remove_last_augmented_batch(len(generated), augmented_data_dir=scenario_augmented_dir)
            scenario_results.append({
                "iteration": i + 1, 
                "worst_group_accuracy": worst_acc_before,
                "overall_accuracy": overall_acc_before
            })
        else:
            scenario_model_path = Path(new_model_path)
            model_service.set_current_model_path(str(scenario_model_path))
            scenario_results.append({
                "iteration": i + 1, 
                "worst_group_accuracy": worst_acc_after,
                "overall_accuracy": overall_acc_after
            })

    # Save scenario run results
    results_filename = RESULTS_DIR / f"{scenario_name}_guide_run_{run_number}_results.json"
    with open(results_filename, "w") as f:
        json.dump(scenario_results, f, indent=4)
    logging.info(f"Saved results for scenario '{scenario_name}' run {run_number} to {results_filename}")

    # Cleanup
    if scenario_augmented_dir.exists():
        shutil.rmtree(scenario_augmented_dir)

if __name__ == "__main__":
    logging.info("Setting up the dataset...")
    data_service.load_and_validate_dataset(IMAGE_DIR_PATH, METADATA_CSV_PATH, TARGET_ATTRIBUTE, FAIRNESS_ATTRIBUTE)

    logging.info("Checking for an existing initial model...")
    initial_model_path = TRAINED_MODELS_DIR / INITIAL_MODEL_NAME
    if not initial_model_path.exists():
        logging.info("No existing model found, training a new one...")
        image_paths, labels, _ = data_service.get_train_val_image_paths_and_labels(include_augmented=False)
        model_service.train_model(image_paths, labels, architecture=ARCHITECTURE, updated_model_path=INITIAL_MODEL_NAME)
    
    scenarios = {
        "ccds": {"sampling_strategy": "ccds"},
        "random": {"sampling_strategy": "random"},
        "confident_misclassifications": {"sampling_strategy": "confident_misclassifications"},
        "uncertain_classifications": {"sampling_strategy": "uncertain_classifications"},
    }

    for run in range(1, NUM_RUNS + 1):
        for name, params in scenarios.items():
            run_guide_image_scenario(name, params, run)
    
    logging.info("--- All guide image runs for all scenarios complete ---")
