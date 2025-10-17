import sys
import os
import json
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import data_service, fairness_service
from utils.config import TRAINED_MODELS_DIR

# --- Configuration (from run_experiment.py) ---
RESULTS_DIR = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/ablation_2.2_ccds_sampling/results")
IMAGE_DIR_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/"
METADATA_CSV_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/adience_sample_1000.csv"
TARGET_ATTRIBUTE = "label"
FAIRNESS_ATTRIBUTE = "label"
INITIAL_MODEL_NAME = "model.pth"
ARCHITECTURE = "resnet"

def update_results():
    """
    Iterates through existing result files, calculates the overall accuracy for each
    iteration's model, and updates the JSON file.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Setting up the dataset for evaluation...")
    data_service.load_and_validate_dataset(IMAGE_DIR_PATH, METADATA_CSV_PATH, TARGET_ATTRIBUTE, FAIRNESS_ATTRIBUTE)
    image_paths, labels = data_service.get_test_image_paths_and_labels()

    result_files = list(RESULTS_DIR.glob("*_run_*_results.json"))
    if not result_files:
        logging.warning("No result files found to update.")
        return

    for result_file in result_files:
        logging.info(f"Processing {result_file}...")
        with open(result_file, 'r') as f:
            results_data = json.load(f)

        updated = False
        for item in results_data:
            if "overall_accuracy" in item:
                continue  # Skip if already updated

            iteration = item["iteration"]
            
            # Reconstruct model path from filename and iteration
            parts = result_file.stem.split('_')
            scenario_name = parts[0]
            run_number = parts[2]

            if iteration == 0:
                 # The initial model was copied for the run
                model_name = f"{scenario_name}_run_{run_number}_model.pth"
            else:
                model_name = f"{scenario_name}_run_{run_number}_iter_{iteration}.pth"
            
            model_path = TRAINED_MODELS_DIR / model_name

            if not model_path.exists():
                logging.error(f"Model not found for {result_file} at iteration {iteration}: {model_path}")
                # Special case for initial model before runs were separated
                if iteration == 0 and not Path(f"{scenario_name}_run_{run_number}_model.pth").exists():
                     model_path = TRAINED_MODELS_DIR / INITIAL_MODEL_NAME
                     if not model_path.exists():
                         logging.error(f"Initial model {INITIAL_MODEL_NAME} also not found. Skipping.")
                         continue
                else:
                    continue

            logging.info(f"  Calculating accuracy for iteration {iteration} with model {model_path}...")
            overall_accuracy = fairness_service.calculate_overall_accuracy(str(model_path), image_paths, labels)
            item["overall_accuracy"] = overall_accuracy
            updated = True
            logging.info(f"    -> Overall Accuracy: {overall_accuracy:.4f}")

        if updated:
            with open(result_file, 'w') as f:
                json.dump(results_data, f, indent=4)
            logging.info(f"Updated {result_file} with overall accuracy.")
        else:
            logging.info(f"{result_file} is already up-to-date.")

if __name__ == "__main__":
    update_results()
    logging.info("--- Finished updating all result files. ---")
