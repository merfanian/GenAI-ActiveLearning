import sys
import os
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import (
    text_data_service,
    text_model_service,
    fairness_service,
    text_augmentation_service,
)
from utils.config import AUGMENTED_TEXT_DIR, TRAINED_MODELS_DIR
import importlib.util

# Dynamically import the config file
spec = importlib.util.spec_from_file_location(
    "config",
    "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/exp_1.3_text_modality/config.py"
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

get_benchmark_config = config_module.get_benchmark_config
NUM_ITERATIONS = config_module.NUM_ITERATIONS
AUGMENTATION_BATCH_SIZE = config_module.AUGMENTATION_BATCH_SIZE
ACCURACY_THRESHOLD = config_module.ACCURACY_THRESHOLD
NUM_RUNS = config_module.NUM_RUNS
RESULTS_DIR = config_module.RESULTS_DIR

# --- Configuration ---
RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_our_method(
    config: dict, run_number: int, logger: logging.Logger, baseline_model_path: str, all_labels: list, test_labels_true: list
):
    logger.info("--- Running Scenario: Our Method (LLM-based Augmentation) ---")
    progression = []
    scenario_model_path = baseline_model_path

    test_df = text_data_service.get_test_metadata_df()
    test_texts, _ = text_data_service.get_test_texts_and_labels()
    test_labels = list(config["label_mapping"].keys())
    for i in range(NUM_ITERATIONS):
        gp_raw = fairness_service.calculate_group_performances_text(
            scenario_model_path, test_df, test_texts, test_labels, config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=all_labels
        )
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        worst_acc_before = worst_raw.get("accuracy", 0.0)
        progression.append({"iteration": i, "worst_group_accuracy": worst_acc_before})

        if worst_acc_before >= ACCURACY_THRESHOLD:
            logger.info("Accuracy threshold met. Stopping.")
            break

        misclassified_tuples = worst_raw.get("misclassified_texts", [])
        
        misclassified_tuples.sort(key=lambda x: x[1], reverse=True)
        num_examples = AUGMENTATION_BATCH_SIZE // 2
        example_texts = [text for text, prob in misclassified_tuples[:num_examples]]

        generated_texts = text_augmentation_service.generate_text_samples(
            worst_raw["attributes"],
            AUGMENTATION_BATCH_SIZE,
            config=config,
            example_texts=example_texts
        )
        text_data_service.add_augmented_data(generated_texts, config)

        train_texts_aug, train_labels_aug = text_data_service.get_train_val_texts_and_labels(
            include_augmented=True
        )
        new_model_path = text_model_service.train_model(
            train_texts_aug,
            train_labels_aug,
            config=config,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_our_method_run_{run_number}_iter_{i + 1}.pth",
            all_labels=all_labels,
        )

        new_gp_raw = fairness_service.calculate_group_performances_text(
            new_model_path, test_df, test_texts, test_labels,  config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=all_labels
        )
        new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
        worst_acc_after = new_worst_raw.get("accuracy", 0.0)

        if worst_acc_after < worst_acc_before:
            text_data_service.remove_last_augmented_batch(len(generated_texts))
        else:
            scenario_model_path = new_model_path

    final_gp = fairness_service.calculate_group_performances_text(
        scenario_model_path, test_df, test_texts, test_labels,  config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=all_labels
    )
    final_worst = fairness_service.find_worst_performing_group(final_gp)
    final_overall = fairness_service.calculate_overall_accuracy_text(
        scenario_model_path, test_texts, test_labels_true, all_labels=all_labels
    )
    
    # --- Save Augmented Dataset ---
    augmented_df = text_data_service.get_final_training_df()
    save_dir = RESULTS_DIR / "augmented_datasets"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{config['name']}_our_method_run_{run_number}.csv"
    augmented_df.to_csv(save_path, index=False)
    logger.info(f"Saved final augmented dataset to {save_path}")
    # --- End Save ---
    
    return final_worst.get("accuracy", 0.0), final_overall, progression


def run_backtranslation_baseline(
    config: dict, run_number: int, logger: logging.Logger, baseline_model_path: str, all_labels: list, test_labels_true: list
):
    logger.info("--- Running Scenario: Back-Translation Augmentation ---")
    progression = []
    scenario_model_path = baseline_model_path

    test_df = text_data_service.get_test_metadata_df()
    test_texts, _ = text_data_service.get_test_texts_and_labels()
    test_labels = list(config["label_mapping"].keys())

    for i in range(NUM_ITERATIONS):
        att =  config["fairness_attribute"] if not config["target_fairness_equal"] else "label"
        gp_raw = fairness_service.calculate_group_performances_text(
            scenario_model_path, test_df, test_texts, test_labels, att, all_labels=all_labels
        )
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        worst_acc_before = worst_raw.get("accuracy", 0.0)
        progression.append({"iteration": i, "worst_group_accuracy": worst_acc_before})

        if worst_acc_before >= ACCURACY_THRESHOLD:
            logger.info("Accuracy threshold met. Stopping.")
            break

        train_df = text_data_service.get_train_metadata_df()
        worst_group_texts = train_df[train_df[att] == worst_raw['attributes'][att]]['text'].tolist()

        if not worst_group_texts:
            logger.warning(f"No training texts found for the worst-performing group: {worst_raw['attributes']}. Skipping augmentation for this iteration.")
            continue

        generated_texts = text_augmentation_service.augment_with_backtranslation(
            worst_group_texts,
            worst_raw["attributes"],
            AUGMENTATION_BATCH_SIZE,
        )
        text_data_service.add_augmented_data(generated_texts, config)

        train_texts_aug, train_labels_aug = text_data_service.get_train_val_texts_and_labels(
            include_augmented=True
        )
        new_model_path = text_model_service.train_model(
            train_texts_aug,
            train_labels_aug,
            config=config,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_backtranslation_run_{run_number}_iter_{i + 1}.pth",
            all_labels=all_labels,
        )

        new_gp_raw = fairness_service.calculate_group_performances_text(
            new_model_path, test_df, test_texts, test_labels,  config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=all_labels
        )
        new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
        worst_acc_after = new_worst_raw.get("accuracy", 0.0)

        if worst_acc_after < worst_acc_before:
            text_data_service.remove_last_augmented_batch(len(generated_texts))
        else:
            scenario_model_path = new_model_path

    final_gp = fairness_service.calculate_group_performances_text(
        scenario_model_path, test_df, test_texts, test_labels,  config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=all_labels
    )
    final_worst = fairness_service.find_worst_performing_group(final_gp)
    final_overall = fairness_service.calculate_overall_accuracy_text(
        scenario_model_path, test_texts, test_labels_true, all_labels=all_labels
    )
    
    # --- Save Augmented Dataset ---
    augmented_df = text_data_service.get_final_training_df()
    save_dir = RESULTS_DIR / "augmented_datasets"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{config['name']}_backtranslation_run_{run_number}.csv"
    augmented_df.to_csv(save_path, index=False)
    logger.info(f"Saved final augmented dataset to {save_path}")
    # --- End Save ---
    
    return final_worst.get("accuracy", 0.0), final_overall, progression


def run_eda_baseline(
    config: dict, run_number: int, logger: logging.Logger, baseline_model_path: str, eda_strategy: str, all_labels: list, test_labels_true: list
):
    logger.info(f"--- Running Scenario: EDA Augmentation ({eda_strategy}) ---")
    progression = []
    scenario_model_path = baseline_model_path

    test_df = text_data_service.get_test_metadata_df()
    test_texts, _ = text_data_service.get_test_texts_and_labels()
    test_labels = list(config["label_mapping"].keys())
    att =  config["fairness_attribute"] if not config["target_fairness_equal"] else "label"
    for i in range(NUM_ITERATIONS):
        gp_raw = fairness_service.calculate_group_performances_text(
            scenario_model_path, test_df, test_texts, test_labels, att, all_labels=all_labels
        )
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        worst_acc_before = worst_raw.get("accuracy", 0.0)
        progression.append({"iteration": i, "worst_group_accuracy": worst_acc_before})

        if worst_acc_before >= ACCURACY_THRESHOLD:
            logger.info("Accuracy threshold met. Stopping.")
            break

        train_df = text_data_service.get_train_metadata_df()
        worst_group_texts = train_df[train_df[att] == worst_raw['attributes'][att]]['text'].tolist()

        generated_texts = text_augmentation_service.augment_with_eda(
            worst_group_texts,
            worst_raw["attributes"],
            AUGMENTATION_BATCH_SIZE,
            eda_strategy
        )
        text_data_service.add_augmented_data(generated_texts, config)

        train_texts_aug, train_labels_aug = text_data_service.get_train_val_texts_and_labels(
            include_augmented=True
        )
        new_model_path = text_model_service.train_model(
            train_texts_aug,
            train_labels_aug,
            config=config,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_eda_{eda_strategy}_run_{run_number}_iter_{i + 1}.pth",
            all_labels=all_labels,
        )

        new_gp_raw = fairness_service.calculate_group_performances_text(
            new_model_path, test_df, test_texts, test_labels,  att, all_labels=all_labels
        )
        new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
        worst_acc_after = new_worst_raw.get("accuracy", 0.0)

        if worst_acc_after < worst_acc_before:
            text_data_service.remove_last_augmented_batch(len(generated_texts))
        else:
            scenario_model_path = new_model_path

    final_gp = fairness_service.calculate_group_performances_text(
        scenario_model_path, test_df, test_texts, test_labels,  att, all_labels=all_labels
    )
    final_worst = fairness_service.find_worst_performing_group(final_gp)
    final_overall = fairness_service.calculate_overall_accuracy_text(
        scenario_model_path, test_texts, test_labels_true, all_labels=all_labels
    )
    
    # --- Save Augmented Dataset ---
    augmented_df = text_data_service.get_final_training_df()
    save_dir = RESULTS_DIR / "augmented_datasets"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{config['name']}_eda_{eda_strategy}_run_{run_number}.csv"
    augmented_df.to_csv(save_path, index=False)
    logger.info(f"Saved final augmented dataset to {save_path}")
    # --- End Save ---
    
    return final_worst.get("accuracy", 0.0), final_overall, progression


def run_benchmark(config: dict, run_number: int):
    benchmark_name = config["name"]
    logger = logging.getLogger(f"{benchmark_name}_run_{run_number}")
    # Setup logger...
    config["target_fairness_equal"] = config["fairness_attribute"] == config["target_attribute"]
    text_data_service.load_and_validate_dataset(config)
    train_texts, train_labels = text_data_service.get_train_val_texts_and_labels(False)
    test_df = text_data_service.get_test_metadata_df()
    test_texts, test_labels_true = text_data_service.get_test_texts_and_labels()
    test_labels = list(config["label_mapping"].keys())

    # --- Baseline ---
    baseline_model_path = text_model_service.train_model(
        train_texts,
        train_labels,
        config=config,
        updated_model_path=f"{benchmark_name}_run_{run_number}_baseline.pth",
        all_labels=test_labels,
    )
    gp_raw = fairness_service.calculate_group_performances_text(
        baseline_model_path, test_df, test_texts, test_labels, config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=test_labels
    )
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    initial_worst = worst_raw.get("accuracy", 0.0)
    initial_overall = fairness_service.calculate_overall_accuracy_text(
        baseline_model_path, test_texts, test_labels_true, all_labels=test_labels
    )
    print(f"{initial_worst=}, {initial_overall=}")
    # --- Our Method ---
    text_data_service.clear_augmented_data()
    our_final_worst, our_final_overall, our_progression = run_our_method(
        config, run_number, logger, baseline_model_path, test_labels, test_labels_true
    )

    print(f"{our_final_worst=}, {our_final_overall=}")
    # --- Reweighing Baseline ---
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight("balanced", classes=unique_labels, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    reweigh_model_path = text_model_service.train_model(
        train_texts,
        train_labels,
        config=config,
        updated_model_path=f"{benchmark_name}_run_{run_number}_reweigh.pth",
        class_weights=class_weights_tensor,
        all_labels=test_labels,
    )
    reweigh_gp = fairness_service.calculate_group_performances_text(
        reweigh_model_path, test_df, test_texts, test_labels,  config["fairness_attribute"] if not config["target_fairness_equal"] else "label", all_labels=test_labels
    )
    reweigh_worst = fairness_service.find_worst_performing_group(reweigh_gp).get("accuracy", 0.0)
    reweigh_overall = fairness_service.calculate_overall_accuracy_text(
        reweigh_model_path, test_texts, test_labels_true, all_labels=test_labels
    )

    print(f"{reweigh_worst=}, {reweigh_overall=}")

    # --- Back-Translation Baseline ---
    text_data_service.clear_augmented_data()
    bt_final_worst, bt_final_overall, bt_progression = run_backtranslation_baseline(
         config, run_number, logger, baseline_model_path, test_labels, test_labels_true
     )

    print(f"{bt_final_worst=}, {bt_final_overall=}")

    # --- EDA Baselines ---
    eda_results = {}
    for strategy in ['sr', 'ri', 'rs', 'rd']:
        text_data_service.clear_augmented_data()
        eda_final_worst, eda_final_overall, eda_progression = run_eda_baseline(
            config, run_number, logger, baseline_model_path, strategy, test_labels, test_labels_true
        )
        eda_results[strategy] = {
            "overall": eda_final_overall,
            "worst": eda_final_worst,
            "progression": eda_progression,
        }

    results = {
        "benchmark_name": benchmark_name,
        "run": run_number,
        "initial_overall": initial_overall,
        "initial_worst": initial_worst,
        "our_method_overall": our_final_overall,
        "our_method_worst": our_final_worst,
        "our_method_progression": our_progression,
        "reweigh_overall": reweigh_overall,
        "reweigh_worst": reweigh_worst,
        "backtranslation_overall": bt_final_overall,
        "backtranslation_worst": bt_final_worst,
        "backtranslation_progression": bt_progression,
        "eda_results": eda_results,
    }

    results_path = RESULTS_DIR / f"{benchmark_name}_run_{run_number}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Benchmark run {run_number} complete. Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Run text modality fairness experiments.")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["ag_news", "bias_in_bios"],
        help="The benchmark to run.",
    )
    args = parser.parse_args()

    benchmark_config = get_benchmark_config(args.benchmark)

    for run in range(1, NUM_RUNS + 1):
        logging.info(
            f"========== STARTING BENCHMARK: {benchmark_config['name']}, RUN {run}/{NUM_RUNS} =========="
        )
        run_benchmark(benchmark_config, run)
        logging.info(
            f"========== COMPLETED BENCHMARK: {benchmark_config['name']}, RUN {run}/{NUM_RUNS} =========="
        )


if __name__ == "__main__":
    main()