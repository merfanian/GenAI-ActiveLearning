import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import logging
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from dotenv import load_dotenv

load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import (
    data_service,
    model_service,
    fairness_service,
    llm_augmentation_service,
    baseline_service,
)
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR
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
    ANIMALS_LABEL_TARGET
)

# --- Configuration ---
RESULTS_DIR = Path(
    "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/exp_1.1_fairness_improvement/results"
)
RESULTS_DIR.mkdir(exist_ok=True)
ARCHITECTURE = "resnet"
NUM_ITERATIONS = 12
AUGMENTATION_BATCH_SIZE = 10
ACCURACY_THRESHOLD = 0.95
NUM_RUNS = 3

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
    #     {
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


def run_our_method(config: dict, run_number: int, logger, baseline_model_path: str):
    logger.info(f"--- Running Scenario: Our Method ---")

    progression = []
    scenario_model_path = baseline_model_path
    model_service.set_current_model_path(scenario_model_path)

    benchmark_augmented_dir = (
        AUGMENTED_IMAGES_DIR / f"{config['name']}_run_{run_number}"
    )
    if benchmark_augmented_dir.exists():
        shutil.rmtree(benchmark_augmented_dir)
    benchmark_augmented_dir.mkdir(parents=True)

    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

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

        generated = llm_augmentation_service.generate_image_and_get_label(
            worst_raw["attributes"],
            AUGMENTATION_BATCH_SIZE,
            exploration_steepness=10.0,
            augmented_data_dir=benchmark_augmented_dir,
            attribute_mapping=config["attribute_mapping"],
            target_label_mapping=config["target_label_mapping"],
            validate_quality=True,
            sampling_strategy="random",
        )
        for item in generated:
            data_service.add_augmented_data(
                item["filename"],
                item["attributes_used"],
                item["llm_acquired_label"],
                augmented_data_dir=benchmark_augmented_dir,
            )

        train_paths_aug, train_labels_aug = (
            data_service.get_train_val_image_paths_and_labels(
                include_augmented=True, augmented_data_dir=benchmark_augmented_dir
            )
        )
        new_model_path = model_service.train_model(
            train_paths_aug,
            train_labels_aug,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_run_{run_number}_iter_{i + 1}.pth",
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
                len(generated), augmented_data_dir=benchmark_augmented_dir
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

    return final_worst_raw.get("accuracy", 0.0), final_overall, progression


def run_our_method_plus_reweighing(
    config: dict, run_number: int, logger, baseline_model_path: str
):
    logger.info(f"--- Running Scenario: Our Method + Reweighing ---")

    progression = []
    scenario_model_path = baseline_model_path
    model_service.set_current_model_path(scenario_model_path)

    benchmark_augmented_dir = (
        AUGMENTED_IMAGES_DIR / f"{config['name']}_run_{run_number}_reweigh"
    )
    if benchmark_augmented_dir.exists():
        shutil.rmtree(benchmark_augmented_dir)
    benchmark_augmented_dir.mkdir(parents=True)

    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

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

        generated = llm_augmentation_service.generate_image_and_get_label(
            worst_raw["attributes"],
            AUGMENTATION_BATCH_SIZE,
            exploration_steepness=10.0,
            augmented_data_dir=benchmark_augmented_dir,
            attribute_mapping=config["attribute_mapping"],
            target_label_mapping=config["target_label_mapping"],
            validate_quality=True,
            sampling_strategy="random",
        )
        for item in generated:
            data_service.add_augmented_data(
                item["filename"],
                item["attributes_used"],
                item["llm_acquired_label"],
                augmented_data_dir=benchmark_augmented_dir,
            )

        train_paths_aug, train_labels_aug = (
            data_service.get_train_val_image_paths_and_labels(
                include_augmented=True, augmented_data_dir=benchmark_augmented_dir
            )
        )

        unique_labels = np.unique(train_labels_aug)
        class_weights = compute_class_weight(
            "balanced", classes=unique_labels, y=train_labels_aug
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        new_model_path = model_service.train_model(
            train_paths_aug,
            train_labels_aug,
            existing_model_path=scenario_model_path,
            updated_model_path=f"{config['name']}_run_{run_number}_reweigh_iter_{i + 1}.pth",
            class_weights=class_weights_tensor,
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
                len(generated), augmented_data_dir=benchmark_augmented_dir
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

    return final_worst_raw.get("accuracy", 0.0), final_overall, progression


def run_our_method_plus_smote(
    config: dict, run_number: int, logger, baseline_model_path: str
):
    logger.info(f"--- Running Scenario: Our Method + SMOTE ---")

    progression = []
    feature_extractor_path = (
        baseline_model_path  # This remains frozen as per service implementation
    )

    benchmark_augmented_dir = (
        AUGMENTED_IMAGES_DIR / f"{config['name']}_run_{run_number}_smote"
    )
    if benchmark_augmented_dir.exists():
        shutil.rmtree(benchmark_augmented_dir)
    benchmark_augmented_dir.mkdir(parents=True)

    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    # Initial SMOTE model on original data
    train_paths, train_labels = data_service.get_train_val_image_paths_and_labels(
        include_augmented=False
    )
    smote_model_path, feature_extractor = baseline_service.train_with_smote(
        train_paths,
        train_labels,
        ARCHITECTURE,
        f"{config['name']}_run_{run_number}_smote_iter_0.pkl",
    )

    for i in range(NUM_ITERATIONS):
        worst_group, overall_acc = baseline_service.evaluate_smote_model(
            smote_model_path,
            feature_extractor,
            test_df,
            test_paths,
            test_labels,
            config["fairness_attribute"],
        )
        worst_acc_before = worst_group.get("accuracy", 0.0)
        progression.append({"iteration": i, "worst_group_accuracy": worst_acc_before})

        if worst_acc_before >= ACCURACY_THRESHOLD:
            logger.info("Accuracy threshold met. Stopping.")
            break

        generated = llm_augmentation_service.generate_image_and_get_label(
            worst_group["attributes"],
            AUGMENTATION_BATCH_SIZE,
            exploration_steepness=10.0,
            augmented_data_dir=benchmark_augmented_dir,
            attribute_mapping=config["attribute_mapping"],
            target_label_mapping=config["target_label_mapping"],
            validate_quality=True,
            sampling_strategy="random",
        )
        for item in generated:
            data_service.add_augmented_data(
                item["filename"],
                item["attributes_used"],
                item["llm_acquired_label"],
                augmented_data_dir=benchmark_augmented_dir,
            )

        train_paths_aug, train_labels_aug = (
            data_service.get_train_val_image_paths_and_labels(
                include_augmented=True, augmented_data_dir=benchmark_augmented_dir
            )
        )

        new_smote_model_path, _ = baseline_service.train_with_smote(
            train_paths_aug,
            train_labels_aug,
            ARCHITECTURE,
            f"{config['name']}_run_{run_number}_smote_iter_{i + 1}.pkl",
        )

        new_worst_group, _ = baseline_service.evaluate_smote_model(
            new_smote_model_path,
            feature_extractor,
            test_df,
            test_paths,
            test_labels,
            config["fairness_attribute"],
        )
        worst_acc_after = new_worst_group.get("accuracy", 0.0)

        if worst_acc_after < worst_acc_before:
            data_service.remove_last_augmented_batch(
                len(generated), augmented_data_dir=benchmark_augmented_dir
            )
        else:
            smote_model_path = new_smote_model_path

    final_worst_group, final_overall = baseline_service.evaluate_smote_model(
        smote_model_path,
        feature_extractor,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    final_worst = final_worst_group.get("accuracy", 0.0)

    return final_worst, final_overall, progression


def run_benchmark(config: dict, run_number: int):
    benchmark_name = config["name"]
    logger = logging.getLogger(f"{benchmark_name}_run_{run_number}")
    # Setup logger...

    message = data_service.load_and_validate_dataset(
        config["image_dir_path"],
        config["metadata_csv_path"],
        config["target_attribute"],
        config["fairness_attribute"],
    )
    if "error" in message:
        logger.error(f"Data loading/validation failed: {str(message['error'])[:100]}")
        return

    train_paths, train_labels, train_groups = data_service.get_train_val_image_paths_and_labels(
        include_augmented=False, fairness_attribute=config["fairness_attribute"]
    )
    test_df = data_service.get_test_metadata_df()
    test_paths, test_labels = data_service.get_test_image_paths_and_labels()

    # --- Baseline ---
    baseline_model_path = model_service.train_model(
        train_paths,
        train_labels,
        architecture=ARCHITECTURE,
        updated_model_path=f"{benchmark_name}_run_{run_number}_baseline.pth",
    )
    gp_raw = fairness_service.calculate_group_performances(
        baseline_model_path,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    initial_worst = worst_raw.get("accuracy", 0.0)
    initial_overall = fairness_service.calculate_overall_accuracy(
        baseline_model_path, test_paths, test_labels
    )

    # --- Our Method ---
    our_final_worst, our_final_overall, progression = run_our_method(
        config, run_number, logger, baseline_model_path
    )

    # --- Reweighing Baseline ---
    reweigh_model_path = baseline_service.train_with_reweighing(
        train_paths,
        train_labels,
        ARCHITECTURE,
        f"{benchmark_name}_run_{run_number}_reweigh.pth",
    )
    reweigh_gp_raw = fairness_service.calculate_group_performances(
        reweigh_model_path,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    reweigh_worst_raw = fairness_service.find_worst_performing_group(reweigh_gp_raw)
    reweigh_worst = reweigh_worst_raw.get("accuracy", 0.0)
    reweigh_overall = fairness_service.calculate_overall_accuracy(
        reweigh_model_path, test_paths, test_labels
    )

    # --- SMOTE Baseline ---
    smote_model_path, smote_feature_extractor = baseline_service.train_with_smote(
        train_paths,
        train_labels,
        ARCHITECTURE,
        f"{benchmark_name}_run_{run_number}_smote.pkl",
    )
    smote_worst_group, smote_overall = baseline_service.evaluate_smote_model(
        smote_model_path,
        smote_feature_extractor,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    smote_worst = smote_worst_group.get("accuracy", 0.0)

    # --- Our Method + Reweighing ---
    our_reweigh_worst, our_reweigh_overall, reweigh_progression = (
        run_our_method_plus_reweighing(config, run_number, logger, baseline_model_path)
    )

    # --- Our Method + SMOTE ---
    our_smote_worst, our_smote_overall, smote_progression = run_our_method_plus_smote(
        config, run_number, logger, baseline_model_path
    )

    # --- GroupDRO Baseline ---
    groupdro_model_path = model_service.train_model(
        train_paths,
        train_labels,
        groups=train_groups,
        architecture=ARCHITECTURE,
        updated_model_path=f"{benchmark_name}_run_{run_number}_groupdro.pth",
        use_group_dro=True,
    )
    groupdro_gp_raw = fairness_service.calculate_group_performances(
        groupdro_model_path,
        test_df,
        test_paths,
        test_labels,
        config["fairness_attribute"],
    )
    groupdro_worst_raw = fairness_service.find_worst_performing_group(groupdro_gp_raw)
    groupdro_worst = groupdro_worst_raw.get("accuracy", 0.0)
    groupdro_overall = fairness_service.calculate_overall_accuracy(
        groupdro_model_path, test_paths, test_labels
    )

    results = {
        "benchmark_name": benchmark_name,
        "run": run_number,
        "initial_overall": initial_overall,
        "initial_worst": initial_worst,
        "our_method_overall": our_final_overall,
        "our_method_worst": our_final_worst,
        "reweigh_overall": reweigh_overall,
        "reweigh_worst": reweigh_worst,
        "smote_overall": smote_overall,
        "smote_worst": smote_worst,
        "our_reweigh_overall": our_reweigh_overall,
        "our_reweigh_worst": our_reweigh_worst,
        "our_smote_overall": our_smote_overall,
        "our_smote_worst": our_smote_worst,
        "groupdro_overall": groupdro_overall,
        "groupdro_worst": groupdro_worst,
        "progression": progression,
        "reweigh_progression": reweigh_progression,
        "smote_progression": smote_progression,
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
