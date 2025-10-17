import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch.nn as nn

from dotenv import load_dotenv
load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import text_data_service, fairness_service, text_model_service
from utils.config import TRAINED_MODELS_DIR

# --- Configuration ---
RESULTS_DIR = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/groupdro_text_baseline/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    "bias_in_bios": {
        "name": "bias_in_bios",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/groupdro_text_baseline/bias_in_bios_processed.csv",
        "text_column": "text",
        "target_attribute": "profession_label", 
        "fairness_attribute": "gender_label",
        "target_fairness_equal": False,
        "architecture": "distilbert-base-uncased",
        "hyperparameters": {
            "max_length": 256,
            "batch_size": 16,
            "epochs": 4,
            "learning_rate": 5e-5
        },
        "label_mapping": {
            "accountant": 0, "architect": 1, "attorney": 2, "chiropractor": 3, "comedian": 4,
            "composer": 5, "dentist": 6, "dietitian": 7, "dj": 8, "filmmaker": 9,
            "interior_designer": 10, "journalist": 11, "model": 12, "nurse": 13, "painter": 14,
            "paralegal": 15, "pastor": 16, "personal_trainer": 17, "photographer": 18, "physician": 19,
            "poet": 20, "professor": 21, "psychologist": 22, "rapper": 23, "software_engineer": 24,
            "surgeon": 25, "teacher": 26, "yoga_teacher": 27
        }
    },
    "ag_news": {
        "name": "ag_news", 
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/groupdro_text_baseline/ag_news_processed.csv",
        "text_column": "description",
        "target_attribute": "label_name",
        "fairness_attribute": "fairness_group",  # For AG News, we'll use label as both target and fairness
        "target_fairness_equal": True,
        "architecture": "distilbert-base-uncased",
        "hyperparameters": {
            "max_length": 128,
            "batch_size": 16,
            "epochs": 3,
            "learning_rate": 5e-5
        },
        "label_mapping": {
            "World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3
        }
    }
}

NUM_RUNS = 3
GROUP_DRO_ETA = 0.1

def prepare_text_data(texts, labels, groups, tokenizer, label_encoder, group_encoder, max_length):
    """Prepare text data for GroupDRO training"""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    label_ids = label_encoder.transform(labels)
    group_ids = group_encoder.transform(groups)
    
    dataset = TensorDataset(
        encodings['input_ids'], 
        encodings['attention_mask'], 
        torch.tensor(label_ids),
        torch.tensor(group_ids)
    )
    return dataset

def train_groupdro_text_model(
    texts: list[str],
    labels: list,
    groups: list,
    config: dict,
    updated_model_path: str,
    all_labels: list = None,
    group_dro_eta: float = 0.1
):
    """Train a DistilBERT model with GroupDRO"""
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    output_path = TRAINED_MODELS_DIR / updated_model_path
    
    architecture = config["architecture"]
    hyperparameters = config["hyperparameters"]

    tokenizer = DistilBertTokenizer.from_pretrained(architecture)
    label_encoder = LabelEncoder()
    group_encoder = LabelEncoder()
    
    if all_labels is not None:
        label_encoder.fit(all_labels)
    else:
        label_encoder.fit(labels)
    group_encoder.fit(groups)
    
    num_labels = len(label_encoder.classes_)
    num_groups = len(group_encoder.classes_)

    dataset = prepare_text_data(texts, labels, groups, tokenizer, label_encoder, group_encoder, hyperparameters["max_length"])
    dataloader = DataLoader(dataset, batch_size=hyperparameters["batch_size"], shuffle=True)

    model = DistilBertForSequenceClassification.from_pretrained(architecture, num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    
    # GroupDRO setup
    group_weights = torch.ones(num_groups)
    group_weights.requires_grad = False
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    model.train()
    for epoch in range(hyperparameters["epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, batch_labels, group_indices = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            per_sample_losses = criterion(logits.view(-1, num_labels), batch_labels.view(-1))
            
            # GroupDRO update
            for g in range(num_groups):
                group_mask = (group_indices == g)
                if group_mask.any():
                    group_loss = per_sample_losses[group_mask].mean()
                    group_weights[g] *= torch.exp(group_dro_eta * group_loss.detach())

            group_weights = group_weights / (group_weights.sum())
            weights = group_weights[group_indices]
            loss = (per_sample_losses * weights).sum()

            loss.backward()
            optimizer.step()
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save encoders
    torch.save({
        'label_encoder': label_encoder,
        'group_encoder': group_encoder
    }, output_path / 'encoders.pth')
    
    logging.info(f"GroupDRO model trained and saved to {output_path}")
    return str(output_path)

def predict_text_with_encoders(model_path: str, texts: list[str]):
    """Make predictions using saved encoders"""
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Load encoders
    encoders = torch.load(Path(model_path) / 'encoders.pth', map_location='cpu')
    label_encoder = encoders['label_encoder']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    probabilities = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
            predictions.append(predicted_class_id)
            probabilities.append(max_prob)
            
    return predictions, probabilities

def run_groupdro_benchmark(config: dict, run_number: int):
    """Run GroupDRO experiment for a single dataset"""
    dataset_name = config["name"]
    logger = logging.getLogger(f"{dataset_name}_run_{run_number}")
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    text_data_service.load_and_validate_dataset(config)
    
    train_texts, train_labels = text_data_service.get_train_val_texts_and_labels(include_augmented=False)
    test_df = text_data_service.get_test_metadata_df()
    test_texts, test_labels_true = text_data_service.get_test_texts_and_labels()
    
    # Get group labels for training
    train_df = text_data_service.get_train_metadata_df()
    train_groups = train_df[config["fairness_attribute"]].tolist()
    
    # Get all possible labels
    all_labels = list(config["label_mapping"].keys())
    
    logger.info(f"Training data: {len(train_texts)} samples, {len(set(train_groups))} groups")
    
    # Train GroupDRO model
    logger.info("Training GroupDRO model...")
    groupdro_model_path = train_groupdro_text_model(
        train_texts,
        train_labels,
        train_groups,
        config,
        f"{dataset_name}_run_{run_number}_groupdro.pth",
        all_labels=all_labels,
        group_dro_eta=GROUP_DRO_ETA
    )
    
    # Evaluate GroupDRO model
    logger.info("Evaluating GroupDRO model...")
    groupdro_gp_raw = fairness_service.calculate_group_performances_text(
        groupdro_model_path, test_df, test_texts, all_labels, config["fairness_attribute"], all_labels=all_labels
    )
    groupdro_worst_raw = fairness_service.find_worst_performing_group(groupdro_gp_raw)
    groupdro_worst = groupdro_worst_raw.get("accuracy", 0.0)
    groupdro_overall = fairness_service.calculate_overall_accuracy_text(
        groupdro_model_path, test_texts, test_labels_true, all_labels=all_labels
    )
    
    # Train baseline model for comparison
    logger.info("Training baseline model...")
    baseline_model_path = text_model_service.train_model(
        train_texts,
        train_labels,
        config,
        f"{dataset_name}_run_{run_number}_baseline.pth",
        all_labels=all_labels
    )
    
    baseline_gp_raw = fairness_service.calculate_group_performances_text(
        baseline_model_path, test_df, test_texts, all_labels, config["fairness_attribute"], all_labels=all_labels
    )
    baseline_worst_raw = fairness_service.find_worst_performing_group(baseline_gp_raw)
    baseline_worst = baseline_worst_raw.get("accuracy", 0.0)
    baseline_overall = fairness_service.calculate_overall_accuracy_text(
        baseline_model_path, test_texts, test_labels_true, all_labels=all_labels
    )
    
    # Compile results
    results = {
        "dataset_name": dataset_name,
        "run": run_number,
        "baseline_worst_group_accuracy": baseline_worst,
        "baseline_overall_accuracy": baseline_overall,
        "groupdro_worst_group_accuracy": groupdro_worst,
        "groupdro_overall_accuracy": groupdro_overall,
        "groupdro_improvement_worst": groupdro_worst - baseline_worst,
        "groupdro_improvement_overall": groupdro_overall - baseline_overall,
        "group_performances_baseline": [
            {"group": str(k), "accuracy": v["accuracy"], "count": v["count"]} 
            for k, v in baseline_gp_raw.items()
        ],
        "group_performances_groupdro": [
            {"group": str(k), "accuracy": v["accuracy"], "count": v["count"]} 
            for k, v in groupdro_gp_raw.items()
        ]
    }
    
    # Save results
    results_path = RESULTS_DIR / f"{dataset_name}_run_{run_number}_groupdro_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Baseline - Worst: {baseline_worst:.4f}, Overall: {baseline_overall:.4f}")
    logger.info(f"GroupDRO - Worst: {groupdro_worst:.4f}, Overall: {groupdro_overall:.4f}")
    logger.info(f"Improvement - Worst: {groupdro_worst - baseline_worst:.4f}, Overall: {groupdro_overall - baseline_overall:.4f}")

def main():
    """Run GroupDRO experiments on all datasets"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger("groupdro_text_experiment")
    logger.info("Starting GroupDRO text experiments...")
    
    for run in range(1, NUM_RUNS + 1):
        for dataset_name, config in DATASETS.items():
            try:
                logger.info(f"Running {dataset_name} - Run {run}")
                run_groupdro_benchmark(config, run)
            except Exception as e:
                logger.error(f"Error in {dataset_name} run {run}: {e}")
                continue
    
    logger.info("All GroupDRO text experiments completed!")

if __name__ == "__main__":
    main()
