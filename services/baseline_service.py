import logging
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
import joblib
import pandas as pd

from services import model_service
from services import fairness_service

def train_with_reweighing(train_paths: list[str], train_labels: list[str], architecture: str, updated_model_path: str) -> str:
    """
    Trains a model using class weights to handle data imbalance.
    """
    logging.info("Training model with class reweighing...")
    
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    model_path = model_service.train_model(
        train_paths, 
        train_labels, 
        architecture=architecture, 
        updated_model_path=updated_model_path,
        class_weights=class_weights_tensor
    )
    return model_path

def train_with_smote(train_paths: list[str], train_labels: list[str], architecture: str, updated_model_path: str) -> tuple[str, torch.nn.Module]:
    """
    Trains a classifier on data balanced with SMOTE.
    Returns the path to the saved model bundle and the feature extractor used.
    """
    logging.info("Extracting features for SMOTE...")
    
    feature_extractor = model_service.get_feature_extractor(architecture)
    transform = model_service.get_default_transform()
    
    # Create a temporary dataset to get the label-to-index mapping
    temp_dataset = model_service.ImageDataset(train_paths, train_labels, transform)
    idx_to_label = {i: label for label, i in temp_dataset.label_to_idx.items()}

    dataset = model_service.ImageDataset(train_paths, train_labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = feature_extractor(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.cpu().numpy())
            
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    logging.info(f"Applying SMOTE to {features.shape[0]} samples...")
    smote = SMOTE(random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    
    logging.info(f"Training classifier on {features_resampled.shape[0]} balanced samples...")
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(features_resampled, labels_resampled)
    
    model_bundle = {
        'model': classifier,
        'idx_to_label': idx_to_label
    }
    
    smote_model_path = f"trained_models/{updated_model_path}"
    joblib.dump(model_bundle, smote_model_path)
    
    return smote_model_path, feature_extractor

def evaluate_smote_model(model_bundle_path: str, feature_extractor: torch.nn.Module, test_df: pd.DataFrame, test_paths: list[str], test_labels: list[str], fairness_attribute: str):
    """
    Evaluates a scikit-learn model trained on SMOTE-balanced features.
    """
    logging.info(f"Evaluating SMOTE model: {model_bundle_path}")
    
    # Load the model and label mapping
    model_bundle = joblib.load(model_bundle_path)
    model = model_bundle['model']
    idx_to_label = model_bundle['idx_to_label']

    # Extract features from the test set
    transform = model_service.get_default_transform()
    dataset = model_service.ImageDataset(test_paths, test_labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    with torch.no_grad():
        for imgs, _ in loader:
            feats = feature_extractor(imgs)
            features.append(feats.cpu().numpy())
    features = np.concatenate(features)

    # Make predictions
    predicted_indices = model.predict(features)
    predicted_labels = [idx_to_label[i] for i in predicted_indices]

    # Calculate fairness metrics
    df = test_df.copy()
    df["predicted_label"] = predicted_labels
    
    group_perfs = {}
    groups = df.groupby(fairness_attribute)
    for group_vals, group_df in groups:
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        key = tuple(zip([fairness_attribute], group_vals))
        correct = (group_df["predicted_label"].astype(str) == group_df["label"].astype(str)).sum()
        count = len(group_df)
        acc = correct / count if count else 0.0
        group_perfs[key] = {"accuracy": acc, "count": count}
        
    worst_group = fairness_service.find_worst_performing_group(group_perfs)
    
    correct_total = (df["predicted_label"].astype(str) == df["label"].astype(str)).sum()
    overall_accuracy = correct_total / len(df) if len(df) > 0 else 0.0
    
    return worst_group, overall_accuracy