import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path

from utils.config import TRAINED_MODELS_DIR

_tokenizer = None
_label_encoder = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return _tokenizer

def _get_label_encoder(labels=None, all_labels=None):
    global _label_encoder
    if _label_encoder is None:
        _label_encoder = LabelEncoder()
        if all_labels is not None:
            _label_encoder.fit(all_labels)
        elif labels is not None:
            _label_encoder.fit(labels)
    return _label_encoder

def _prepare_data(texts: list[str], labels: list, tokenizer, label_encoder, max_length: int):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    label_ids = label_encoder.transform(labels)
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(label_ids))
    return dataset

def train_model(
    texts: list[str],
    labels: list,
    config: dict,
    updated_model_path: str,
    all_labels: list = None,
    existing_model_path: str = None,
    class_weights=None,
):
    """
    Trains or fine-tunes a DistilBERT model.
    """
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    output_path = TRAINED_MODELS_DIR / updated_model_path
    
    architecture = config["architecture"]
    hyperparameters = config["hyperparameters"]

    tokenizer = DistilBertTokenizer.from_pretrained(architecture)
    label_encoder = _get_label_encoder(labels, all_labels=all_labels)
    num_labels = len(label_encoder.classes_)

    dataset = _prepare_data(texts, labels, tokenizer, label_encoder, hyperparameters["max_length"])
    dataloader = DataLoader(dataset, batch_size=hyperparameters["batch_size"], shuffle=True)

    if existing_model_path:
        logging.info(f"Loading existing model from {existing_model_path}")
        model = DistilBertForSequenceClassification.from_pretrained(existing_model_path)
    else:
        logging.info(f"Initializing new {architecture} model.")
        model = DistilBertForSequenceClassification.from_pretrained(architecture, num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    
    if class_weights is not None:
        class_weights = class_weights.to(device)

    model.train()
    for epoch in range(hyperparameters["epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            
            loss = outputs.loss
            if class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                logits = outputs.logits
                loss = loss_fct(logits.view(-1, num_labels), batch_labels.view(-1))

            loss.backward()
            optimizer.step()
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logging.info(f"Model trained and saved to {output_path}")
    return str(output_path)

def predict_text(model_path: str, texts: list[str]):
    """
    Makes predictions on a list of texts using a trained model.
    Returns predicted class IDs and their probabilities.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
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