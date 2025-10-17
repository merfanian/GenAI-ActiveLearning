import logging
from .model_service import predict
from . import text_model_service
from sklearn.preprocessing import LabelEncoder


def calculate_group_performances(model_path: str, metadata_df, all_image_paths: list[str], labels: list[str], fairness_attribute: str) -> dict:
    logging.debug(f"calculate_group_performances called with model_path={model_path}, metadata_df shape={getattr(metadata_df, 'shape', None)}, all_image_paths count={len(all_image_paths)}")
    preds = predict(model_path, all_image_paths, labels)
    logging.debug(f"Predictions: {preds}")
    df = metadata_df.copy()
    df["predicted_label"] = [p["predicted_label"] for p in preds]
    
    group_perfs = {}
    if fairness_attribute:
        groups = df.groupby(fairness_attribute)
    else:
        groups = [((), df)]

    for group_vals, group_df in groups:
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        key = tuple(zip([fairness_attribute], group_vals))
        correct = (group_df["predicted_label"] == group_df["label"]).sum()
        count = len(group_df)
        acc = correct / count if count else 0.0
        group_perfs[key] = {"accuracy": acc, "count": count}
    
    logging.debug(f"Group performances computed: {group_perfs}")
    return group_perfs

def calculate_group_performances_text(model_path: str, metadata_df, all_texts: list[str], labels: list, fairness_attribute: str, all_labels: list = None) -> dict:
    logging.debug(f"calculate_group_performances_text called with model_path={model_path}, metadata_df shape={getattr(metadata_df, 'shape', None)}, all_texts count={len(all_texts)}")
    
    label_encoder = LabelEncoder()
    if all_labels is not None:
        label_encoder.fit(all_labels)
    else:
        label_encoder.fit(labels) # Fit on all possible labels
    
    predicted_label_ids, probabilities = text_model_service.predict_text(model_path, all_texts)
    predicted_labels = label_encoder.inverse_transform(predicted_label_ids)
    
    df = metadata_df.copy()
    df["predicted_label"] = predicted_labels
    df["text"] = all_texts
    df["probability"] = probabilities
    
    group_perfs = {}
    groups = df.groupby(fairness_attribute)

    for group_val, group_df in groups:
        key = ((fairness_attribute, group_val),)
        correct = (group_df["predicted_label"] == group_df["label"]).sum()
        count = len(group_df)
        acc = correct / count if count else 0.0
        
        misclassified_df = group_df[group_df["predicted_label"] != group_df["label"]]
        misclassified_texts = list(zip(misclassified_df["text"], misclassified_df["probability"]))
        
        group_perfs[key] = {
            "accuracy": acc,
            "count": count,
            "misclassified_texts": misclassified_texts
        }
    
    logging.debug(f"Text group performances computed for {len(group_perfs)} groups.")
    return group_perfs


def find_worst_performing_group(group_performances: dict) -> dict:
    logging.debug(f"find_worst_performing_group called with group_performances={group_performances}")
    if not group_performances:
        logging.debug("No group performances to evaluate, returning empty dict")
        return {}
    worst = min(group_performances.items(), key=lambda x: x[1]["accuracy"])
    attrs_tuple, metrics = worst
    result = {
        "attributes": dict(attrs_tuple),
        "accuracy": metrics["accuracy"],
        "count": metrics["count"],
        "misclassified_texts": metrics.get("misclassified_texts", [])
    }
    logging.debug(f"Worst performing group result: {result['attributes']}, accuracy: {result['accuracy']}")
    return result

def calculate_overall_accuracy(model_path: str, all_image_paths: list[str], labels: list[str]) -> float:
    logging.debug(f"calculate_overall_accuracy called with model_path={model_path}, all_image_paths count={len(all_image_paths)}")
    preds = predict(model_path, all_image_paths, labels)
    predicted_labels = [p["predicted_label"] for p in preds]
    correct = sum(1 for p, l in zip(predicted_labels, labels) if p == l)
    total = len(labels)
    accuracy = correct / total if total else 0.0
    logging.debug(f"Overall accuracy: {accuracy}")
    return accuracy

def calculate_overall_accuracy_text(model_path: str, all_texts: list[str], labels: list, all_labels: list = None) -> float:
    logging.debug(f"calculate_overall_accuracy_text called with model_path={model_path}, all_texts count={len(all_texts)}")
    
    label_encoder = LabelEncoder()
    if all_labels is not None:
        label_encoder.fit(all_labels)
    else:
        label_encoder.fit(labels)
    
    predicted_label_ids, _ = text_model_service.predict_text(model_path, all_texts)
    predicted_labels = label_encoder.inverse_transform(predicted_label_ids)
    
    correct = sum(1 for p, l in zip(predicted_labels, labels) if p == l)
    total = len(labels)
    accuracy = correct / total if total else 0.0
    logging.debug(f"Overall text accuracy: {accuracy}")
    return accuracy
