import logging
import numpy as np

from services.text_model_service import predict, get_current_model_path
from services.text_data_service import (
    get_holdout_metadata_df,
    get_holdout_texts_and_labels,
)
from services.generation_clients import get_generation_client

_generator = get_generation_client(modality="text")

def generate_text_and_get_label(
    attributes_for_generation: dict, num_texts: int = 5, alpha: float = 2.0, exploration_steepness: float = 10.0
) -> list[dict]:
    logging.debug(
        f"generate_text_and_get_label called with attributes_for_generation={attributes_for_generation}, num_texts={num_texts}, alpha={alpha}, exploration_steepness={exploration_steepness}"
    )
    results: list[dict] = []
    model_path = get_current_model_path()
    logging.debug(f"Using model_path={model_path}")

    holdout_df = get_holdout_metadata_df().copy()
    holdout_texts, holdout_labels = get_holdout_texts_and_labels()
    holdout_df["text"] = holdout_texts

    for attr, val in attributes_for_generation.items():
        if attr not in holdout_df.columns:
            raise ValueError(f"Attribute {attr} not in holdout metadata.")
        holdout_df = holdout_df[holdout_df[attr] == val]
    if holdout_df.empty:
        raise ValueError(f"No holdout texts found for group {attributes_for_generation}")

    preds = predict(model_path, holdout_df["text"].tolist())
    items = []
    for (text, true_label), pred in zip(holdout_df[["text", "label"]].values, preds):
        items.append((text, true_label, pred))

    misclassified_ratio = sum(1 for _, true_label, pred in items if pred["predicted_label"] != true_label) / len(items) if items else 0
    exploration_probability = 1 - misclassified_ratio
    logging.info(f"Exploration probability: {exploration_probability:.4f} based on misclassified ratio: {misclassified_ratio:.2f}")

    weights = np.array([alpha * (1 - (1 if p["predicted_label"] == tl else 0)) * p["probabilities"][p["predicted_label"]] + (1 if p["predicted_label"] == tl else 0) * (1 - p["probabilities"][p["predicted_label"]]) for _, tl, p in items], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights.sum()

    num_generated = 0
    while num_generated < num_texts:
        idx = np.random.choice(len(items), p=weights)
        guide_text, _, _ = items[idx]

        use_guide = np.random.random() > exploration_probability
        current_guide = guide_text if use_guide else None
        
        prompt_gen = f"Generate a sentence with a similar sentiment to the following text: \'{guide_text}\'" if use_guide else "Generate a sentence with a clear positive or negative sentiment."
        generated_text = _generator.generate_text(prompt=prompt_gen, guide_text=current_guide)

        if not generated_text:
            logging.warning("Text generation failed. Trying again.")
            continue

        label_options = ["positive", "negative", "neutral"]
        prompt_label = f"Classify the sentiment of the following text. Options: {', '.join(label_options)}."
        acquired_label = _generator.get_text_label(generated_text, prompt_label, label_options)

        if acquired_label in label_options:
            results.append({
                "text": generated_text,
                "attributes_used": attributes_for_generation,
                "llm_acquired_label": acquired_label,
            })
            num_generated += 1
            logging.info(f"Generated text with label \'{acquired_label}\': \"{generated_text}\"")
        else:
            logging.warning(f"Could not classify generated text, discarding. Acquired label: {acquired_label}")

    return results
