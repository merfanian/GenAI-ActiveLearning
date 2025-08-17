import logging
import uuid

from PIL import Image
import numpy as np

from services.model_service import predict, get_current_model_path
from services.data_service import (
    get_holdout_metadata_df,
    get_holdout_image_paths_and_labels,
)
from utils.config import AUGMENTED_IMAGES_DIR
from utils.attribute_mappings import ATTRIBUTE_VALUE_MAPPING, TARGET_LABEL_MAPPING
from services.generation_clients import get_generation_client

_generator = get_generation_client()


def generate_image_and_get_label(
    attributes_for_generation: dict, num_images: int = 5, alpha: float = 2.0, 
    exploration_steepness: float = 10.0, use_guide_image: bool = None, 
    sampling_strategy: str = "ccds"
) -> list[dict]:
    """
    Sample guide images from the holdout images of the target group using a custom weighting strategy,
    then use inpainting to generate new images guided by each sampled image and classify them to obtain target labels.
    The weighting strategy (CCDS) is designed to prioritize images that are misclassified with high confidence
    and images that are correctly classified but with low confidence.
    Returns a list of dicts with generated image paths, original attributes, and acquired labels.
    """
    logging.debug(
        f"generate_image_and_get_label called with attributes_for_generation={attributes_for_generation}, num_images={num_images}, alpha={alpha}, exploration_steepness={exploration_steepness}, sampling_strategy='{sampling_strategy}'"
    )
    results: list[dict] = []
    model_path = get_current_model_path()
    logging.debug(f"Using model_path={model_path}")

    holdout_df = get_holdout_metadata_df().copy()
    logging.debug(f"Holdout DataFrame shape: {holdout_df.shape}")
    holdout_paths, holdout_labels = get_holdout_image_paths_and_labels()
    holdout_df["image_path"] = holdout_paths

    for attr, val in attributes_for_generation.items():
        logging.debug(f"Filtering holdout data on attribute {attr}=={val}")
        if attr not in holdout_df.columns:
            raise ValueError(f"Attribute {attr} not in holdout metadata.")
        holdout_df = holdout_df[holdout_df[attr] == val]
    if holdout_df.empty:
        logging.debug(
            f"No holdout images found for attributes {attributes_for_generation}"
        )
        raise ValueError(
            f"No holdout images found for group {attributes_for_generation}"
        )

    preds = predict(model_path, holdout_df["image_path"].tolist(), holdout_df["label"].tolist())
    logging.debug(f"Predictions for holdout images: {preds}")
    items = []
    for (path, true_label), pred in zip(
        holdout_df[["image_path", "label"]].values, preds
    ):
        items.append((path, true_label, pred))

    misclassified_size = sum(1 for _, true_label, pred in items if pred["predicted_label"] != true_label)
    misclassified_ratio = misclassified_size / len(items) if len(items) > 0 else 0
    
    exploration_probability = 1 - misclassified_ratio
    
    logging.info(f"Exploration probability: {exploration_probability:.4f} based on misclassified ratio: {misclassified_ratio:.2f}")
    
    sampling_items = items
    weights = []
    if sampling_strategy == "ccds":
        logging.info("Using CCDS sampling strategy.")
        for _, true_label, pred in items:
            p = pred["probabilities"][pred["predicted_label"]]
            y = 1 if pred["predicted_label"] == true_label else 0
            weight = alpha * (1 - y) * p + y * (1 - p)
            weights.append(weight)
    elif sampling_strategy == "random":
        logging.info("Using random sampling strategy.")
        # For random sampling, we consider the entire holdout set, not just the worst group
        holdout_df = get_holdout_metadata_df().copy()
        holdout_paths, holdout_labels = get_holdout_image_paths_and_labels()
        holdout_df["image_path"] = holdout_paths
        preds = predict(model_path, holdout_df["image_path"].tolist(), holdout_df["label"].tolist())
        sampling_items = []
        for (path, true_label), pred in zip(holdout_df[["image_path", "label"]].values, preds):
            sampling_items.append((path, true_label, pred))
        weights = np.ones(len(sampling_items))
    elif sampling_strategy == "confident_misclassifications":
        logging.info("Using confident misclassifications sampling strategy.")
        sampling_items = [item for item in items if item[2]["predicted_label"] != item[1]]
        for _, true_label, pred in sampling_items:
            p = pred["probabilities"][pred["predicted_label"]]
            weights.append(p)
    elif sampling_strategy == "uncertain_classifications":
        logging.info("Using uncertain classifications sampling strategy.")
        sampling_items = [item for item in items if item[2]["predicted_label"] == item[1]]
        for _, true_label, pred in sampling_items:
            p = pred["probabilities"][pred["predicted_label"]]
            weights.append(1 - p)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    weights = np.array(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()


    descs: list[str] = []
    
    label = None
    for attr, val in attributes_for_generation.items():
        logging.debug(f"Mapping attribute {attr} value {val} to description")
        mapping = ATTRIBUTE_VALUE_MAPPING.get(attr)
        if mapping is None or val not in mapping:
            raise ValueError(f"No description mapping for attribute {attr} value {val}")
        descs.append(mapping[val])
        label = val
    prompt_gen = f"high resolution high quality realistic image of a {', '.join(descs)}."
    logging.debug(f"Image inpainting prompt: {prompt_gen}")
    logging.info(f"Image inpainting prompt: {prompt_gen}")
    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    num_generated = 0
    while num_generated < num_images:
        # Sample a single guide image in each iteration
        idx = np.random.choice(len(items), p=weights)
        guide_path, _, _ = items[idx]

        acquired_label = "discard"
        generated_image = None

        # This inner loop now only retries for the *same* guide if generation fails but the user wants to retry
        # A discard from the user will break this loop and cause a new guide to be sampled.
        while acquired_label == "discard":
            # Determine whether to explore or exploit based on the exploration_probability
            use_guide = np.random.random() > exploration_probability 
            if use_guide_image is not None:
                use_guide = use_guide_image
                
            current_guide_path = guide_path if use_guide else None

            if use_guide:
                logging.info(f"Performing exploitation using guide image: {current_guide_path}")
            else:
                logging.info("Performing exploration (no guide image).")

            generated_image = _generator.generate_image(prompt=prompt_gen, guide_image_path=current_guide_path)

            if generated_image is None:
                logging.warning(f"Image generation failed for guide: {current_guide_path}. Trying a new guide.")
                acquired_label = None  # Exit inner loop to sample a new guide
                continue

            # Check if the image is all black. If so, discard and resample.
            if not generated_image.getbbox():
                logging.warning(f"Generated image is all black for guide: {current_guide_path}. Discarding and resampling.")
                acquired_label = None  # Exit inner loop to sample a new guide
                continue

            target_name = TARGET_LABEL_MAPPING["name"]
            label_options = list(TARGET_LABEL_MAPPING["mapping"].values())
            prompt_label = (
                f"What is the {target_name} of the person in the following image?\n"
                f"Options: {', '.join(label_options)}.\n"
                f"Please respond with one of the options."
            )

            # acquired_label = _generator.get_label(generated_image, prompt_label, label_options)
            acquired_label = label

            if acquired_label == "discard":
                logging.info("Image discarded by user. Sampling a new guide image.")
                # Break the inner loop to resample a new guide in the outer loop
                break

        if acquired_label is None or acquired_label == "discard":
            # Move to the next iteration of the outer loop to sample a new guide
            continue

        filename = f"{uuid.uuid4().hex}.png"
        out_path = AUGMENTED_IMAGES_DIR / filename
        generated_image.save(out_path)
        logging.debug(f"Saved accepted image to {out_path}")

        logging.info(
            f"Generated image {out_path} from guide image {guide_path} with label '{acquired_label}'"
        )

        mapping = TARGET_LABEL_MAPPING.get("mapping", {})
        llm_acquired_label = next((k for k, v in mapping.items() if v == acquired_label), acquired_label)
        results.append(
            {
                "filename": str(filename),
                "attributes_used": attributes_for_generation,
                "llm_acquired_label": llm_acquired_label,
            }
        )
        num_generated += 1
    logging.debug(f"generate_image_and_get_label returning {results}")
    return results
