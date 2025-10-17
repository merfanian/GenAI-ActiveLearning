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
from services.quality_service import is_image_high_quality_and_relevant

_generator = None

def get_generator():
    global _generator
    if _generator is None:
        _generator = get_generation_client()
    return _generator

def generate_image_and_get_label(
    attributes_for_generation: dict, num_images: int = 5, alpha: float = 1.0,
    exploration_mode: str = "balanced",
    sampling_strategy: str = "ccds", augmented_data_dir = AUGMENTED_IMAGES_DIR,
    validate_quality: bool = True,
    attribute_mapping: dict = None,
    target_label_mapping: dict = None,
    mask_level: str = "moderate",
    return_guide_images: bool = False
) -> tuple[list[dict], int]:
    """
    Sample guide images from the holdout images of the target group using a custom weighting strategy,
    then use inpainting to generate new images guided by each sampled image and classify them to obtain target labels.
    The weighting strategy (CCDS) is designed to prioritize images that are misclassified with high confidence
    and images that are correctly classified but with low confidence.
    
    Args:
        return_guide_images: If True, returns the guide images directly instead of generating new images.
                            This is useful for analyzing the effect of guide image selection without generation.
    
    Returns a list of dicts with generated image paths, original attributes, and acquired labels.
    """
    generator = get_generator()
    logging.debug(
        f"generate_image_and_get_label called with attributes_for_generation={attributes_for_generation}, num_images={num_images}, alpha={alpha}, exploration_mode='{exploration_mode}', sampling_strategy='{sampling_strategy}'"
    )
    results: list[dict] = []
    num_rejected = 0
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

    exploration_probability = 0.0
    if exploration_mode == "balanced":
        misclassified_size = sum(1 for _, true_label, pred in items if pred["predicted_label"] != true_label)
        misclassified_ratio = misclassified_size / len(items) if len(items) > 0 else 0
    
        exploration_probability = 1 - 4 * misclassified_ratio
        exploration_probability = exploration_probability if exploration_probability > 0 else 0
    
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
        for _, true_label, pred in items:
            p = pred["probabilities"][pred["predicted_label"]]
            y = 1 if pred["predicted_label"] == true_label else 0
            weight = (1 - y) * p
            weights.append(weight)
    elif sampling_strategy == "uncertain_classifications":
        logging.info("Using uncertain classifications sampling strategy.")
        for _, true_label, pred in items:
            p = pred["probabilities"][pred["predicted_label"]]
            y = 1 if pred["predicted_label"] == true_label else 0
            weight = y * (1 - p)
            weights.append(weight)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    weights = np.array(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    logging.info(f"#weights:{len(weights)} , weights: {weights}")
    # Use provided mappings or fall back to defaults
    attr_map = attribute_mapping
    target_map = target_label_mapping

    descs: list[str] = []
    
    label = None
    for attr, val in attributes_for_generation.items():
        logging.debug(f"Mapping attribute {attr} value {val} to description")
        mapping = attr_map.get(attr)
        if mapping is None or val not in mapping:
            raise ValueError(f"No description mapping for attribute {attr} value {val}")
        
        # label = val # For single attribute generation, we can directly use the value as label
        descs.append(mapping[val] + " " + attr)
        # descs.append(mapping[val])
    prompt_gen = f"high resolution high quality realistic photo of a {', '.join(descs)} person (male or female) in reality."
    # prompt_gen = f"high resolution high quality realistic photo of a {', '.join(descs)}"

    logging.info(f"Image inpainting prompt: {prompt_gen}")
    augmented_data_dir.mkdir(parents=True, exist_ok=True)

    num_generated = 0
    while num_generated < num_images:
        # Sample a single guide image in each iteration
        idx = np.random.choice(len(sampling_items), p=weights)
        guide_path, true_label, pred = sampling_items[idx]

        if return_guide_images:
            # Return guide images directly without generation
            logging.info(f"Returning guide image directly: {guide_path}")
            
            # Copy the guide image to the augmented directory
            import shutil
            filename = f"guide_{uuid.uuid4().hex}.png"
            out_path = augmented_data_dir / filename
            shutil.copy2(guide_path, out_path)
            
            # Use the true label as the acquired label
            acquired_label = true_label
            
            logging.info(f"Copied guide image {out_path} with label '{acquired_label}'")
            
            mapping = target_map.get("mapping", {})
            llm_acquired_label = next((k for k, v in mapping.items() if v == acquired_label), acquired_label)
            results.append(
                {
                    "filename": str(filename),
                    "attributes_used": attributes_for_generation,
                    "llm_acquired_label": llm_acquired_label,
                    "guide_image_path": guide_path,  # Add original guide path for reference
                    "model_confidence": pred["probabilities"][pred["predicted_label"]],
                    "model_prediction": pred["predicted_label"],
                    "true_label": true_label
                }
            )
            num_generated += 1
            continue

        # Original generation logic
        acquired_label = "discard"
        generated_image = None

        # This inner loop now only retries for the *same* guide if generation fails but the user wants to retry
        # A discard from the user will break this loop and cause a new guide to be sampled.
        while acquired_label == "discard":
            if exploration_mode == "exploitation_only":
                use_guide = True
            elif exploration_mode == "exploration_only":
                use_guide = False
            elif exploration_mode == "balanced":
                use_guide = np.random.random() > exploration_probability
            else:
                raise ValueError(f"Unknown exploration_mode: {exploration_mode}")
                
            current_guide_path = guide_path if use_guide else None

            if use_guide:
                logging.info(f"Performing exploitation using guide image: {current_guide_path}")
            else:
                logging.info("Performing exploration (no guide image).")

            generator = get_generator()
            generated_image = generator.generate_image(prompt=prompt_gen, guide_image_path=current_guide_path, mask_level=mask_level)

            if generated_image is None:
                logging.warning(f"Image generation failed for guide: {current_guide_path}. Trying a new guide.")
                num_rejected += 1
                acquired_label = None  # Exit inner loop to sample a new guide
                continue

            # Check if the image is all black. If so, discard and resample.
            if validate_quality and not generated_image.getbbox():
                logging.warning(f"Generated image is all black for guide: {current_guide_path}. Discarding and resampling.")
                num_rejected += 1
                acquired_label = None  # Exit inner loop to sample a new guide
                continue

            # Validate the quality and relevance of the generated image
            if validate_quality and not is_image_high_quality_and_relevant(generated_image, prompt_gen):
                logging.warning(f"Generated image failed quality validation for guide: {current_guide_path}. Discarding and resampling.")
                num_rejected += 1
                acquired_label = None # Exit inner loop to sample a new guide
                continue

            target_name = target_map["name"]
            label_options = list(target_map["mapping"].values())
            prompt_label = (
                f"What is the {target_name} of the person in the following image?\n"
                f"Options: {', '.join(label_options)}.\n"
                f"Please respond with one of the options."
            )

            acquired_label = _generator.get_label(generated_image, prompt_label, label_options)
            # acquired_label = label

            if acquired_label == "discard":
                logging.info("Image discarded by user. Sampling a new guide image.")
                num_rejected += 1
                # Break the inner loop to resample a new guide in the outer loop
                break

        if acquired_label is None or acquired_label == "discard":
            # Move to the next iteration of the outer loop to sample a new guide
            continue

        filename = f"{uuid.uuid4().hex}.png"
        out_path = augmented_data_dir / filename
        generated_image.save(out_path)
        logging.debug(f"Saved accepted image to {out_path}")

        logging.info(
            f"Generated image {out_path} from guide image {guide_path} with label '{acquired_label}'"
        )

        mapping = target_map.get("mapping", {})
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
    return results, num_rejected
