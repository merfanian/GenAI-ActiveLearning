import base64
import io
import logging
import os
import urllib.request
import uuid

from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from PIL import Image
import numpy as np

from services.model_service import predict, get_current_model_path
from services.data_service import get_holdout_metadata_df, get_holdout_image_paths_and_labels, add_augmented_data
from utils.config import AUGMENTED_IMAGES_DIR
from utils.attribute_mappings import ATTRIBUTE_VALUE_MAPPING, TARGET_LABEL_MAPPING


def generate_image_and_get_label(attributes_for_generation: dict, num_images: int = 5) -> list[dict]:
    """
    Sample guide images from the misclassified holdout images of the target group based on model confidence,
    then use inpainting to generate new images guided by each sampled image and classify them to obtain target labels.
    Returns a list of dicts with generated image paths, original attributes, and acquired labels.
    """
    logging.debug(f"generate_image_and_get_label called with attributes_for_generation={attributes_for_generation}, num_images={num_images}")
    results: list[dict] = []
    model_path = get_current_model_path()
    logging.debug(f"Using model_path={model_path}")

    holdout_df = get_holdout_metadata_df().copy()
    logging.debug(f"Holdout DataFrame shape: {holdout_df.shape}")
    holdout_paths, _ = get_holdout_image_paths_and_labels()
    holdout_df["image_path"] = holdout_paths

    for attr, val in attributes_for_generation.items():
        logging.debug(f"Filtering holdout data on attribute {attr}=={val}")
        if attr not in holdout_df.columns:
            raise ValueError(f"Attribute {attr} not in holdout metadata.")
        holdout_df = holdout_df[holdout_df[attr] == val]
    if holdout_df.empty:
        logging.debug(f"No holdout images found for attributes {attributes_for_generation}")
        raise ValueError(f"No holdout images found for group {attributes_for_generation}")

    preds = predict(model_path, holdout_df["image_path"].tolist())
    logging.debug(f"Predictions for holdout images: {preds}")
    items = []
    for (path, true_label), pred in zip(holdout_df[["image_path", "label"]].values, preds):
        items.append((path, true_label, pred))

    misclassified = [it for it in items if it[2]["predicted_label"] != it[1]]
    logging.debug(f"Misclassified items count: {len(misclassified)}")
    if not misclassified:
        misclassified = items

    weights = np.array([it[2]["probabilities"][it[2]["predicted_label"]] for it in misclassified], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    sample_k = min(num_images, len(misclassified))
    indices = np.random.choice(len(misclassified), size=sample_k, replace=False, p=weights)
    selected = [misclassified[i] for i in indices]
    logging.debug(f"Sampling {sample_k} guide images from misclassified items: indices={indices}")

    descs: list[str] = []
    for attr, val in attributes_for_generation.items():
        logging.debug(f"Mapping attribute {attr} value {val} to description")
        mapping = ATTRIBUTE_VALUE_MAPPING.get(attr)
        if mapping is None or val not in mapping:
            raise ValueError(f"No description mapping for attribute {attr} value {val}")
        descs.append(mapping[val])
    prompt_gen = f"Using this reference image, generate a high-quality inpainted image of a person who is {', '.join(descs)}."
    logging.debug(f"Image inpainting prompt: {prompt_gen}")
    logging.info(f"Image inpainting prompt: {prompt_gen}")
    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for guide_path, _, _ in selected:
        logging.debug(f"Generating inpainted image using guide image path: {guide_path}")
        guide_img = Image.open(guide_path).convert("RGBA")
        mask = Image.new("RGBA", guide_img.size, (255, 255, 255, 255))
        mask_io = io.BytesIO()
        mask.save(mask_io, format="PNG")
        mask_io.seek(0)

        with open(guide_path, "rb") as guide_file:
            try:
                response = client.images.edit(
                    image=[guide_file], prompt=prompt_gen, n=1, model="gpt-image-1",
                )
                logging.debug("Used generate_edit for image generation")
            except Exception as e:
                logging.error(f"Failed to generate edit for image generation: {e}")
                # response = client.images.generate(prompt=prompt_gen, n=1, size="256x256")
                # logging.debug("Fallback to generate for image generation")

        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        filename = f"{uuid.uuid4().hex}.png"
        out_path = AUGMENTED_IMAGES_DIR / filename

        image = Image.open(io.BytesIO(image_bytes))
        image.save(out_path)
        logging.debug(f"Saved generated image to {out_path}")

        target_name = TARGET_LABEL_MAPPING["name"]
        label_options = list(TARGET_LABEL_MAPPING["mapping"].values())
        prompt_label = (
            f"What is the {target_name} of the person in the following image?\n"
            f"Options: {', '.join(label_options)}.\n"
            f"Please respond with one of the options."
        )
        logging.debug(f"Labeling prompt: {prompt_label}")
        logging.info(f"Labeling prompt: {prompt_label}")
        chat_resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_label}], temperature=0
        )
        llm_label_str = chat_resp.choices[0].message.content.strip().lower()
        logging.debug(f"LLM raw label string: {llm_label_str}")

        inv_map = {v.lower(): k for k, v in TARGET_LABEL_MAPPING["mapping"].items()}
        if llm_label_str not in inv_map:
            logging.debug(f"Unexpected LLM label string: {llm_label_str}")
            raise ValueError(f"Unexpected label from LLM: {llm_label_str}")
        llm_label = inv_map[llm_label_str]

        results.append(
            {
                "filename": str(filename),
                "attributes_used": attributes_for_generation,
                "llm_acquired_label": llm_label,
            }
        )
    logging.debug(f"generate_image_and_get_label returning {results}")
    return results
