import logging
import uuid
import random
from PIL import Image, ImageDraw
from utils.config import AUGMENTED_IMAGES_DIR
from .data_service import get_metadata_df

def generate_image_and_get_label(attributes_for_generation: dict, num_images: int = 1) -> list[dict]:
    results = []
    attrs_str = ", ".join(f"{k}={v}" for k, v in attributes_for_generation.items())
    prompt1 = f"Generate image with {attrs_str}."
    logging.info(prompt1)
    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(num_images):
        filename = f"{uuid.uuid4().hex}.png"
        path = AUGMENTED_IMAGES_DIR / filename
        img = Image.new("RGB", (100, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), f"Img: {attributes_for_generation}", fill="black")
        img.save(path)
        prompt2 = f"What is the label for the image generated with {attrs_str}?"
        logging.info(prompt2)
        labels = get_metadata_df()["label"].unique().tolist()
        llm_label = random.choice(labels)
        logging.info(f"Simulated LLM label: {llm_label}")
        results.append({"image_path": str(path), "attributes_used": attributes_for_generation, "llm_acquired_label": llm_label})
    return results