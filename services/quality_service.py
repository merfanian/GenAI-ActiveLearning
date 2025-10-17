import logging
import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Initialize the CLIP model
# Using a smaller, faster model is often sufficient for this kind of validation.
# "clip-ViT-B-32" is a good balance of performance and speed.
_clip_model = SentenceTransformer("clip-ViT-B-32")
logging.info("CLIP model loaded for quality validation.")

# --- Quality Validation Heuristics ---

def is_blurry(image: Image.Image, threshold: float = 80.0) -> bool:
    """
    Detects if an image is blurry using the variance of the Laplacian.
    A lower variance suggests a blurrier image.
    """
    # Convert PIL Image to grayscale OpenCV format
    cv_image = np.array(image.convert("L"))
    laplacian_var = cv2.Laplacian(cv_image, cv2.CV_64F).var()
    logging.debug(f"Image Laplacian variance: {laplacian_var:.2f} (Threshold: {threshold})")
    return laplacian_var < threshold

def get_clip_similarity(image: Image.Image, text_prompt: str) -> float:
    """
    Computes the cosine similarity between the image and text prompt embeddings.
    """
    try:
        # Encode both the image and the prompt
        image_embedding = _clip_model.encode(image, convert_to_tensor=True)
        text_embedding = _clip_model.encode(text_prompt, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(image_embedding, text_embedding)
        
        # Return the score as a float
        score = similarity.item()
        logging.debug(f"CLIP similarity score: {score:.4f} for prompt: '{text_prompt}'")
        return score
    except Exception as e:
        logging.error(f"Error computing CLIP similarity: {e}", exc_info=True)
        return 0.0 # Return a score of 0 if an error occurs

# --- Main Validation Service ---

def is_image_high_quality_and_relevant(
    image: Image.Image,
    prompt: str,
    blur_threshold: float = 20.0,
    clip_similarity_threshold: float = 0.27,
) -> bool:
    """
    Validates a generated image based on heuristics for quality and relevance.

    Args:
        image: The PIL Image to validate.
        prompt: The text prompt used for generation.
        blur_threshold: The threshold for Laplacian variance to detect blur.
        clip_similarity_threshold: The minimum cosine similarity score for relevance.

    Returns:
        True if the image passes all quality checks, False otherwise.
    """
    logging.debug("Starting quality and relevance validation for generated image.")

    # # 1. Blur Detection (fast pre-check)
    # if is_blurry(image, blur_threshold):
    #     logging.warning("Image failed quality validation: Detected as blurry.")
    #     return False

    # 2. CLIP Similarity Check (relevance)
    similarity_score = get_clip_similarity(image, prompt)
    if similarity_score < clip_similarity_threshold:
        logging.warning(
            f"Image failed relevance validation: CLIP score {similarity_score:.4f} is below threshold {clip_similarity_threshold}."
        )
        return False

    logging.info(f"Image passed all quality and relevance checks (CLIP score: {similarity_score:.4f}).")
    return True
