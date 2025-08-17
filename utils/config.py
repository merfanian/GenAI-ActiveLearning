from pathlib import Path

AUGMENTED_IMAGES_DIR = Path("augmented_images")
TRAINED_MODELS_DIR = Path("trained_models")

# --- Dataset Configuration ---
# Set these values to the dataset you want to use.
# Example for UTKFace:
# DATASET_NAME = "utkface"
# IMAGE_DIR = "resources/utkface/utkface_aligned_cropped/UTKFace"
# METADATA_CSV = "resources/utkface/utkface_all_metadata.csv"
# ATTRIBUTE_MAPPINGS_MODULE = "utils.attribute_mappings"

# Example for FFHQ:
# DATASET_NAME = "ffhq"
# IMAGE_DIR = "resources/ffhq/images1024x1024"
# METADATA_CSV = "resources/ffhq/ffhq_all_metadata.csv"
# ATTRIBUTE_MAPPINGS_MODULE = "utils.attribute_mappings"

# Example for Adience:
# DATASET_NAME = "adience"
# IMAGE_DIR = "resources/adience/AdienceBenchmarkGenderAndAgeClassification/faces"
# METADATA_CSV = "resources/adience/adience_all_metadata.csv"
# ATTRIBUTE_MAPPINGS_MODULE = "utils.attribute_mappings"

# Example for Fruits:
DATASET_NAME = "fruits"
IMAGE_DIR = "resources/fruits/datasets/"
METADATA_CSV = "resources/fruits/fruits.csv"
ATTRIBUTE_MAPPINGS_MODULE = "utils.attribute_mappings"

# Example for Sentiment Analysis:
TEXT_DATASET_NAME = "sentiment"
TEXT_METADATA_CSV = "resources/sentiment/sentiment_data.csv"

# --- LLM Configuration ---
# The model to use for data augmentation.
# Options: "openai", "gemini", "human_in_the_loop"
GENERATION_CLIENT = "openai"
OPENAI_MODEL = "dall-e-3"
GEMINI_MODEL = "gemini-1.5-pro-vision-001"
MAX_TOKENS = 1024
TEMPERATURE = 0.4
# --- Orchestration Configuration ---
# The number of iterations to run the active learning loop.
NUM_ITERATIONS = 5
# The number of images to generate in each iteration.
NUM_IMAGES_TO_GENERATE = 5
# The path to the initial model to use for the first iteration.
# If None, a new model will be trained from scratch.
INITIAL_MODEL_PATH = None
# The name of the experiment, for logging and output files.
EXPERIMENT_NAME = f"{DATASET_NAME}_{GENERATION_CLIENT}_{NUM_ITERATIONS}"
RESULTS_DIR = Path("notebooks/results")
RESULTS_FILE = RESULTS_DIR / f"{EXPERIMENT_NAME}.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logging_config_file = "logging.conf"
