# ChameleonV2

FastAPI backend for an image classification and fairness-aware data augmentation system.

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- Pandas
- Pillow
- torch
- torchvision
- scikit-learn
- openai
- python-dotenv

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your OpenAI API key in a `.env` file at the project root.  The image
generation backend can be selected with the `GENERATION_PROVIDER` variable.  By
default `openai` is used.  When using the local generator provide the API
configuration as shown below:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
# choose either 'openai' or 'local'
echo "GENERATION_PROVIDER=openai" >> .env
# settings for the local generator
echo "LOCAL_GEN_API_URL=http://localhost" >> .env
echo "LOCAL_GEN_PORT=8001" >> .env
echo "LOCAL_MASK_ENDPOINT=/v1/images/masks" >> .env
echo "LOCAL_GENERATE_ENDPOINT=/v1/images/edits" >> .env
echo "LOCAL_LABEL_ENDPOINT=/v1/images/labels" >> .env
# choose 'url' (default) to call a running labeler service or
# 'model' to use the local perfect model at trained_models/perfect.pth
echo "LOCAL_LABEL_MODE=url" >> .env
```

When using standalone generator services for masking, image generation and
labeling, you can instead provide full URLs for each endpoint:

```bash
echo "LOCAL_MASK_URL=http://mask-generator:8000/v1/images/masks" >> .env
echo "LOCAL_GENERATE_URL=http://image-generator:8000/v1/images/edits" >> .env
echo "LOCAL_LABEL_URL=http://labeler:8000/v1/images/labels" >> .env
echo "LOCAL_LABEL_MODE=url" >> .env
```

## Usage

Start the server:

```bash
uvicorn main:app --reload
```

### Docker

To run ChameleonV2 and the accompanying local generator services with
`docker-compose`:

```bash
docker-compose up --build
```

The compose file starts this application together with the mask generator,
image generator and labeler containers on a shared network. Persistent
volumes are mounted for the `augmented_images` and `trained_models`
directories so that data is preserved between runs. Environment variables
can be stored in a local `.env` file which `docker-compose` will load
automatically.

## Generate UTKFace images from pixel data

To create individual image files from the pixel data stored in a CSV file inside a ZIP archive, run:

```bash
python resources/extract_utkface_images.py path/to/utkface.zip utkface
```

## Default Model

A simple CNN (two convolutional layers and two fully connected layers) implemented in PyTorch is used by default for
image classification. Training runs for 3 epochs using the Adam optimizer.

## API Endpoints

**Dataset splitting:** After `/setup_dataset`, the data is split into train (40%), validation (20%), holdout (20%), and
test (20%) subsets using a fixed random seed. Endpoints `/train_initial_model` and `/run_iterative_augmentation_cycle`
train on the combined train+validation sets, `/evaluate_fairness` evaluates performance on the test set, and guide
images for augmentation are sampled from the holdout set only.

- `POST /setup_dataset`
- `POST /train_initial_model` (optional JSON body `{"use_augmented": bool}`; when true, includes previously generated augmented images in the training data)
- `GET /evaluate_fairness`
- `POST /run_iterative_augmentation_cycle`

  When running augmentation, k guide images are sampled from the misclassified holdout images of the worst-performing
  group (weighted by model confidence on the wrong labels), and inpainting is used to generate new images guided by
  these images. Default k=5 (configurable via augmentation_batch_size).

A companion metadata CSV for all generated augmented images is saved to `augmented_images/augmented_dataset_metadata.csv`, containing the attribute values and LLM-acquired labels for each image. To include these images in initial training, set `use_augmented=true` in the `/train_initial_model` request.
