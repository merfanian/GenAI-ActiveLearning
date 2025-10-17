# Robustify

FastAPI backend for an image/text classification and task-aware data augmentation system.

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
- diffusers
- transformers
- accelerate
- invisible-watermark

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your OpenAI API key in a `.env` file at the project root. The image
generation backend can be selected with the `GENERATION_PROVIDER` variable. By
default, `local_inpainting` is used, which leverages a local Stable Diffusion
model.

To use OpenAI, set `GENERATION_PROVIDER=openai`. For other local generation
services, set `GENERATION_PROVIDER=local`.

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
# choose 'local_inpainting', 'openai', or 'local'
echo "GENERATION_PROVIDER=local_inpainting" >> .env
```

When using the `local` or `local_inpainting` providers, you must provide URLs for the
masking and labeling services.

```bash
echo "LOCAL_MASK_URL=http://mask-generator:8000/v1/images/masks" >> .env
echo "LOCAL_LABEL_URL=http://labeler:8000/v1/images/labels" >> .env
# choose 'url' to call the labeler service, 'model' for the local model,
# or 'openai' to query OpenAI directly
echo "LOCAL_LABEL_MODE=url" >> .env
```

When using the `local` provider, you must also provide a URL for the image
generation service:

```bash
echo "LOCAL_GENERATE_URL=http://image-generator:8000/v1/images/edits" >> .env
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

## Download Adience dataset

Use the following utility to download the Adience benchmark and build a
metadata CSV compatible with this application:

```bash
python resources/download_adience.py adience
```

## Default Model

A simple CNN (two convolutional layers and two fully connected layers) implemented in PyTorch is used by default for
image classification. Training runs for 3 epochs using the Adam optimizer.

## API Endpoints

**Dataset splitting:** After `/setup_dataset`, the data is split into train (40%), validation (20%), holdout (20%), and
test subsets. To obtain reliable fairness metrics, a minimum of five samples from every attribute group are placed in
the test set even if this exceeds the nominal 20% split. Endpoints `/train_initial_model` and
`/run_iterative_augmentation_cycle` train on the combined train+validation sets, `/evaluate_fairness` evaluates
performance on the test set, and guide images for augmentation are sampled from the holdout set only. The fairness
endpoint requires the name of the model file (e.g. `model.pth`) to evaluate.

- `POST /setup_dataset`
- `POST /train_initial_model` (optional JSON body `{"use_augmented": bool}`; when true, includes previously generated augmented images in the training data)
- `GET /evaluate_fairness?model_name=<file.pth>`
- `POST /run_iterative_augmentation_cycle`

  When running augmentation, k guide images are sampled from the misclassified holdout images of the worst-performing
  group (weighted by model confidence on the wrong labels), and inpainting is used to generate new images guided by
  these images. Default k=5 (configurable via augmentation_batch_size).

A companion metadata CSV for all generated augmented images is saved to `augmented_images/augmented_dataset_metadata.csv`, containing the attribute values and LLM-acquired labels for each image. To include these images in initial training, set `use_augmented=true` in the `/train_initial_model` request.
