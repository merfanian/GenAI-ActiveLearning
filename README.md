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

Set up your OpenAI API key in a `.env` file at the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

Start the server:

```bash
uvicorn main:app --reload
```

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