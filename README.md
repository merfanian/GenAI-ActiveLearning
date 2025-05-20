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

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Start the server:
```bash
uvicorn main:app --reload
```

## Default Model

A simple CNN (two convolutional layers and two fully connected layers) implemented in PyTorch is used by default for image classification. Training runs for 3 epochs using the Adam optimizer.

## API Endpoints

- `POST /setup_dataset`
- `POST /train_initial_model`
- `GET /evaluate_fairness`
- `POST /run_iterative_augmentation_cycle`