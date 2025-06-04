import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

from utils.config import TRAINED_MODELS_DIR

_current_model_path = "trained_models/model.pth"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_idx, transform=None):
        self.image_paths = image_paths
        self.targets = [label_to_idx[lbl] for lbl in labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def train_model(image_paths: list[str], labels: list[str], existing_model_path: str = None) -> str:
    logging.debug(f"train_model called with image_paths={len(image_paths)} images, existing_model_path={existing_model_path}")
    logging.info(f"Training model with {len(image_paths)} images. Base model: {existing_model_path}")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    logging.debug(f"Label mapping created: label_to_idx={label_to_idx}, idx_to_label={idx_to_label}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(image_paths, labels, label_to_idx, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    logging.debug(f"Dataset and DataLoader initialized: dataset size={len(dataset)}, batch_size=32")

    model = SimpleCNN(num_classes=len(unique_labels))
    if existing_model_path:
        checkpoint = torch.load(existing_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 3
    for epoch in range(1, epochs + 1):
        logging.debug(f"Starting epoch {epoch}/{epochs}")
        for imgs, targets in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            logging.debug(f"Batch loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    model_file = TRAINED_MODELS_DIR / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "idx_to_label": idx_to_label}, model_file)
    logging.debug(f"Saved trained model to {model_file}")
    return str(model_file)


def predict(model_path: str, image_paths: list[str]) -> list[dict]:
    logging.debug(f"predict called with model_path={model_path}, image_paths count={len(image_paths)}")
    logging.info(f"Predicting with model {model_path} for {len(image_paths)} images.")
    checkpoint = torch.load(model_path, map_location="cpu")
    idx_to_label = checkpoint["idx_to_label"]
    logging.debug(f"Loaded checkpoint with idx_to_label mapping of size={len(idx_to_label)}")

    model = SimpleCNN(num_classes=len(idx_to_label))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.debug("Model loaded and set to evaluation mode")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    results = []
    with torch.no_grad():
        for path in image_paths:
            logging.debug(f"Predicting image at path: {path}")
            img = Image.open(path).convert("RGB")
            inp = transform(img).unsqueeze(0)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
            prob_dict = {idx_to_label[i]: probs[i] for i in range(len(probs))}
            pred_idx = int(torch.argmax(logits, dim=1).item())
            result = {"predicted_label": idx_to_label[pred_idx], "probabilities": prob_dict}
            logging.debug(f"Prediction result: {result}")
            results.append(result)
    logging.debug(f"predict returning results for {len(results)} images")
    return results

def set_current_model_path(model_path: str):
    global _current_model_path
    logging.debug(f"set_current_model_path called with model_path={model_path}")
    _current_model_path = model_path

def get_current_model_path() -> str:
    logging.debug(f"get_current_model_path returning {_current_model_path}")
    return _current_model_path