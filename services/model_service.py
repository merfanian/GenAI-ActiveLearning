import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

from utils.config import TRAINED_MODELS_DIR

_current_model_path = None

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
    logging.info(f"Training model with {len(image_paths)} images. Base model: {existing_model_path}")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(image_paths, labels, label_to_idx, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN(num_classes=len(unique_labels))
    if existing_model_path:
        checkpoint = torch.load(existing_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 3
    for _ in range(epochs):
        for imgs, targets in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model_file = TRAINED_MODELS_DIR / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "idx_to_label": idx_to_label}, model_file)
    return str(model_file)


def predict(model_path: str, image_paths: list[str]) -> list[dict]:
    logging.info(f"Predicting with model {model_path} for {len(image_paths)} images.")
    checkpoint = torch.load(model_path, map_location="cpu")
    idx_to_label = checkpoint["idx_to_label"]

    model = SimpleCNN(num_classes=len(idx_to_label))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    results = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            inp = transform(img).unsqueeze(0)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
            prob_dict = {idx_to_label[i]: probs[i] for i in range(len(probs))}
            pred_idx = int(torch.argmax(logits, dim=1).item())
            results.append({"predicted_label": idx_to_label[pred_idx], "probabilities": prob_dict})
    return results

def set_current_model_path(model_path: str):
    global _current_model_path
    _current_model_path = model_path

def get_current_model_path() -> str:
    return _current_model_path