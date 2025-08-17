import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights

from utils.config import TRAINED_MODELS_DIR

_current_model_path = "trained_models/model.pth"


def create_resnet_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def create_mobilenet_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label_to_idx[label]


def train_model(image_paths: list[str], labels: list[str], architecture: str = "resnet", existing_model_path: str = None, updated_model_path: str = None) -> str:
    logging.debug(f"train_model called with architecture='{architecture}', {len(image_paths)} images, existing_model_path={existing_model_path}")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    if architecture == "resnet":
        model = create_resnet_model(num_classes=len(dataset.unique_labels), pretrained=False)
    elif architecture == "mobilenet":
        model = create_mobilenet_model(num_classes=len(dataset.unique_labels), pretrained=False)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if existing_model_path:
        checkpoint = torch.load(existing_model_path, map_location="cpu")
        if checkpoint.get("architecture") != architecture:
            logging.warning(f"Architecture mismatch: requested '{architecture}', but checkpoint is '{checkpoint.get('architecture')}'. Training from scratch.")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    for epoch in range(1, epochs + 1):
        for imgs, targets in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    idx_to_label = {i: label for label, i in dataset.label_to_idx.items()}
    model_file = TRAINED_MODELS_DIR / "model.pth" if not updated_model_path else TRAINED_MODELS_DIR / updated_model_path
    torch.save({
        "model_state_dict": model.state_dict(),
        "idx_to_label": idx_to_label,
        "architecture": architecture
    }, model_file)
    logging.debug(f"Saved trained model to {model_file}")
    return str(model_file)


def predict(model_path: str, image_paths: list[str], labels: list[str]) -> list[dict]:
    logging.debug(f"predict called with model_path={model_path}, image_paths count={len(image_paths)}")
    checkpoint = torch.load(model_path, map_location="cpu")
    idx_to_label = checkpoint["idx_to_label"]
    architecture = checkpoint.get("architecture", "resnet") # Default to resnet for old models

    if architecture == "resnet":
        model = create_resnet_model(num_classes=len(idx_to_label), pretrained=False)
    elif architecture == "mobilenet":
        model = create_mobilenet_model(num_classes=len(idx_to_label), pretrained=False)
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).tolist()
            preds = torch.argmax(logits, dim=1).tolist()
            for i in range(len(preds)):
                pred_label = idx_to_label[preds[i]]
                prob_dict = {idx_to_label[j]: probs[i][j] for j in range(len(probs[i]))}
                results.append({"predicted_label": pred_label, "probabilities": prob_dict})

    return results

def set_current_model_path(model_path: str):
    global _current_model_path
    _current_model_path = model_path

def get_current_model_path() -> str:
    return _current_model_path
