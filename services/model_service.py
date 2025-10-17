import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights, densenet121, DenseNet121_Weights

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


def create_densenet_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, groups=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.groups = groups
        self.transform = transform
        
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        
        if self.groups:
            self.unique_groups = sorted(list(set(groups)))
            self.group_to_idx = {group: i for i, group in enumerate(self.unique_groups)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        label_idx = self.label_to_idx[label]
        
        if self.groups:
            group = self.groups[idx]
            group_idx = self.group_to_idx[group]
            return img, label_idx, group_idx
        
        return img, label_idx


class PredictionImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_feature_extractor(architecture: str = "resnet") -> nn.Module:
    if architecture == "resnet":
        model = create_resnet_model(num_classes=1000, pretrained=True) # Load pretrained
        model.fc = nn.Identity() # Remove the final classification layer
    elif architecture == "mobilenet":
        model = create_mobilenet_model(num_classes=1000, pretrained=True)
        model.classifier = nn.Identity()
    elif architecture == "densenet":
        model = create_densenet_model(num_classes=1000, pretrained=True)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported architecture for feature extraction: {architecture}")
    model.eval()
    return model

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutout_data(x, length=16):
    h, w = x.size(2), x.size(3)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x_rand = np.random.randint(w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x_rand - length // 2, 0, w)
    x2 = np.clip(x_rand + length // 2, 0, w)
    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(x)
    x = x * mask
    return x

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def train_model(
    image_paths: list[str], 
    labels: list[str], 
    groups: list[str] = None,
    architecture: str = "resnet", 
    existing_model_path: str = None, 
    updated_model_path: str = None, 
    class_weights: torch.Tensor = None,
    use_group_dro: bool = False,
    group_dro_eta: float = 0.1,
    augmentation_method: str = None
) -> str:
    logging.debug(f"train_model called with architecture='{architecture}', {len(image_paths)} images, existing_model_path={existing_model_path}")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    transform = get_default_transform()
    dataset = ImageDataset(image_paths, labels, groups, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    if architecture == "resnet":
        model = create_resnet_model(num_classes=len(dataset.unique_labels), pretrained=False)
    elif architecture == "mobilenet":
        model = create_mobilenet_model(num_classes=len(dataset.unique_labels), pretrained=False)
    elif architecture == "densenet":
        model = create_densenet_model(num_classes=len(dataset.unique_labels), pretrained=False)
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
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean' if not use_group_dro else 'none')
    
    if use_group_dro:
        if not groups:
            raise ValueError("Group labels must be provided to use GroupDRO.")
        num_groups = len(dataset.unique_groups)
        group_weights = torch.ones(num_groups)
        group_weights.requires_grad = False

    epochs = 10
    for epoch in range(1, epochs + 1):
        for data in loader:
            optimizer.zero_grad()
            
            if use_group_dro:
                imgs, targets, group_indices = data
                outputs = model(imgs)
                per_sample_losses = criterion(outputs, targets)
                
                for g in range(num_groups):
                    group_mask = (group_indices == g)
                    if group_mask.any():
                        group_loss = per_sample_losses[group_mask].mean()
                        group_weights[g] *= torch.exp(group_dro_eta * group_loss.detach())

                group_weights = group_weights / (group_weights.sum())
                
                weights = group_weights[group_indices]
                loss = (per_sample_losses * weights).sum()

            else:
                imgs, targets = data
                
                if augmentation_method == 'mixup':
                    imgs, targets_a, targets_b, lam = mixup_data(imgs, targets)
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                elif augmentation_method == 'cutout':
                    imgs = cutout_data(imgs)
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                elif augmentation_method == 'cutmix':
                    imgs, targets_a, targets_b, lam = cutmix_data(imgs, targets)
                    outputs = model(imgs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
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
    elif architecture == "densenet":
        model = create_densenet_model(num_classes=len(idx_to_label), pretrained=False)
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PredictionImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    results = []
    with torch.no_grad():
        for imgs in loader:
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
