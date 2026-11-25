"""Training, evaluation, prediction, and reproducibility helpers."""

import gc
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, DEFAULT_SEED, DEFAULT_STD
from .data import LungXRayDataset


def seed_everything(seed=DEFAULT_SEED):
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _autocast(device, enabled):
    return torch.autocast(device_type=torch.device(device).type, enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot train on an empty data loader.")
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    return running_loss / total, correct / total


@torch.no_grad()
def predict_probs(model, loader, device, use_amp=False):
    model.eval()
    all_probabilities = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
        all_probabilities.append(probabilities.float().cpu().numpy())

    if not all_probabilities:
        raise ValueError("Cannot predict with an empty data loader.")
    return np.concatenate(all_probabilities, axis=0)


def predict_with_tta(
    model,
    dataframe,
    device,
    img_size=DEFAULT_IMAGE_SIZE,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    batch_size=32,
    num_workers=2,
    use_amp=False,
):
    """Average predictions across the original image and three fixed TTA variants."""
    probabilities = []
    for mode in (None, "contrast_up", "brightness_up", "zoom_center"):
        dataset = LungXRayDataset(
            dataframe,
            img_size=img_size,
            mean=mean,
            std=std,
            tta_mode=mode,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        probabilities.append(predict_probs(model, loader, device, use_amp))
    return np.mean(probabilities, axis=0)


def cleanup_gpu(*objects):
    """Collect Python objects and release cached CUDA allocations when available."""
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
