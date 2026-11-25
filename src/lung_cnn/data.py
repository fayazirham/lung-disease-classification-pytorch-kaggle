"""Dataset discovery, splitting, and image loading utilities."""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_SEED,
    DEFAULT_STD,
    VALID_IMAGE_EXTENSIONS,
)


def find_dataset_root(input_dir, class_names=CLASS_NAMES):
    """Find the directory containing all expected class subdirectories."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Dataset input directory does not exist: {input_dir}")

    required = set(class_names)
    for root, dirs, _ in os.walk(input_dir):
        if required.issubset(dirs):
            return Path(root)

    expected = ", ".join(class_names)
    raise FileNotFoundError(
        f"Could not find all class folders ({expected}) beneath {input_dir}."
    )


def build_image_dataframe(data_dir, class_names=CLASS_NAMES):
    """Build a dataframe of image paths and stable, sorted class labels."""
    data_dir = Path(data_dir)
    sorted_classes = tuple(sorted(class_names))
    records = []
    for label, class_name in enumerate(sorted_classes):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                records.append(
                    {
                        "filepath": str(image_path),
                        "label_name": class_name,
                        "label": label,
                    }
                )

    if not records:
        raise ValueError(f"No supported images found in the class folders under {data_dir}.")
    return pd.DataFrame.from_records(records, columns=("filepath", "label_name", "label"))


def split_dataframe(dataframe, seed=DEFAULT_SEED):
    """Create reproducible stratified 70/15/15 train, validation, and test splits."""
    if dataframe.empty:
        raise ValueError("Cannot split an empty image dataframe.")
    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe["label"],
        random_state=seed,
    )
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        stratify=temporary_df["label"],
        random_state=seed,
    )
    return train_df, validation_df, test_df


class LungXRayDataset(Dataset):
    """Load normalized RGB X-ray images with optional training augmentation or TTA."""

    def __init__(
        self,
        dataframe,
        img_size=DEFAULT_IMAGE_SIZE,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        is_train=False,
        tta_mode=None,
    ):
        self.filepaths = dataframe["filepath"].to_numpy()
        self.labels = dataframe["label"].to_numpy()
        self.img_size = img_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.is_train = is_train
        self.tta_mode = tta_mode

    def __len__(self):
        return len(self.filepaths)

    def _augment_pil(self, image):
        if random.random() < 0.7:
            angle = random.uniform(-8.0, 8.0)
            image = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )

        if random.random() < 0.6:
            width, height = image.size
            scale = random.uniform(0.90, 1.0)
            new_width, new_height = int(width * scale), int(height * scale)
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            image = image.crop(
                (left, top, left + new_width, top + new_height)
            ).resize(
                (self.img_size, self.img_size),
                Image.Resampling.BILINEAR,
            )

        if random.random() < 0.6:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.88, 1.15))
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.12))
        return image

    def _apply_tta(self, image):
        if self.tta_mode == "contrast_up":
            return ImageEnhance.Contrast(image).enhance(1.08)
        if self.tta_mode == "brightness_up":
            return ImageEnhance.Brightness(image).enhance(1.05)
        if self.tta_mode == "zoom_center":
            width, height = image.size
            margin = int(width * 0.04)
            return image.crop(
                (margin, margin, width - margin, height - margin)
            ).resize(
                (self.img_size, self.img_size),
                Image.Resampling.BILINEAR,
            )
        return image

    def __getitem__(self, index):
        with Image.open(self.filepaths[index]) as source:
            image = source.convert("RGB").resize(
                (self.img_size, self.img_size),
                Image.Resampling.BILINEAR,
            )

        if self.is_train:
            image = self._augment_pil(image)
        elif self.tta_mode is not None:
            image = self._apply_tta(image)

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std

        if self.is_train and random.random() < 0.35:
            height, width = random.randint(12, 28), random.randint(12, 28)
            y0 = random.randint(0, self.img_size - height)
            x0 = random.randint(0, self.img_size - width)
            array[y0 : y0 + height, x0 : x0 + width, :] = 0.0

        image_tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image_tensor, label


def create_dataloaders(
    train_df,
    validation_df,
    test_df,
    img_size=DEFAULT_IMAGE_SIZE,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    batch_size=32,
    num_workers=2,
):
    """Create train, validation, and test data loaders from dataframe splits."""
    datasets = (
        LungXRayDataset(train_df, img_size, mean, std, is_train=True),
        LungXRayDataset(validation_df, img_size, mean, std),
        LungXRayDataset(test_df, img_size, mean, std),
    )
    return tuple(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=index == 0,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for index, dataset in enumerate(datasets)
    )
