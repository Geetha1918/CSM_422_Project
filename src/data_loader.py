"""Image loading utilities with no deep-learning framework dependency."""

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


CLASS_NAMES = ("biodegradable", "recyclable", "non_recyclable")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_split(split_dir: str | Path, image_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Load labelled images and return flattened, normalized pixels and labels.

    Each 32 x 32 grayscale image becomes a 1024-dimensional input vector. Pixel
    values are divided by 255 so gradient descent works on a stable scale.
    """
    split_path = Path(split_dir)
    features: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_path / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(file_path) as image:
                image = image.convert("L").resize((image_size, image_size))
                pixels = np.asarray(image, dtype=np.float64).reshape(-1) / 255.0
            features.append(pixels)
            labels.append(label)

    if not features:
        raise ValueError(f"No images found in {split_path}")
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def one_hot(labels: np.ndarray, number_of_classes: int) -> np.ndarray:
    """Convert integer labels into one-hot target rows for Softmax training."""
    targets = np.zeros((len(labels), number_of_classes), dtype=np.float64)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def describe_split(features: np.ndarray, labels: np.ndarray) -> str:
    counts = {CLASS_NAMES[i]: int(np.sum(labels == i)) for i in range(len(CLASS_NAMES))}
    return f"images={len(features)}, input_dimension={features.shape[1]}, class_counts={counts}"
