"""Image-folder loading and train/test splitting, following the original notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

IMAGE_SIZE = (128, 128)
SPLIT_SEED = 2


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset used in the project.

    Attributes:
        key: Name used on the command line.
        description: Human-readable description.
        class_dirs: Sub-folder names under the data root, in label order (label 0 first).
        class_names: Display names for the classes, in label order.
        test_size: Fraction held out for testing (as in the notebooks).
        patience: Early-stopping patience used in the notebook, if it differs from the default.
    """

    key: str
    description: str
    class_dirs: tuple[str, ...]
    class_names: tuple[str, ...]
    test_size: float
    patience: int | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_dirs)


DATASETS: dict[str, DatasetSpec] = {
    "ct-kidney": DatasetSpec(
        key="ct-kidney",
        description="CT kidney dataset: cyst, normal, stone, tumor (12,446 images)",
        class_dirs=("Cyst", "Normal", "Stone", "Tumor"),
        class_names=("cyst", "normal", "stone", "tumor"),
        test_size=0.2,
    ),
}


def load_image_folder(
    root: str | Path,
    class_dirs: tuple[str, ...] | list[str],
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load images from ``root/<class_dir>/*`` into float32 arrays scaled to [0, 1].

    Images are read with OpenCV (so channels stay in BGR order, as in the notebooks)
    and resized to ``image_size``. Files are read in sorted order, which makes the
    result deterministic; the notebooks used ``os.listdir`` order, so the exact
    train/test membership can differ slightly from the original runs.

    Returns:
        ``(images, labels)`` with shapes ``(N, H, W, 3)`` and ``(N,)``.
    """
    root = Path(root)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_dir in enumerate(class_dirs):
        folder = root / class_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            image = cv2.imread(str(path))
            if image is None:  # not an image
                continue
            images.append(cv2.resize(image, image_size))
            labels.append(label)
    if not images:
        raise ValueError(f"No images found under {root}")
    return np.asarray(images, dtype="float32") / 255.0, np.asarray(labels, dtype="int64")


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    test_size: float,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle, then split randomly, exactly as the notebooks did.

    Returns:
        ``x_train, x_test, y_train, y_test`` with one-hot labels.
        Note: the notebooks do not stratify this split.
    """
    onehot = keras.utils.to_categorical(labels, num_classes)
    x, y = shuffle(images, onehot, random_state=seed)
    return train_test_split(x, y, test_size=test_size, random_state=seed)


def holdout_split(
    x: np.ndarray, y: np.ndarray, val_fraction: float = 0.1, seed: int = SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a validation set off the *training* data (stratified).

    Used by the ``holdout`` protocol so early stopping and checkpoint selection never see
    the test set.

    Returns:
        ``x_train, x_val, y_train, y_val``.
    """
    return train_test_split(
        x, y, test_size=val_fraction, random_state=seed, stratify=y.argmax(axis=1)
    )
