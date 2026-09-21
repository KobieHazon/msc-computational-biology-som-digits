"""Validated loading and iteration for the supplied digit images."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float64]


class DigitsDataset:
    """Load normalized flattened images and their evaluation-only labels."""

    def __init__(self, images_path: Path, labels_path: Path, *, normalize: bool = True):
        self.images_path = images_path
        self.labels_path = labels_path
        self.normalize = normalize
        self._images: FloatMatrix | None = None
        self._labels: npt.NDArray[np.uint8] | None = None
        self._joint: np.ndarray | None = None

    @property
    def images(self) -> FloatMatrix:
        if self._images is None:
            images = np.loadtxt(self.images_path, delimiter=",", dtype=float, ndmin=2)
            if images.ndim != 2 or images.shape[0] == 0 or images.shape[1] == 0:
                raise ValueError("digit images must be a non-empty two-dimensional CSV")
            if not np.all(np.isfinite(images)):
                raise ValueError("digit images must contain only finite values")
            if np.any(images < 0) or np.any(images > 255):
                raise ValueError("digit image pixels must be between 0 and 255")
            if self.normalize:
                images /= 255.0
            self._images = np.ascontiguousarray(images)
        return self._images

    @property
    def labels(self) -> npt.NDArray[np.uint8]:
        if self._labels is None:
            labels = np.loadtxt(self.labels_path, delimiter=",", dtype=np.int64, ndmin=1)
            labels = np.asarray(labels).reshape(-1)
            if labels.shape[0] != self.images.shape[0]:
                raise ValueError("the number of labels must match the number of images")
            if np.any(labels < 0) or np.any(labels > 9):
                raise ValueError("digit labels must be integers between 0 and 9")
            self._labels = np.ascontiguousarray(labels, dtype=np.uint8)
        return self._labels

    @property
    def image_count(self) -> int:
        return self.images.shape[0]

    @property
    def flattened_image_size(self) -> int:
        return self.images.shape[1]

    def reset_iteration_order(self) -> None:
        """Restore source order before a newly seeded training run."""
        self._joint = None

    def __iter__(self) -> Iterator[tuple[np.uint8, FloatMatrix]]:
        if self._joint is None:
            self._joint = np.empty(
                self.image_count,
                dtype=np.dtype(
                    [
                        ("image", self.images.dtype, self.flattened_image_size),
                        ("label", self.labels.dtype),
                    ]
                ),
            )
            self._joint["image"] = self.images
            self._joint["label"] = self.labels
            self._joint = np.ascontiguousarray(self._joint)

        # The implementation reshuffled before every epoch.
        np.random.shuffle(self._joint)
        for row in self._joint:
            yield row["label"], row["image"]
