from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


class DigitsImagesReader:
    def __init__(self, digits_images_csv_path: Path, digits_images_keys_csv_path: Path, normalize: bool = True):
        self._digits_images_csv_path = digits_images_csv_path
        self._digits_images_key_csv_path = digits_images_keys_csv_path

        self._digits_images: Optional[np.ndarray] = None
        self._digits_images_keys: Optional[np.array] = None
        self._normalize = normalize

    def get_digits_images(self) -> np.ndarray:
        if self._digits_images is None:
            self._digits_images = np.genfromtxt(self._digits_images_csv_path,
                                                delimiter=',',
                                                dtype=np.uint8 if not self._normalize else float)
            if self._normalize:
                self._digits_images /= 255.0

        return self._digits_images

    def _get_digits_images_keys(self) -> np.array:
        if self._digits_images_keys is None:
            self._digits_images_keys = np.genfromtxt(self._digits_images_key_csv_path, delimiter=',', dtype=np.uint8)

        return self._digits_images_keys

    def __iter__(self) -> Iterator[Tuple[np.uint8, np.array]]:
        digits_images = self.get_digits_images()
        digits_images_keys = self._get_digits_images_keys()

        shuffled_images = np.empty((digits_images.shape[0], digits_images.shape[1] + 1),
                                   dtype=digits_images.dtype)
        shuffled_images[:, :-1] = digits_images
        shuffled_images[:, -1] = digits_images_keys.flatten()
        np.random.shuffle(shuffled_images)

        for shuffled_image_row in shuffled_images:
            yield shuffled_image_row[-1], shuffled_image_row[:-1]

    def get_flattened_image_size(self) -> int:
        if self._digits_images is None:
            return self.get_digits_images().shape[-1]
        else:
            first_row = np.genfromtxt(self._digits_images_csv_path, delimiter=',', dtype=np.uint8, max_rows=1)
            return first_row.shape[-1]
