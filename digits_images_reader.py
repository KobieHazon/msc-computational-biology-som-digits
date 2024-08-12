"""
Read the digit images input and format it for the SOM
"""
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


class DigitsImagesReader:
    """
    Reader for the digits images
    """

    def __init__(self, digits_images_csv_path: Path, digits_images_keys_csv_path: Path, normalize: bool = True):
        """
        :param digits_images_csv_path: csv file with rows representing flattened digit images
        :param digits_images_keys_csv_path: csv file representing the true digits
        :param normalize: should normalize the digit pixels to [0, 1]
        """
        self._digits_images_csv_path = digits_images_csv_path
        self._digits_images_key_csv_path = digits_images_keys_csv_path
        self._normalize = normalize

        self._digits_images: Optional[np.ndarray] = None  # save in-memory
        self._digits_images_keys: Optional[np.array] = None  # save in-memory
        self._joint_digits_images_keys: Optional[np.array] = None  # save in-memory

    def get_digits_images(self) -> np.ndarray:
        """
        :return: the digits image numpy matrix
        """
        if self._digits_images is None:
            print('Reading digits images...')
            self._digits_images = np.loadtxt(self._digits_images_csv_path,
                                             delimiter=',',
                                             dtype=np.uint8 if not self._normalize else float)
            if self._normalize:
                self._digits_images /= 255.0  # the maximum grayscale value
            self._digits_images = np.ascontiguousarray(self._digits_images)
            print('Finished reading digits images!')

        return self._digits_images

    def _get_digits_images_keys(self) -> np.array:
        """
        :return: the digit keys numpy array
        """
        if self._digits_images_keys is None:
            self._digits_images_keys = np.loadtxt(self._digits_images_key_csv_path,
                                                  delimiter=',',
                                                  dtype=np.uint8)
            self._digits_images_keys = np.ascontiguousarray(self._digits_images_keys)

        return self._digits_images_keys

    def __iter__(self) -> Iterator[Tuple[np.uint8, np.array]]:
        """
        Iterator over the digits images, returning key, value tuples
        """
        if self._joint_digits_images_keys is None:
            digits_images = self.get_digits_images()
            digits_images_keys = self._get_digits_images_keys()
            self._joint_digits_images_keys = np.empty(digits_images.shape[0],
                                                      dtype=np.dtype(  # new dtype to synchronize the shuffle
                                                          [('image', digits_images.dtype, digits_images.shape[1]),
                                                           ('key', digits_images_keys.dtype)]))
            self._joint_digits_images_keys['image'] = digits_images
            self._joint_digits_images_keys['key'] = digits_images_keys.flatten()
            self._joint_digits_images_keys = np.ascontiguousarray(self._joint_digits_images_keys)

        np.random.shuffle(self._joint_digits_images_keys)
        for row in self._joint_digits_images_keys:
            yield row['key'], row['image']

    def get_flattened_image_size(self) -> int:
        """
        :return: the flattened image size
        """
        if self._digits_images is None:  # if already read the file
            return self.get_digits_images().shape[-1]
        else:  # need to only read the first line
            first_row = np.genfromtxt(self._digits_images_csv_path, delimiter=',', dtype=np.uint8, max_rows=1)
            return first_row.shape[-1]
