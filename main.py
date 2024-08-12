"""
Main module for running the SOM algorithm on the digit images problem
"""
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.digits_images_reader import DigitsImagesReader
from src.digits_images_som import DigitsImagesSOM

DIGITS_FLATTENED_IMAGES_CSV_PATH: Path = Path("../Digits test.csv")
DIGITS_IMAGES_KEYS_CSV_PATH: Path = Path("../Exercise 3 keys.csv")


def main(digits_images_path: Path, digits_images_keys_path: Path, seed: Optional[int] = None):
    """
    :param digits_images_path: path to the digit images csv file
    :param digits_images_keys_path: path to the digit images answer-key csv file
    :param seed: seed to use for reproducibility
    """
    if seed:
        np.random.seed(seed)

    digits_images_reader = DigitsImagesReader(digits_images_path, digits_images_keys_path)
    digits_images_som = DigitsImagesSOM(digits_images_reader)

    som_result = digits_images_som.run_clustering()

    som_result.show_digit_confidence_matrix()
    som_result.show_dominant_digit_matrix()
    som_result.show_neurons_matrix_graphically()
    som_result.plot_quantization_errors()
    som_result.plot_topographical_errors()


if __name__ == '__main__':
    main(DIGITS_FLATTENED_IMAGES_CSV_PATH, DIGITS_IMAGES_KEYS_CSV_PATH)
