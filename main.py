import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.digits_images_reader import DigitsImagesReader
from src.digits_images_som import DigitsImagesSOM

DIGITS_FLATTENED_IMAGES_CSV_PATH: Path = Path("../Digits test.csv")
DIGITS_IMAGES_KEYS_CSV_PATH: Path = Path("../Exercise 3 keys.csv")


# TODO: add documentation
def main(digits_images_path: Path, digits_images_keys_path: Path, seed: Optional[int] = None):
    if seed:
        np.random.seed(seed)

    digits_images_reader = DigitsImagesReader(digits_images_path, digits_images_keys_path)
    digits_images_som = DigitsImagesSOM(digits_images_reader)

    som_result = digits_images_som.run_clustering()

    som_result.show_dominant_digit()
    som_result.show_neurons_graphically()


if __name__ == '__main__':
    start_time = time.time()
    main(DIGITS_FLATTENED_IMAGES_CSV_PATH, DIGITS_IMAGES_KEYS_CSV_PATH, 1999)
    print(f"finished running in {time.time() - start_time} seconds")
