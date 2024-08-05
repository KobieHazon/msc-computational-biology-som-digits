import math
from collections import Counter
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.digits_images_reader import DigitsImagesReader
from src.utils import matrix_gaussian_weights


class DigitsImagesSOM:
    # TODO: optimize parameters
    NEURON_MESH_WIDTH: int = 10
    NEURON_MESH_HEIGHT: int = 10
    NEIGHBORHOOD_SIZE: float = 5  # TODO: parameter for report

    # TODO: find out if our implementation is suitable for epochs or not
    TRAIN_ITERATIONS: int = 50  # TODO: parameter for report
    LEARNING_RATE: float = 0.5  # TODO: parameter for report

    def __init__(self, digits_images_reader: DigitsImagesReader):
        self._digits_images_reader = digits_images_reader

        self._neurons: Optional[np.ndarray] = None
        self._neuron_x_indices, self._neuron_y_indices = np.meshgrid(
            np.arange(self.NEURON_MESH_WIDTH), np.arange(self.NEURON_MESH_HEIGHT)
        )
        self._neuron_x_indices, self._neuron_y_indices = (
            self._neuron_x_indices.astype(float), self._neuron_y_indices.astype(float)
        )
        self._neuron_x_indices[::-2] -= 0.5
        self._neuron_y_indices *= (3.0 / 2.0) / np.sqrt(3)

    def _init_clustering(self):
        sample_size = self._digits_images_reader.get_flattened_image_size()
        self._neurons = np.random.rand(self.NEURON_MESH_WIDTH, self.NEURON_MESH_HEIGHT, sample_size)
        # TODO: should we normalize the weights?

    def _get_closest_neuron(self, to_image: np.array) -> Tuple:
        neuron_distances = np.linalg.norm(np.subtract(to_image, self._neurons), axis=-1, ord=2)
        closest_neuron_flattened_index = np.argmin(neuron_distances)
        closest_neuron_index = np.unravel_index(closest_neuron_flattened_index,
                                                (self.NEURON_MESH_WIDTH, self.NEURON_MESH_HEIGHT))
        return closest_neuron_index

    def _clustering_step(self, digit_image: np.array, iteration_decay: float, neighborhood_decay: float):
        closest_neuron = self._get_closest_neuron(digit_image)

        neighborhood_weights = iteration_decay * matrix_gaussian_weights(closest_neuron,
                                                                         neighborhood_decay,
                                                                         self._neuron_x_indices,
                                                                         self._neuron_y_indices)  # TODO: parameter for report(gaussian)

        neuron_difference = digit_image - self._neurons
        self._neurons += neighborhood_weights[:, :, np.newaxis] * neuron_difference  # TODO: parameter for report

    def run_clustering(self) -> np.ndarray:  # TODO: export the result to its own type with analyze logic in it
        self._init_clustering()

        for iteration in range(self.TRAIN_ITERATIONS):
            print(f"starting train iteration {iteration}")
            iteration_decay = self.LEARNING_RATE / (
                    1 + iteration * (100 / self.TRAIN_ITERATIONS))  # TODO: parameter for report
            neighborhood_decay = (self.NEIGHBORHOOD_SIZE /
                                  (1 + (iteration * (
                                          self.NEIGHBORHOOD_SIZE - 1) / self.TRAIN_ITERATIONS)))  # TODO: parameter for report

            for _, digit_image in self._digits_images_reader:
                self._clustering_step(digit_image, iteration_decay, neighborhood_decay)

        return self._neurons

    def show_dominant_digit(self):  # TODO: add color mapping based on the percentage value to see the clusters better
        neuron_to_digits: Dict[Tuple[int, int], Counter[int]] = {}
        for digit_image_key, digit_image in self._digits_images_reader:
            closest_neuron = self._get_closest_neuron(digit_image)
            if closest_neuron not in neuron_to_digits:
                neuron_to_digits[closest_neuron] = Counter()
            neuron_to_digits[closest_neuron][digit_image_key] += 1

        neurons_common_digit_string = np.empty((self.NEURON_MESH_HEIGHT, self.NEURON_MESH_WIDTH), dtype=object)
        for (row, col), counter in neuron_to_digits.items():
            most_common_element, count = counter.most_common(1)[0]
            neurons_common_digit_string[row, col] = \
                f"{int(most_common_element)}({math.ceil(100 * count / sum(counter.values()))}%)"

        fig, ax = plt.subplots()
        ax.axis('off')
        table = ax.table(cellText=neurons_common_digit_string, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.show()

    def show_neurons_graphically(self):
        def reshape_to_image(cell):
            return cell[:28 * 28].reshape(28, 28) * 255.0

        fig, axes = plt.subplots(self.NEURON_MESH_HEIGHT, self.NEURON_MESH_WIDTH, figsize=(15, 15))
        for i in range(self.NEURON_MESH_HEIGHT):
            for j in range(self.NEURON_MESH_WIDTH):
                image = reshape_to_image(self._neurons[i, j])
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].axis('off')

        plt.show()

    def _report_error(self) -> float:  # TODO: implement with both quantization and topological errors
        pass
