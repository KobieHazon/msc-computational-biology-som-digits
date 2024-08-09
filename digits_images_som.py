import functools
from collections import Counter
from typing import Counter as CounterType, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.digits_images_reader import DigitsImagesReader
from src.utils import matrix_gaussian_weights, plot_frequency_matrix


class _DigitsImagesSOMResult:
    def __init__(self,
                 neurons: np.ndarray,
                 som_instance: 'DigitsImagesSOM',
                 digits_images_reader: DigitsImagesReader):
        self._neurons = neurons
        self._som_instance = som_instance
        self._digits_images_reader = digits_images_reader

    @functools.lru_cache
    def _get_neuron_frequencies(self) -> Dict[Tuple[int, int], CounterType[int]]:
        neuron_to_digits: Dict[Tuple[int, int], Counter[int]] = {}
        for digit_image_key, digit_image in self._digits_images_reader:
            closest_neuron = self._som_instance.get_closest_neuron(digit_image)
            if closest_neuron not in neuron_to_digits:
                neuron_to_digits[closest_neuron] = Counter()
            neuron_to_digits[closest_neuron][digit_image_key] += 1

        return neuron_to_digits

    def show_dominant_digit(self):  # TODO: add color mapping based on the percentage value to see the clusters better
        neuron_to_digits = self._get_neuron_frequencies()
        plot_frequency_matrix(neuron_to_digits, self._neurons.shape[:2])

    def show_neurons_graphically(self):
        def reshape_to_image(cell):
            return cell[:28 * 28].reshape(28, 28) * 255.0

        table_height, table_width = self._neurons.shape[:2]
        fig, axes = plt.subplots(table_height, table_width, figsize=(15, 15))
        for i in range(table_height):
            for j in range(table_width):
                image = reshape_to_image(self._neurons[i, j])
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].axis('off')

        plt.show()

    # TODO: add another colormap that shows the different dominant digits in different distinct colors


class DigitsImagesSOM:
    # TODO: optimize parameters
    NEURON_MESH_WIDTH: int = 10
    NEURON_MESH_HEIGHT: int = 10
    NEIGHBORHOOD_SIZE: float = 5  # TODO: parameter for report

    # TODO: find out if our implementation is suitable for epochs or not
    TRAIN_ITERATIONS: int = 10  # TODO: parameter for report
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

    def get_closest_neuron(self, to_image: np.array) -> Tuple[int, int]:
        neuron_distances = np.linalg.norm(np.subtract(to_image, self._neurons), axis=-1, ord=2)
        closest_neuron_flattened_index = np.argmin(neuron_distances)
        closest_neuron_index = np.unravel_index(closest_neuron_flattened_index,
                                                (self.NEURON_MESH_WIDTH, self.NEURON_MESH_HEIGHT))
        closest_neuron_index = (int(closest_neuron_index[0]), int(closest_neuron_index[1]))
        return closest_neuron_index

    def _clustering_step(self, digit_image: np.array, iteration_decay: float, neighborhood_decay: float):
        closest_neuron = self.get_closest_neuron(digit_image)

        neighborhood_weights = iteration_decay * matrix_gaussian_weights(closest_neuron,
                                                                         neighborhood_decay,
                                                                         self._neuron_x_indices,
                                                                         self._neuron_y_indices)  # TODO: parameter for report(gaussian)

        neuron_difference = digit_image - self._neurons
        self._neurons += neighborhood_weights[:, :, np.newaxis] * neuron_difference  # TODO: parameter for report

    def run_clustering(self) -> _DigitsImagesSOMResult:
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

        return _DigitsImagesSOMResult(self._neurons, self, self._digits_images_reader)

    def _report_error(self) -> float:  # TODO: implement with both quantization and topological errors
        pass
