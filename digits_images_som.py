"""
Module containing the SOM algorithm for clustering digits images
"""
import functools
import math
from collections import Counter
from itertools import cycle
from typing import Counter as CounterType, Dict, List, Optional, Tuple

import numpy as np
import scipy
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.digits_images_reader import DigitsImagesReader
from src.digits_images_som_result import _DigitsImagesSOMResult

NEURON_MESH_WIDTH: int = 10  # number of columns in the neuron mesh
NEURON_MESH_HEIGHT: int = 10  # number of rows in the neuron mesh


class DigitsImagesSOM:
    """
    SOM algorithm for clustering digits images
    """
    NEURONS_INIT_PCA_COMPONENTS: int = 10  # number of PCA components to use for initialization

    INIT_NEIGHBORHOOD_RADIUS: float = 4.5  # the initial neighborhood radius
    INIT_LEARNING_RATE: float = 0.2  # the initial learning rate for the weights

    EPOCH_COUNT: int = 50  # the number of epochs to train the model for
    INPUT_ITERATION_RATIO: float = 0.75  # the ratio of the input len to iterate over each epoch

    def __init__(self, digits_images_reader: DigitsImagesReader):
        """
        :param digits_images_reader: Reader object for the digit images
        """
        self._digits_images_reader = digits_images_reader

        self._neurons: Optional[np.ndarray] = None  # neuron vectors mesh

        self._x_indices = np.arange(NEURON_MESH_WIDTH)  # indices for the x values
        self._y_indices = np.arange(NEURON_MESH_HEIGHT)  # indices for the y values
        self._neuron_x_indices, self._neuron_y_indices = np.meshgrid(self._x_indices, self._y_indices)
        self._neuron_x_indices = self._neuron_x_indices.astype(float)
        self._neuron_y_indices = self._neuron_y_indices.astype(float)
        self._neuron_x_indices[::-2] -= 0.5  # hexagonal mapping
        self._neuron_y_indices *= (3.0 / 2.0) / np.sqrt(3)  # hexagonal mapping

        self._quantization_error_history: List[float] = []
        self._topographical_error_history: List[float] = []

        self._epochs_decay_rate = self.EPOCH_COUNT / 3
        self._neighborhood_decay_rate = (self.INIT_NEIGHBORHOOD_RADIUS - 1) / self.EPOCH_COUNT

    def _init_clustering(self):
        """
        Initialise the clustering neurons using PCA components
        """
        train_data = self._digits_images_reader.get_digits_images()
        print(f"Initializing neurons using {self.NEURONS_INIT_PCA_COMPONENTS} PCA components...")
        pca = PCA(n_components=self.NEURONS_INIT_PCA_COMPONENTS)
        train_data_reduced = pca.fit_transform(train_data)

        weights = np.random.rand(NEURON_MESH_HEIGHT * NEURON_MESH_WIDTH, train_data_reduced.shape[1])
        weights = pca.inverse_transform(weights) + pca.mean_
        self._neurons = weights.reshape(NEURON_MESH_HEIGHT, NEURON_MESH_WIDTH, train_data.shape[-1])
        self._neurons = np.ascontiguousarray(self._neurons)
        print(f"Finished initializing neurons!")

    def get_closest_euclidean_neuron(self, to_image: np.array) -> Tuple[int, int]:
        """
        :param to_image: image to calculate the shortest distance from neuron
        :return: 2d indices of the closest neuron
        """
        squared_distances = np.sum((self._neurons - to_image) ** 2, axis=2)

        closest_neuron_flattened_index = np.argmin(squared_distances)
        return divmod(closest_neuron_flattened_index, NEURON_MESH_WIDTH)

    def get_closest_cosine_neuron(self, to_image: np.array) -> Tuple[int, int]:
        """
        :param to_image: image to calculate the shortest distance from neuron
        :return: 2d indices of the closest neuron
        """
        dot_product = np.einsum('ijk,k->ij', self._neurons, to_image)
        neuron_norms = np.linalg.norm(self._neurons, axis=2)
        image_norm = np.linalg.norm(to_image)
        cosine_distances = 1 - (dot_product / (neuron_norms * image_norm))

        closest_neuron_flattened_index = np.argmin(cosine_distances)
        return divmod(closest_neuron_flattened_index, NEURON_MESH_WIDTH)

    @functools.lru_cache(maxsize=NEURON_MESH_WIDTH * NEURON_MESH_HEIGHT)
    def _get_neighborhood_weights(self,
                                  closest_neuron: Tuple[int, int],
                                  iteration_decay: float,
                                  neighborhood_decay: float) -> np.ndarray:
        """
        :param closest_neuron: indices of the closest neuron
        :param iteration_decay: learning rate decay value
        :param neighborhood_decay: neighborhood size decay value
        :return: matrix the same size as self._neurons
        """
        return iteration_decay * self._matrix_gaussian_weights(closest_neuron,
                                                               neighborhood_decay,
                                                               self._neuron_x_indices,
                                                               self._neuron_y_indices)

    def _clustering_step(self, digit_image: np.array, iteration_decay: float, neighborhood_decay: float, epoch: int):
        """
        One epoch of the SOM algorithm
        :param digit_image: input image to run the SOM algorithm on
        :param iteration_decay: epoch learning rate decay value
        :param neighborhood_decay: epoch neighborhood size decay value
        """
        if (epoch // 5) % 2 == 0:
            closest_neuron = self.get_closest_euclidean_neuron(digit_image)
        else:
            closest_neuron = self.get_closest_cosine_neuron(digit_image)

        neighborhood_weights = self._get_neighborhood_weights(closest_neuron, iteration_decay, neighborhood_decay)
        self._neurons += neighborhood_weights[:, :, np.newaxis] * (digit_image - self._neurons)

    def run_clustering(self) -> _DigitsImagesSOMResult:
        """
        runs the clustering algorithm
        :return: Result object of the digits images clustering algorithm
        """
        self._init_clustering()
        input_len = len(self._digits_images_reader.get_digits_images())
        iterations_per_epoch = math.ceil(self.INPUT_ITERATION_RATIO * input_len)  # number of iterations in every epoch

        print("Starting SOM algorithm on the digits images...")
        epoch_tracker = tqdm(range(self.EPOCH_COUNT), desc="Epochs", dynamic_ncols=True)
        for epoch in epoch_tracker:
            iteration_decay = self.INIT_LEARNING_RATE * self._epochs_decay_rate / (
                    (self._epochs_decay_rate + epoch) * 1.8)
            neighborhood_decay = self.INIT_NEIGHBORHOOD_RADIUS / ((1 + (epoch * self._neighborhood_decay_rate)) * 1.5)

            for iteration_number, (_, digit_image) in enumerate(cycle(self._digits_images_reader)):
                if iteration_number >= iterations_per_epoch:
                    break
                self._clustering_step(digit_image, iteration_decay, neighborhood_decay, epoch)

            self._report_error()
            epoch_tracker.set_postfix({"Quantization Error": self._quantization_error_history[-1],
                                       "Topographical Error": self._topographical_error_history[-1]}, refresh=True)
            epoch_tracker.write('', end='')

        print("Finished SOM algorithm on the digits images!\nsee the visualizations :)")
        return _DigitsImagesSOMResult(self._neurons,
                                      self._get_neuron_frequencies(),
                                      self._quantization_error_history,
                                      self._topographical_error_history)

    def _closest_neurons_to_images(self, k: int) -> np.ndarray:
        """
        Returns the k closest neurons to each image
        :param k: number of closest neurons to find
        :return: matrix x where x[i, j] is the j'th closest neuron to image i
        """
        flat_neurons = self._neurons.reshape(-1, self._neurons.shape[2])
        flat_neurons_sq = np.square(flat_neurons).sum(axis=1, keepdims=True)
        digits_images = self._digits_images_reader.get_digits_images()
        digits_images_sq = self._get_digits_images_squared()
        cross_term = scipy.linalg.blas.dgemm(1.0, digits_images, flat_neurons.T)

        closest_neurons = np.argpartition(
            -2 * cross_term + digits_images_sq + flat_neurons_sq.T, k, axis=1
        )[:, :k].squeeze()
        return np.divmod(closest_neurons, NEURON_MESH_WIDTH)

    def _quantization_error(self, closest_neuron_to_image: np.ndarray) -> float:
        """
        :return: quantization error for the neurons
        """
        digits_images = self._digits_images_reader.get_digits_images()
        closest_neuron_image_rows, closest_neuron_image_cols = closest_neuron_to_image[0], closest_neuron_to_image[1]
        quantization = self._neurons[
            closest_neuron_image_rows[:, 0].squeeze(), closest_neuron_image_cols[:, 0].squeeze()
        ]
        return np.linalg.norm(digits_images - quantization, axis=1).mean()

    def _topographical_error(self, closest_two_neurons_to_images: np.ndarray) -> float:
        """
        :return: topographical error for the neurons
        """
        closest_two_neurons_rows, closest_two_neurons_cols = closest_two_neurons_to_images
        hexagonal_closest_neuron_coords = np.stack(
            (
                self._neuron_x_indices[closest_two_neurons_rows[:, 0], closest_two_neurons_cols[:, 0]],
                self._neuron_y_indices[closest_two_neurons_rows[:, 0], closest_two_neurons_cols[:, 0]]
            ),
            axis=-1
        )
        hexagonal_second_closest_neuron_coords = np.stack(
            (
                self._neuron_x_indices[closest_two_neurons_rows[:, 1], closest_two_neurons_cols[:, 1]],
                self._neuron_y_indices[closest_two_neurons_rows[:, 1], closest_two_neurons_cols[:, 1]]
            ),
            axis=-1
        )

        dx = np.abs(hexagonal_closest_neuron_coords[:, 0] - hexagonal_second_closest_neuron_coords[:, 0])
        dy = np.abs(hexagonal_closest_neuron_coords[:, 1] - hexagonal_second_closest_neuron_coords[:, 1])
        topographic_errors = (dx > 1) | (dy > 1) | (np.abs(dx - dy) > 1)
        topographic_error_rate = np.mean(topographic_errors)
        return topographic_error_rate

    def _report_error(self):
        """
        Calculate the quantization and topographical errors, saves them
        """
        closest_neurons_to_images = self._closest_neurons_to_images(2)

        quantization_error = self._quantization_error(closest_neurons_to_images)
        self._quantization_error_history.append(quantization_error)

        topographical_error = self._topographical_error(closest_neurons_to_images)
        self._topographical_error_history.append(topographical_error)

    @staticmethod
    def _matrix_gaussian_weights(center, sigma, x_indices: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
        """
        :param center: center of the gaussian
        :param sigma: sigma of the gaussian
        :param x_indices: mapping for the location of the neurons (x)
        :param y_indices: mapping for the location of the neurons (y)
        :return: weight matrix of the gaussian around the center
        """
        d = 2 * sigma * sigma
        exponent = -(np.square(x_indices - x_indices.T[center]) + np.square(y_indices - y_indices.T[center])) / d
        return np.exp(exponent).T

    @functools.lru_cache
    def _get_digits_images_squared(self):
        """
        Caches the squared result of the digit images
        """
        digits_images = self._digits_images_reader.get_digits_images()
        return np.square(digits_images).sum(axis=1, keepdims=True)

    def _get_neuron_frequencies(self) -> Dict[Tuple[int, int], CounterType[int]]:
        """
        :return: frequencies of the input data mapped to the neurons
        """
        neuron_to_digits: Dict[Tuple[int, int], Counter[int]] = {}
        for digit_image_key, digit_image in self._digits_images_reader:
            closest_neuron = self.get_closest_euclidean_neuron(digit_image)
            if closest_neuron not in neuron_to_digits:
                neuron_to_digits[closest_neuron] = Counter()
            neuron_to_digits[closest_neuron][digit_image_key] += 1

        return neuron_to_digits


"""
I tried also using bubble neighborhood, leaving it here for fun :)

    def _matrix_bubble_weights(self, center, sigma, x_indices: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
        ax = np.logical_and(self._x_indices > center[0] - sigma, self._x_indices < center[0] + sigma)
        ay = np.logical_and(self._y_indices > center[1] - sigma, self._y_indices < center[1] + sigma)
        return np.outer(ay, ax) * 1.
"""
