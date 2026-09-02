"""Recovered self-organizing-map algorithm with configurable run parameters."""

from __future__ import annotations

import functools
import math
from collections import Counter
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import numpy.typing as npt
import scipy.linalg.blas
from sklearn.decomposition import PCA
from tqdm import tqdm

from .dataset import DigitsDataset, FloatMatrix
from .result import SOMResult


@dataclass(frozen=True)
class SOMConfig:
    """Training parameters for the recovered SOM."""

    width: int = 10
    height: int = 10
    pca_components: int = 10
    initial_neighborhood_radius: float = 4.5
    initial_learning_rate: float = 0.2
    epochs: int = 50
    input_iteration_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("SOM width and height must be at least 2")
        if self.pca_components < 1:
            raise ValueError("pca_components must be positive")
        if self.initial_neighborhood_radius <= 1:
            raise ValueError("initial_neighborhood_radius must be greater than 1")
        if self.initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0 < self.input_iteration_ratio <= 1:
            raise ValueError("input_iteration_ratio must be in (0, 1]")


class SelfOrganizingMap:
    """Cluster flattened images on a two-dimensional hexagonal neuron mesh."""

    def __init__(self, dataset: DigitsDataset, config: SOMConfig | None = None):
        self.dataset = dataset
        self.config = config or SOMConfig()
        self._neurons: FloatMatrix | None = None

        x_indices = np.arange(self.config.width)
        y_indices = np.arange(self.config.height)
        self._neuron_x_indices, self._neuron_y_indices = np.meshgrid(x_indices, y_indices)
        self._neuron_x_indices = self._neuron_x_indices.astype(float)
        self._neuron_y_indices = self._neuron_y_indices.astype(float)
        self._neuron_x_indices[::-2] -= 0.5
        self._neuron_y_indices *= (3.0 / 2.0) / np.sqrt(3)

        self._quantization_errors: list[float] = []
        self._topographical_errors: list[float] = []
        self._epochs_decay_rate = self.config.epochs / 3
        self._neighborhood_decay_rate = (
            self.config.initial_neighborhood_radius - 1
        ) / self.config.epochs

    @property
    def neurons(self) -> FloatMatrix:
        if self._neurons is None:
            raise RuntimeError("the SOM has not been fitted")
        return self._neurons

    def _initialize_neurons(self) -> None:
        training_data = self.dataset.images
        max_components = min(training_data.shape)
        if self.config.pca_components > max_components:
            raise ValueError(f"pca_components cannot exceed {max_components}")

        # Leaving PCA's random_state unset is required for the recovered seed sequence.
        pca = PCA(n_components=self.config.pca_components)
        reduced = pca.fit_transform(training_data)
        weights = np.random.rand(self.config.height * self.config.width, reduced.shape[1])
        weights = pca.inverse_transform(weights) + pca.mean_
        self._neurons = np.ascontiguousarray(
            weights.reshape(
                self.config.height,
                self.config.width,
                training_data.shape[-1],
            )
        )

    def closest_euclidean_neuron(self, image: npt.ArrayLike) -> tuple[int, int]:
        squared_distances = np.sum((self.neurons - image) ** 2, axis=2)
        flat_index = int(np.argmin(squared_distances))
        return divmod(flat_index, self.config.width)

    def closest_cosine_neuron(self, image: npt.ArrayLike) -> tuple[int, int]:
        image_array = np.asarray(image)
        dot_product = np.einsum("ijk,k->ij", self.neurons, image_array)
        neuron_norms = np.linalg.norm(self.neurons, axis=2)
        image_norm = np.linalg.norm(image_array)
        if image_norm == 0:
            return self.closest_euclidean_neuron(image_array)
        denominator = neuron_norms * image_norm
        similarities = np.divide(
            dot_product,
            denominator,
            out=np.full_like(dot_product, -np.inf),
            where=denominator != 0,
        )
        return divmod(int(np.argmax(similarities)), self.config.width)

    # A fitted model owns and explicitly clears this bounded cache between runs.
    @functools.lru_cache(maxsize=100)  # noqa: B019
    def _neighborhood_weights(
        self,
        closest_neuron: tuple[int, int],
        iteration_decay: float,
        neighborhood_decay: float,
    ) -> FloatMatrix:
        return iteration_decay * self._matrix_gaussian_weights(
            closest_neuron,
            neighborhood_decay,
            self._neuron_x_indices,
            self._neuron_y_indices,
        )

    def _clustering_step(
        self,
        image: FloatMatrix,
        iteration_decay: float,
        neighborhood_decay: float,
        epoch: int,
    ) -> None:
        if (epoch // 5) % 2 == 0:
            closest_neuron = self.closest_euclidean_neuron(image)
        else:
            closest_neuron = self.closest_cosine_neuron(image)
        weights = self._neighborhood_weights(
            closest_neuron,
            iteration_decay,
            neighborhood_decay,
        )
        self.neurons[:] += weights[:, :, np.newaxis] * (image - self.neurons)

    def fit(self, *, seed: int | None = None, progress: bool = True) -> SOMResult:
        """Train the SOM and return its quality metrics and label frequencies."""
        if seed is not None:
            np.random.seed(seed)
        self._quantization_errors.clear()
        self._topographical_errors.clear()
        self._neighborhood_weights.cache_clear()
        self._squared_images.cache_clear()
        self.dataset.reset_iteration_order()
        self._initialize_neurons()

        iterations_per_epoch = math.ceil(
            self.config.input_iteration_ratio * self.dataset.image_count
        )
        epoch_iterator = tqdm(
            range(self.config.epochs),
            desc="Epochs",
            dynamic_ncols=True,
            disable=not progress,
        )
        for epoch in epoch_iterator:
            iteration_decay = (
                self.config.initial_learning_rate
                * self._epochs_decay_rate
                / ((self._epochs_decay_rate + epoch) * 1.8)
            )
            neighborhood_decay = self.config.initial_neighborhood_radius / (
                (1 + epoch * self._neighborhood_decay_rate) * 1.5
            )
            for iteration_number, (_, image) in enumerate(cycle(self.dataset)):
                if iteration_number >= iterations_per_epoch:
                    break
                self._clustering_step(image, iteration_decay, neighborhood_decay, epoch)

            self._record_errors()
            epoch_iterator.set_postfix(
                {
                    "Quantization Error": self._quantization_errors[-1],
                    "Topographical Error": self._topographical_errors[-1],
                },
                refresh=True,
            )

        return SOMResult(
            neurons=self.neurons.copy(),
            neuron_frequencies=self._neuron_frequencies(),
            quantization_errors=tuple(self._quantization_errors),
            topographical_errors=tuple(self._topographical_errors),
            seed=seed,
            config=self.config,
        )

    def _closest_neurons_to_images(self, k: int) -> npt.NDArray[np.int_]:
        if k < 1 or k >= self.config.width * self.config.height:
            raise ValueError("k must be smaller than the neuron count")
        flat_neurons = self.neurons.reshape(-1, self.neurons.shape[2])
        flat_neurons_squared = np.square(flat_neurons).sum(axis=1, keepdims=True)
        cross_term = scipy.linalg.blas.dgemm(1.0, self.dataset.images, flat_neurons.T)
        closest = np.argpartition(
            -2 * cross_term + self._squared_images() + flat_neurons_squared.T,
            k,
            axis=1,
        )[:, :k].squeeze()
        return np.asarray(np.divmod(closest, self.config.width))

    def _quantization_error(self, closest: npt.NDArray[np.int_]) -> float:
        rows, columns = closest
        quantized = self.neurons[rows[:, 0].squeeze(), columns[:, 0].squeeze()]
        return float(np.linalg.norm(self.dataset.images - quantized, axis=1).mean())

    def _topographical_error(self, closest: npt.NDArray[np.int_]) -> float:
        rows, columns = closest
        first_coordinates = np.stack(
            (
                self._neuron_x_indices[rows[:, 0], columns[:, 0]],
                self._neuron_y_indices[rows[:, 0], columns[:, 0]],
            ),
            axis=-1,
        )
        second_coordinates = np.stack(
            (
                self._neuron_x_indices[rows[:, 1], columns[:, 1]],
                self._neuron_y_indices[rows[:, 1], columns[:, 1]],
            ),
            axis=-1,
        )
        dx = np.abs(first_coordinates[:, 0] - second_coordinates[:, 0])
        dy = np.abs(first_coordinates[:, 1] - second_coordinates[:, 1])
        errors = (dx > 1) | (dy > 1) | (np.abs(dx - dy) > 1)
        return float(np.mean(errors))

    def _record_errors(self) -> None:
        closest = self._closest_neurons_to_images(2)
        self._quantization_errors.append(self._quantization_error(closest))
        self._topographical_errors.append(self._topographical_error(closest))

    @staticmethod
    def _matrix_gaussian_weights(
        center: tuple[int, int],
        sigma: float,
        x_indices: FloatMatrix,
        y_indices: FloatMatrix,
    ) -> FloatMatrix:
        denominator = 2 * sigma * sigma
        exponent = (
            -(
                np.square(x_indices - x_indices.T[center])
                + np.square(y_indices - y_indices.T[center])
            )
            / denominator
        )
        return np.exp(exponent).T

    # A fitted model owns and explicitly clears this one-entry data cache.
    @functools.lru_cache  # noqa: B019
    def _squared_images(self) -> FloatMatrix:
        return np.square(self.dataset.images).sum(axis=1, keepdims=True)

    def _neuron_frequencies(self) -> dict[tuple[int, int], Counter[int]]:
        frequencies: dict[tuple[int, int], Counter[int]] = {}
        for label, image in self.dataset:
            closest = self.closest_euclidean_neuron(image)
            frequencies.setdefault(closest, Counter())[int(label)] += 1
        return frequencies
