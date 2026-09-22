"""Serializable metrics and visualizations for a trained SOM."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import patches

if TYPE_CHECKING:
    from .model import SOMConfig


@dataclass(frozen=True)
class SOMResult:
    """A trained neuron mesh plus evaluation-only label statistics."""

    neurons: npt.NDArray[np.float64]
    neuron_frequencies: dict[tuple[int, int], Counter[int]]
    quantization_errors: tuple[float, ...]
    topographical_errors: tuple[float, ...]
    seed: int | None
    config: SOMConfig

    @property
    def assigned_images(self) -> int:
        return sum(sum(counter.values()) for counter in self.neuron_frequencies.values())

    @property
    def majority_label_purity(self) -> float:
        if not self.assigned_images:
            return 0.0
        majority_count = sum(
            counter.most_common(1)[0][1] for counter in self.neuron_frequencies.values()
        )
        return majority_count / self.assigned_images

    def _array_hash(self, values: npt.ArrayLike) -> str:
        return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "config": asdict(self.config),
            "assigned_images": self.assigned_images,
            "occupied_neurons": len(self.neuron_frequencies),
            "majority_label_purity": self.majority_label_purity,
            "quantization_error_first": self.quantization_errors[0],
            "quantization_error_last": self.quantization_errors[-1],
            "topographical_error_first": self.topographical_errors[0],
            "topographical_error_last": self.topographical_errors[-1],
            "neurons_sha256": self._array_hash(self.neurons),
            "quantization_history_sha256": self._array_hash(self.quantization_errors),
            "topographical_history_sha256": self._array_hash(self.topographical_errors),
        }

    def write_artifacts(self, output_directory: Path, *, show: bool = False) -> list[Path]:
        output_directory.mkdir(parents=True, exist_ok=True)
        paths = [
            self._write_summary(output_directory),
            self._write_metrics(output_directory),
            self._write_neurons(output_directory),
            self._plot_confidence_map(output_directory),
            self._plot_dominant_digit_map(output_directory),
            self._plot_neuron_prototypes(output_directory),
            self._plot_errors(output_directory),
        ]
        if show:
            plt.show()
        plt.close("all")
        return paths

    def _write_summary(self, output_directory: Path) -> Path:
        path = output_directory / "summary.json"
        path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _write_metrics(self, output_directory: Path) -> Path:
        path = output_directory / "metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as metrics_file:
            writer = csv.writer(metrics_file, lineterminator="\n")
            writer.writerow(["epoch", "quantization_error", "topographical_error"])
            writer.writerows(
                (epoch, quantization, topographical)
                for epoch, (quantization, topographical) in enumerate(
                    zip(self.quantization_errors, self.topographical_errors, strict=True),
                    start=1,
                )
            )
        return path

    def _write_neurons(self, output_directory: Path) -> Path:
        path = output_directory / "neurons.npy"
        np.save(path, self.neurons, allow_pickle=False)
        return path

    @staticmethod
    def _new_hex_figure(cells: npt.NDArray[np.object_]) -> tuple[plt.Figure, plt.Axes]:
        rows, columns = cells.shape[:2]
        figure, axes = plt.subplots(figsize=(12, 10))
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_xlim(-2, columns * 1.5 + 2)
        axes.set_ylim(-math.sqrt(3), rows * math.sqrt(3) + math.sqrt(3))
        return figure, axes

    @staticmethod
    def _hexagon(row: int, column: int) -> tuple[patches.RegularPolygon, float, float]:
        x = column * 1.5
        y = row * math.sqrt(3) + (column % 2) * math.sqrt(3) / 2
        hexagon = patches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=1,
            orientation=np.radians(30),
            edgecolor="black",
            linewidth=0.5,
        )
        return hexagon, x, y

    def _plot_confidence_map(self, output_directory: Path) -> Path:
        shape = self.neurons.shape[:2]
        content = np.full(shape, "None\n0%", dtype=object)
        confidence = np.zeros(shape)
        for (row, column), counter in self.neuron_frequencies.items():
            label, count = counter.most_common(1)[0]
            fraction = count / sum(counter.values())
            content[row, column] = f"{label}\n{math.ceil(100 * fraction)}%"
            confidence[row, column] = fraction

        figure, axes = self._new_hex_figure(content)
        colormap = matplotlib.colormaps["Blues"]
        for row in range(shape[0]):
            for column in range(shape[1]):
                hexagon, x, y = self._hexagon(row, column)
                hexagon.set_facecolor(colormap(confidence[row, column]))
                axes.add_patch(hexagon)
                axes.text(x, y, content[row, column], ha="center", va="center", fontsize=7)
        axes.set_title("Dominant Digit Confidence")
        return self._save_figure(figure, output_directory / "confidence-map.png")

    def _plot_dominant_digit_map(self, output_directory: Path) -> Path:
        shape = self.neurons.shape[:2]
        digits = np.full(shape, -1, dtype=int)
        for coordinates, counter in self.neuron_frequencies.items():
            digits[coordinates] = counter.most_common(1)[0][0]

        figure, axes = self._new_hex_figure(digits.astype(object))
        colormap = matplotlib.colormaps["viridis"]
        for row in range(shape[0]):
            for column in range(shape[1]):
                hexagon, x, y = self._hexagon(row, column)
                value = digits[row, column]
                hexagon.set_facecolor(colormap(max(value, 0) / 9))
                axes.add_patch(hexagon)
                axes.text(x, y, str(value) if value >= 0 else "-", ha="center", va="center")
        axes.set_title("Dominant Digit by Neuron")
        return self._save_figure(figure, output_directory / "dominant-digit-map.png")

    def _plot_neuron_prototypes(self, output_directory: Path) -> Path:
        image_side = math.isqrt(self.neurons.shape[2])
        if image_side * image_side != self.neurons.shape[2]:
            raise ValueError("neuron vectors cannot be reshaped into square images")
        figure, axes = plt.subplots(
            self.neurons.shape[0],
            self.neurons.shape[1],
            figsize=(12, 12),
        )
        axes_array = np.atleast_2d(axes)
        for row in range(self.neurons.shape[0]):
            for column in range(self.neurons.shape[1]):
                axes_array[row, column].imshow(
                    self.neurons[row, column].reshape(image_side, image_side),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes_array[row, column].axis("off")
        figure.suptitle("Neuron Prototypes")
        return self._save_figure(figure, output_directory / "neuron-prototypes.png")

    def _plot_errors(self, output_directory: Path) -> Path:
        figure, (quantization_axes, topographical_axes) = plt.subplots(2, 1, figsize=(10, 8))
        epochs = np.arange(1, len(self.quantization_errors) + 1)
        quantization_axes.plot(epochs, self.quantization_errors)
        quantization_axes.set(title="Quantization Error", xlabel="Epoch", ylabel="Mean distance")
        quantization_axes.grid(alpha=0.25)
        topographical_axes.plot(epochs, self.topographical_errors)
        topographical_axes.set(title="Topographical Error", xlabel="Epoch", ylabel="Fraction")
        topographical_axes.grid(alpha=0.25)
        figure.tight_layout()
        return self._save_figure(figure, output_directory / "training-errors.png")

    @staticmethod
    def _save_figure(figure: plt.Figure, path: Path) -> Path:
        figure.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
            metadata={"Software": "msc-computational-biology-som-digits"},
        )
        return path
