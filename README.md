# Self-Organizing Map for Handwritten Digits

A 2024 CS MSc Computational Biology project that implements a self-organizing map from first principles and trains it on 10,000 supplied 28-by-28 grayscale digit images. A 10-by-10 hexagonal neuron mesh learns unlabeled image prototypes; labels are introduced only after training to evaluate cluster confidence and visualize the dominant digit assigned to each neuron.

## Tech Stack

- Python 3.10 or 3.11
- NumPy and SciPy BLAS for vectorized distance calculations
- scikit-learn PCA for data-informed neuron initialization
- Matplotlib for hexagonal confidence maps, prototype images, and error curves
- tqdm for progress reporting
- pytest, Ruff, and uv for validation and reproducibility

The numerical dependencies are intentionally pinned to the recovered 2024 versions. The PCA solver, global NumPy random sequence, initialization, and repeated shuffling all affect the exact stochastic trajectory.

## Algorithm

The implementation:

1. Normalizes the supplied grayscale pixels to `[0, 1]`.
2. Projects the 784-dimensional images onto 10 principal components and initializes 100 neuron vectors through the inverse PCA transform.
3. Trains for 50 epochs over a shuffled 75% sample of the input.
4. Alternates blocks of Euclidean and cosine best-matching-unit selection.
5. Updates the winning neuron and its hexagonal neighborhood using a decaying Gaussian weight and learning rate.
6. Records quantization and topographical error after every epoch.
7. Uses the answer-key labels only after training to calculate majority-label purity and visualization labels.

This is a clustering and visualization model, not a supervised digit classifier. Majority-label purity describes the dominant label in each occupied neuron and is not held-out classification accuracy.

## Setup

```bash
git clone https://github.com/KobieHazon/msc-computational-biology-som-digits.git
cd msc-computational-biology-som-digits
uv sync --dev
```

## Usage

Run the complete reproducibility regression:

```bash
uv run som-digits --seed 2024 --no-progress
```

The validated run takes about 41 seconds on the validation machine. It occupies all 100 neurons, reaches majority-label purity `0.8107`, and changes quantization error from `6.620086` to `5.474671`. Topographical error changes from `0.1074` to `0.1861`; the final increase is retained as part of the recovered behavior rather than presented as an improvement.

The command writes:

- `summary.json`: parameters, metrics, and SHA-256 hashes of numerical results
- `metrics.csv`: quantization and topographical error by epoch
- `neurons.npy`: the trained neuron vectors without pickle serialization
- `confidence-map.png`: dominant label and confidence per neuron
- `dominant-digit-map.png`: color-coded dominant labels
- `neuron-prototypes.png`: learned 28-by-28 neuron images
- `training-errors.png`: both error histories

Use `--show` for interactive figures. The mesh dimensions, PCA components, epoch count, and training ratio are configurable; changing them intentionally leaves the historical regression path.

## Testing

Fast development checks:

```bash
uv run pytest -m "not integration"
uv run ruff check .
uv run ruff format --check .
```

Complete supplied-dataset regression:

```bash
uv run pytest -m integration
```

The integration test verifies the original seed-`2024` neuron and error-history hashes, not only rounded headline metrics.

## Repository Structure

- `assignment/exercise.pdf`: supplied exercise brief converted to a PDF
- `data/`: supplied 10,000-image CSV and evaluation-only label CSV
- `src/som_digits/`: input validation, SOM implementation, CLI, metrics, and visualization
- `tests/`: fast behavior tests and complete-data numerical regression

## Implementation notes

The maintained version adds packaging, explicit deterministic execution, output files, validation, and documentation without replacing the original algorithm.

No coauthored report or historical result files were recovered. The metrics above come from a regression-test seed and are identified accordingly. The local MiniSom checkout was not imported by the solution and is an unrelated third-party reference repository, so it is not included.

## Authorship

Solution by Kobie Hazon and Daniel Ben Zion. The exercise and datasets are identified as supplied material solely to distinguish them from the coauthored solution; this label does not assert ownership by a university or any other institution.

No repository-wide license is declared because the repository combines original work with supplied material whose reuse terms were not recorded.
