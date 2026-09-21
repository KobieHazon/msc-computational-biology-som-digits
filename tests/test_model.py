import hashlib
from pathlib import Path

import numpy as np
import pytest

from som_digits import DigitsDataset, SelfOrganizingMap, SOMConfig

REPOSITORY_ROOT = Path(__file__).parents[1]


def small_dataset(tmp_path: Path) -> DigitsDataset:
    images = tmp_path / "images.csv"
    labels = tmp_path / "labels.csv"
    images.write_text(
        "0,0,0,0\n0,0,255,255\n0,255,0,255\n255,0,255,0\n255,255,0,0\n255,255,255,255\n",
        encoding="utf-8",
    )
    labels.write_text("0\n1\n1\n2\n2\n3\n", encoding="utf-8")
    return DigitsDataset(images, labels)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 1}, "width"),
        ({"pca_components": 0}, "pca_components"),
        ({"epochs": 0}, "epochs"),
        ({"input_iteration_ratio": 0}, "input_iteration_ratio"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SOMConfig(**kwargs)


def test_small_run_is_deterministic_and_uses_labels_only_for_evaluation(tmp_path: Path) -> None:
    config = SOMConfig(width=2, height=2, pca_components=2, epochs=3, input_iteration_ratio=0.5)
    model = SelfOrganizingMap(small_dataset(tmp_path), config)

    first = model.fit(seed=0, progress=False)
    second = model.fit(seed=0, progress=False)

    np.testing.assert_array_equal(first.neurons, second.neurons)
    assert first.quantization_errors == second.quantization_errors
    assert first.topographical_errors == second.topographical_errors
    assert first.assigned_images == 6
    assert 0 <= first.majority_label_purity <= 1


@pytest.mark.integration
def test_full_supplied_dataset_matches_seed_2024_baseline() -> None:
    dataset = DigitsDataset(
        REPOSITORY_ROOT / "data/digits-test.csv",
        REPOSITORY_ROOT / "data/digits-test-keys.csv",
    )

    result = SelfOrganizingMap(dataset).fit(seed=2024, progress=False)
    summary = result.summary()

    assert summary["assigned_images"] == 10_000
    assert summary["occupied_neurons"] == 100
    assert summary["majority_label_purity"] == pytest.approx(0.8107)
    assert summary["quantization_error_first"] == pytest.approx(6.620086159310609)
    assert summary["quantization_error_last"] == pytest.approx(5.474670576851817)
    assert summary["topographical_error_first"] == pytest.approx(0.1074)
    assert summary["topographical_error_last"] == pytest.approx(0.1861)
    # Preserve the original baseline identity, then compare every value with
    # a tight tolerance for BLAS/platform floating-point rounding.
    baseline = np.load(REPOSITORY_ROOT / "tests/fixtures/neurons.npy", allow_pickle=False)
    assert hashlib.sha256(baseline.tobytes()).hexdigest() == (
        "5352fa640de82610d049801fc8846f4a6474fc1856feefc94461056fdff607a0"
    )
    np.testing.assert_allclose(result.neurons, baseline, rtol=1e-12, atol=1e-12)
    history = np.loadtxt(REPOSITORY_ROOT / "tests/fixtures/metrics.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(result.quantization_errors, history[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.topographical_errors, history[:, 2])
