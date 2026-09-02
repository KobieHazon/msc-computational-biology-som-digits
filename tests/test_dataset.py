from pathlib import Path

import numpy as np
import pytest

from som_digits.dataset import DigitsDataset

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_supplied_dataset_has_expected_shape_and_range() -> None:
    dataset = DigitsDataset(
        REPOSITORY_ROOT / "data/digits-test.csv",
        REPOSITORY_ROOT / "data/digits-test-keys.csv",
    )

    assert dataset.images.shape == (10_000, 784)
    assert dataset.labels.shape == (10_000,)
    assert dataset.images.flags.c_contiguous
    assert dataset.labels.flags.c_contiguous
    assert np.min(dataset.images) == 0
    assert np.max(dataset.images) == 1
    assert set(dataset.labels.tolist()) == set(range(10))


def test_image_and_label_count_must_match(tmp_path: Path) -> None:
    images = tmp_path / "images.csv"
    labels = tmp_path / "labels.csv"
    images.write_text("0,255\n255,0\n", encoding="utf-8")
    labels.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="number of labels"):
        _ = DigitsDataset(images, labels).labels


@pytest.mark.parametrize("pixel", ["-1", "256", "nan"])
def test_invalid_pixels_are_rejected(tmp_path: Path, pixel: str) -> None:
    images = tmp_path / "images.csv"
    labels = tmp_path / "labels.csv"
    images.write_text(f"0,{pixel}\n", encoding="utf-8")
    labels.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite|between"):
        _ = DigitsDataset(images, labels).images
