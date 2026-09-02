import json
from collections import Counter

import matplotlib
import numpy as np

from som_digits.model import SOMConfig
from som_digits.result import SOMResult

matplotlib.use("Agg")


def sample_result() -> SOMResult:
    neurons = np.arange(16, dtype=float).reshape(2, 2, 4) / 15
    return SOMResult(
        neurons=neurons,
        neuron_frequencies={(0, 0): Counter({1: 2}), (1, 1): Counter({2: 1, 3: 1})},
        quantization_errors=(2.0, 1.0),
        topographical_errors=(0.5, 0.25),
        seed=7,
        config=SOMConfig(width=2, height=2, pca_components=2, epochs=2),
    )


def test_artifacts_are_complete_and_machine_readable(tmp_path) -> None:
    result = sample_result()

    paths = result.write_artifacts(tmp_path)

    assert len(paths) == 7
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["majority_label_purity"] == 0.75
    np.testing.assert_array_equal(np.load(tmp_path / "neurons.npy"), result.neurons)
    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines() == [
        "epoch,quantization_error,topographical_error",
        "1,2.0,0.5",
        "2,1.0,0.25",
    ]
