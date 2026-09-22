"""Command-line interface for the handwritten-digit SOM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DigitsDataset
from .model import SelfOrganizingMap, SOMConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/digits-test.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/digits-test-keys.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("run-results"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--input-ratio", type=float, default=0.75)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def cli() -> None:
    arguments = build_parser().parse_args()
    try:
        config = SOMConfig(
            width=arguments.width,
            height=arguments.height,
            pca_components=arguments.pca_components,
            epochs=arguments.epochs,
            input_iteration_ratio=arguments.input_ratio,
        )
        result = SelfOrganizingMap(
            DigitsDataset(arguments.images, arguments.labels),
            config,
        ).fit(seed=arguments.seed, progress=not arguments.no_progress)
        paths = result.write_artifacts(arguments.output_dir, show=arguments.show)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    print(f"wrote {len(paths)} artifacts to {arguments.output_dir}")


if __name__ == "__main__":
    cli()
