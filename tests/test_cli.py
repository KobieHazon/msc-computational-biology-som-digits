import json

from som_digits.cli import cli


def test_cli_runs_small_headless_experiment(monkeypatch, tmp_path, capsys) -> None:
    images = tmp_path / "images.csv"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "output"
    images.write_text(
        "0,0,0,0\n0,0,255,255\n0,255,0,255\n255,0,255,0\n255,255,0,0\n255,255,255,255\n",
        encoding="utf-8",
    )
    labels.write_text("0\n1\n1\n2\n2\n3\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "som-digits",
            "--images",
            str(images),
            "--labels",
            str(labels),
            "--output-dir",
            str(output),
            "--width",
            "2",
            "--height",
            "2",
            "--pca-components",
            "2",
            "--epochs",
            "2",
            "--seed",
            "0",
            "--no-progress",
        ],
    )

    cli()

    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["seed"] == 0
    assert "wrote 7 artifacts" in capsys.readouterr().out
