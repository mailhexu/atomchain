import sys
import types

import numpy as np
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from atomchain.training.cli import mlhist_cli, mltraining_cli


def test_mlhist_help_documents_conversion(capsys):
    try:
        mlhist_cli(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "HIST" in out
    assert "--to" in out


def test_mltraining_help_documents_subcommands(capsys):
    try:
        mltraining_cli(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "generate" in out
    assert "artifacts" in out
    assert "train" in out


def test_mltraining_artifacts_help_documents_evaluate(capsys):
    try:
        mltraining_cli(["artifacts", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--evaluate" in out


def test_mlhist_cli_converts_traj_to_hist_and_back(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6)
    )
    traj = tmp_path / "in.traj"
    hist = tmp_path / "out_HIST.nc"
    out = tmp_path / "out.traj"
    write(str(traj), [atoms])
    mlhist_cli([str(traj), str(hist), "--to", "hist"])
    mlhist_cli([str(hist), str(out), "--to", "traj"])
    loaded = read(str(out), ":")
    assert len(loaded) == 1
    assert loaded[0].get_potential_energy() == -1.0


def test_mltraining_train_cli_delegates_to_pymultibinit(monkeypatch, tmp_path):
    module = types.ModuleType("pymultibinit")
    training = types.ModuleType("pymultibinit.training")

    def fake_train_multibinit_model(**kwargs):
        config = tmp_path / "model.conf"
        config.write_text("model", encoding="utf-8")
        return {"model_config": str(config), "output_dir": kwargs["output_dir"]}

    training.train_multibinit_model = fake_train_multibinit_model
    monkeypatch.setitem(sys.modules, "pymultibinit", module)
    monkeypatch.setitem(sys.modules, "pymultibinit.training", training)
    mltraining_cli(
        [
            "train",
            "--ddb",
            "a.ddb",
            "--hist",
            "a_HIST.nc",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "atomchain_training_result.yaml").exists()
