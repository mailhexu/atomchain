import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from atomchain import training_set_generation as tsg


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = -float(len(atoms))
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.zeros(6)


def test_rattle_frames_are_deterministic_and_do_not_mutate_parent():
    parent = bulk("Al", cubic=True)
    original = parent.positions.copy()
    context = tsg.SamplingContext(parent_atoms=parent, seed=7)

    first = tsg.generate_rattle_frames(
        context, tsg.RattleSamplerConfig(count=2, stdev=0.01, cell_stdev=0.001)
    )
    second = tsg.generate_rattle_frames(
        context, tsg.RattleSamplerConfig(count=2, stdev=0.01, cell_stdev=0.001)
    )

    assert len(first) == 2
    np.testing.assert_allclose(parent.positions, original)
    np.testing.assert_allclose(first[0].atoms.positions, second[0].atoms.positions)
    assert first[0].source == "rattle"
    assert first[0].metadata["seed"] == 7
    assert "strain" in first[0].metadata


def test_metastable_interpolation_supports_lambda_three():
    parent = bulk("Al", cubic=True)
    metastable = parent.copy()
    metastable.info["metastable_id"] = "m1"
    metastable.set_scaled_positions(parent.get_scaled_positions() + [0.1, 0.0, 0.0])
    context = tsg.SamplingContext(
        parent_atoms=parent, metastable_structures=[metastable]
    )

    frames = tsg.generate_metastable_interpolation_frames(
        context, tsg.MetastableInterpolationConfig(lambdas=(0.0, 1.0, 3.0))
    )

    assert [frame.metadata["lambda"] for frame in frames] == [0.0, 1.0, 3.0]
    np.testing.assert_allclose(frames[0].atoms.positions, parent.positions)
    np.testing.assert_allclose(frames[1].atoms.positions, metastable.positions)
    assert frames[2].metadata["metastable_id"] == "m1"


def test_metastable_interpolation_supports_exact_count():
    parent = bulk("Al", cubic=True)
    metastable_a = parent.copy()
    metastable_b = parent.copy()
    metastable_a.info["metastable_id"] = "m1"
    metastable_b.info["metastable_id"] = "m2"
    metastable_a.set_scaled_positions(parent.get_scaled_positions() + [0.1, 0.0, 0.0])
    metastable_b.set_scaled_positions(parent.get_scaled_positions() + [0.0, 0.1, 0.0])
    context = tsg.SamplingContext(
        parent_atoms=parent, metastable_structures=[metastable_a, metastable_b]
    )

    frames = tsg.generate_metastable_interpolation_frames(
        context, tsg.MetastableInterpolationConfig(count=1000, lambdas=(0.0, 3.0))
    )

    assert len(frames) == 1000
    assert frames[0].metadata["lambda"] == pytest.approx(0.0)
    assert frames[-1].metadata["lambda"] == pytest.approx(3.0)
    assert {frame.metadata["metastable_id"] for frame in frames} == {"m1", "m2"}
    assert frames[-1].frame_id == "metastable_interpolation-0999"


def test_metastable_interpolation_skips_incompatible_structures():
    parent = bulk("Al")
    incompatible = bulk("Cu", cubic=True)
    context = tsg.SamplingContext(
        parent_atoms=parent, metastable_structures=[incompatible]
    )

    assert tsg.generate_metastable_interpolation_frames(context, {}) == []

    with pytest.raises(ValueError, match="atom count"):
        tsg.generate_metastable_interpolation_frames(
            context, tsg.MetastableInterpolationConfig(skip_incompatible=False)
        )


def test_source_dispatch_rejects_unknown_sampler():
    with pytest.raises(ValueError, match="Unknown sampler"):
        tsg.generate_frames_from_source(
            tsg.SamplingContext(parent_atoms=bulk("Al")), {"name": "missing"}
        )


class FakeDynamics:
    def __init__(self, atoms, *args, **kwargs):
        self.atoms = atoms
        self.callbacks = []
        self.kwargs = kwargs

    def attach(self, callback, interval=1):
        self.callbacks.append((callback, interval))

    def run(self, steps):
        for step in range(1, steps + 1):
            self.nsteps = step
            self.atoms.positions += 0.001
            for callback, interval in self.callbacks:
                if step % interval == 0:
                    callback()


class FakeDynamicsWithInitialCallback(FakeDynamics):
    def attach(self, callback, interval=1):
        self.nsteps = 0
        callback()
        super().attach(callback, interval=interval)


def test_contour_sampler_collects_mocked_dynamics(monkeypatch):
    monkeypatch.setattr(tsg, "_contour_exploration_class", lambda: FakeDynamics)
    context = tsg.SamplingContext(
        parent_atoms=bulk("Al"), seed=3, calculator=ConstantCalculator()
    )

    frames = tsg.generate_contour_frames(
        context, tsg.ContourSamplerConfig(steps=4, sample_interval=2)
    )

    assert len(frames) == 2
    assert frames[0].source == "contour"
    assert frames[0].metadata["step"] == 2
    assert frames[0].metadata["sampler"] == "contour"


def test_contour_sampler_skips_ase_initial_observer_callback(monkeypatch):
    monkeypatch.setattr(
        tsg, "_contour_exploration_class", lambda: FakeDynamicsWithInitialCallback
    )
    context = tsg.SamplingContext(
        parent_atoms=bulk("Al"), seed=3, calculator=ConstantCalculator()
    )

    frames = tsg.generate_contour_frames(
        context, tsg.ContourSamplerConfig(steps=4, sample_interval=2)
    )

    assert len(frames) == 2
    assert [frame.metadata["step"] for frame in frames] == [2, 4]


def test_contour_sampler_requires_calculator():
    with pytest.raises(ValueError, match="requires a calculator"):
        tsg.generate_contour_frames(tsg.SamplingContext(parent_atoms=bulk("Al")), {})


def test_npt_sampler_collects_mocked_dynamics(monkeypatch):
    monkeypatch.setattr(tsg, "_npt_berendsen_class", lambda: FakeDynamics)
    context = tsg.SamplingContext(
        parent_atoms=bulk("Al", cubic=True), seed=3, calculator=ConstantCalculator()
    )

    frames = tsg.generate_npt_md_frames(
        context, tsg.NptMdSamplerConfig(steps=6, sample_interval=3, temperature=200.0)
    )

    assert len(frames) == 2
    assert frames[0].source == "npt_md"
    assert frames[0].metadata["ensemble"] == "npt_berendsen"
    assert frames[0].metadata["temperature"] == pytest.approx(200.0)
