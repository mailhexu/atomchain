"""Training-set structure generation strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from ase import Atoms, units
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

Split = Literal["train", "test"]


@dataclass
class GeneratedFrame:
    """Generated structure with sampler provenance."""

    atoms: Atoms
    source: str
    frame_id: str
    split: Split | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingContext:
    """Shared context passed to structure samplers."""

    parent_atoms: Atoms
    seed: int = 1234
    calculator: Any | None = None
    output_dir: Path | None = None
    metastable_records: list[dict[str, Any]] = field(default_factory=list)
    metastable_structures: list[Atoms] = field(default_factory=list)


@dataclass
class RattleSamplerConfig:
    count: int = 1
    stdev: float = 0.05
    cell_stdev: float | None = None
    split: Split = "train"
    source: str = "rattle"


@dataclass
class MetastableInterpolationConfig:
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    count: int | None = None
    split: Split = "train"
    source: str = "metastable_interpolation"
    skip_incompatible: bool = True


@dataclass
class ContourSamplerConfig:
    steps: int = 100
    sample_interval: int = 10
    count: int | None = None
    split: Split = "train"
    source: str = "contour"
    maxstep: float = 0.5
    parallel_drift: float = 0.1
    energy_target: float | None = None
    angle_limit: float | None = 20.0
    potentiostat_step_scale: float | None = None
    remove_translation: bool = False
    use_frenet_serret: bool = True
    initialization_step_scale: float = 0.01
    use_target_shift: bool = True
    target_shift_previous_steps: int = 10
    use_tangent_curvature: bool = False
    force_consistent: bool | None = None
    trajectory: str | None = None
    logfile: str | None = None
    append_trajectory: bool = False
    loginterval: int = 1


@dataclass
class NptMdSamplerConfig:
    steps: int = 100
    sample_interval: int = 10
    split: Split = "train"
    source: str = "npt_md"
    timestep: float = 1.0
    temperature: float = 300.0
    pressure: float = 1.01325
    taut: float = 100.0
    taup: float = 1000.0
    compressibility: float = 4.57e-5
    trajectory: str | None = None
    logfile: str | None = None
    loginterval: int = 1


def generate_rattle_frames(
    context: SamplingContext, config: RattleSamplerConfig | Mapping[str, Any]
) -> list[GeneratedFrame]:
    """Generate randomly displaced structures."""
    cfg = _coerce_config(config, RattleSamplerConfig)
    _validate_count(cfg.count, "count")
    _validate_nonnegative(cfg.stdev, "stdev")
    if cfg.cell_stdev is not None:
        _validate_nonnegative(cfg.cell_stdev, "cell_stdev")
    rng = np.random.default_rng(context.seed)
    frames = []
    for index in range(cfg.count):
        atoms = context.parent_atoms.copy()
        strain = None
        if cfg.cell_stdev is not None and cfg.cell_stdev > 0.0:
            random_matrix = rng.normal(0.0, cfg.cell_stdev, (3, 3))
            strain = 0.5 * (random_matrix + random_matrix.T)
            atoms.set_cell((np.eye(3) + strain) @ atoms.cell.array, scale_atoms=True)
        if cfg.stdev > 0.0:
            atoms.positions += rng.normal(0.0, cfg.stdev, atoms.positions.shape)
        frame_id = f"{cfg.source}-{index:04d}"
        metadata = {"seed": context.seed, "stdev": cfg.stdev, "index": index}
        if cfg.cell_stdev is not None:
            metadata["cell_stdev"] = cfg.cell_stdev
            metadata["strain"] = None if strain is None else strain.tolist()
        _attach_info(atoms, cfg.source, frame_id, cfg.split, metadata)
        frames.append(GeneratedFrame(atoms, cfg.source, frame_id, cfg.split, metadata))
    return frames


def generate_metastable_interpolation_frames(
    context: SamplingContext,
    config: MetastableInterpolationConfig | Mapping[str, Any],
) -> list[GeneratedFrame]:
    """Generate parent-to-metastable interpolation and extrapolation frames."""
    cfg = _coerce_config(config, MetastableInterpolationConfig)
    lambdas = tuple(float(value) for value in cfg.lambdas)
    if not lambdas or any(not np.isfinite(value) for value in lambdas):
        raise ValueError("lambdas must be a non-empty sequence of finite values")
    count = int(cfg.count) if cfg.count is not None else None
    if count is not None:
        _validate_count(count, "count")
    metastable = _metastable_items(context)
    parent = context.parent_atoms
    compatible = []
    for metastable_id, atoms in metastable:
        if len(atoms) == len(parent):
            compatible.append((metastable_id, atoms))
        elif not cfg.skip_incompatible:
            raise ValueError("metastable structure atom count differs from parent")
    if count is not None:
        return _generate_counted_metastable_interpolation(
            context, cfg, compatible, lambdas, count
        )
    frames = []
    for imeta, (metastable_id, atoms) in enumerate(compatible):
        parent_scaled = parent.get_scaled_positions(wrap=False)
        meta_scaled = atoms.get_scaled_positions(wrap=False)
        delta_scaled = meta_scaled - parent_scaled
        delta_scaled -= np.round(delta_scaled)
        parent_cell = parent.cell.array
        meta_cell = atoms.cell.array
        for ilambda, lambda_value in enumerate(lambdas):
            item = parent.copy()
            item.set_cell(
                parent_cell + lambda_value * (meta_cell - parent_cell),
                scale_atoms=False,
            )
            item.set_scaled_positions(parent_scaled + lambda_value * delta_scaled)
            frame_id = f"{cfg.source}-{imeta:04d}-{ilambda:04d}"
            metadata = {
                "metastable_id": metastable_id,
                "lambda": lambda_value,
                "lambda_index": ilambda,
                "interpolation_mode": "minimum_image_scaled",
            }
            _attach_info(item, cfg.source, frame_id, cfg.split, metadata)
            frames.append(
                GeneratedFrame(item, cfg.source, frame_id, cfg.split, metadata)
            )
    return frames


def _generate_counted_metastable_interpolation(
    context: SamplingContext,
    cfg: MetastableInterpolationConfig,
    metastable: Sequence[tuple[Any, Atoms]],
    lambdas: Sequence[float],
    count: int,
) -> list[GeneratedFrame]:
    if not metastable:
        return []
    parent = context.parent_atoms
    parent_scaled = parent.get_scaled_positions(wrap=False)
    parent_cell = parent.cell.array
    lambda_values = np.linspace(float(min(lambdas)), float(max(lambdas)), count)
    frames = []
    for index, lambda_value in enumerate(lambda_values):
        metastable_index = index % len(metastable)
        metastable_id, atoms = metastable[metastable_index]
        meta_scaled = atoms.get_scaled_positions(wrap=False)
        delta_scaled = meta_scaled - parent_scaled
        delta_scaled -= np.round(delta_scaled)
        meta_cell = atoms.cell.array
        item = parent.copy()
        lambda_float = float(lambda_value)
        item.set_cell(
            parent_cell + lambda_float * (meta_cell - parent_cell), scale_atoms=False
        )
        item.set_scaled_positions(parent_scaled + lambda_float * delta_scaled)
        frame_id = f"{cfg.source}-{index:04d}"
        metadata = {
            "metastable_id": metastable_id,
            "metastable_index": metastable_index,
            "lambda": lambda_float,
            "lambda_index": index,
            "interpolation_mode": "minimum_image_scaled",
        }
        _attach_info(item, cfg.source, frame_id, cfg.split, metadata)
        frames.append(GeneratedFrame(item, cfg.source, frame_id, cfg.split, metadata))
    return frames


def generate_contour_frames(
    context: SamplingContext, config: ContourSamplerConfig | Mapping[str, Any]
) -> list[GeneratedFrame]:
    """Generate frames with ASE contour exploration."""
    cfg = _coerce_config(config, ContourSamplerConfig)
    _validate_dynamics_config(cfg.steps, cfg.sample_interval)
    if context.calculator is None:
        raise ValueError("contour sampler requires a calculator")
    atoms = context.parent_atoms.copy()
    atoms.calc = context.calculator
    rng = np.random.default_rng(context.seed)
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=1.0, rng=rng)
    except TypeError:
        MaxwellBoltzmannDistribution(atoms, temperature_K=1.0)
    collected: list[tuple[Atoms, int | None]] = []

    def collect():
        step = getattr(dyn, "nsteps", None)
        if step == 0:
            return
        collected.append((atoms.copy(), int(step) if step is not None else None))

    ContourExploration = _contour_exploration_class()
    dyn = ContourExploration(
        atoms,
        maxstep=cfg.maxstep,
        parallel_drift=cfg.parallel_drift,
        energy_target=cfg.energy_target,
        angle_limit=cfg.angle_limit,
        potentiostat_step_scale=cfg.potentiostat_step_scale,
        remove_translation=cfg.remove_translation,
        use_frenet_serret=cfg.use_frenet_serret,
        initialization_step_scale=cfg.initialization_step_scale,
        use_target_shift=cfg.use_target_shift,
        target_shift_previous_steps=cfg.target_shift_previous_steps,
        use_tangent_curvature=cfg.use_tangent_curvature,
        rng=rng,
        force_consistent=cfg.force_consistent,
        trajectory=_output_path(context, cfg.trajectory),
        logfile=_output_path(context, cfg.logfile),
        append_trajectory=cfg.append_trajectory,
        loginterval=cfg.loginterval,
    )
    dyn.attach(collect, interval=cfg.sample_interval)
    dyn.run(cfg.steps if cfg.count is None else cfg.count * cfg.sample_interval)
    return _frames_from_collected(
        collected,
        cfg.source,
        cfg.split,
        context.seed,
        "contour",
        cfg.sample_interval,
        cfg.count,
    )


def generate_npt_md_frames(
    context: SamplingContext, config: NptMdSamplerConfig | Mapping[str, Any]
) -> list[GeneratedFrame]:
    """Generate frames from ASE NPT Berendsen dynamics."""
    cfg = _coerce_config(config, NptMdSamplerConfig)
    _validate_dynamics_config(cfg.steps, cfg.sample_interval)
    if context.calculator is None:
        raise ValueError("npt_md sampler requires a calculator")
    atoms = context.parent_atoms.copy()
    atoms.calc = context.calculator
    rng = np.random.default_rng(context.seed)
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature, rng=rng)
    except TypeError:
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature)
    collected: list[tuple[Atoms, int | None]] = []

    def collect():
        step = getattr(dyn, "nsteps", None)
        if step == 0:
            return
        collected.append((atoms.copy(), int(step) if step is not None else None))

    dyn = _npt_berendsen_class()(
        atoms,
        cfg.timestep * units.fs,
        temperature_K=cfg.temperature,
        pressure_au=cfg.pressure * units.bar,
        taut=cfg.taut * units.fs,
        taup=cfg.taup * units.fs,
        compressibility_au=cfg.compressibility,
        trajectory=_output_path(context, cfg.trajectory),
        logfile=_output_path(context, cfg.logfile),
        loginterval=cfg.loginterval,
    )
    dyn.attach(collect, interval=cfg.sample_interval)
    dyn.run(cfg.steps)
    frames = _frames_from_collected(
        collected,
        cfg.source,
        cfg.split,
        context.seed,
        "npt_md",
        cfg.sample_interval,
        None,
    )
    for frame in frames:
        frame.metadata.update(
            {
                "ensemble": "npt_berendsen",
                "temperature": cfg.temperature,
                "pressure": cfg.pressure,
                "timestep": cfg.timestep,
            }
        )
        frame.atoms.info.update(frame.metadata)
    return frames


def generate_frames_from_source(
    context: SamplingContext, source: Mapping[str, Any]
) -> list[GeneratedFrame]:
    """Dispatch one workflow ``sampling.sources`` entry."""
    name = str(source.get("name", "")).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("sampling source requires a name")
    payload = {key: value for key, value in source.items() if key != "name"}
    if name in {"rattle", "random"}:
        payload.setdefault("source", name)
        return generate_rattle_frames(context, payload)
    if name in {"metastable_interpolation", "interpolation"}:
        payload.setdefault("source", "metastable_interpolation")
        return generate_metastable_interpolation_frames(context, payload)
    if name in {"contour", "contour_exploration"}:
        payload.setdefault("source", "contour")
        return generate_contour_frames(context, payload)
    if name in {"npt", "npt_md"}:
        payload.setdefault("source", "npt_md")
        return generate_npt_md_frames(context, payload)
    raise ValueError(f"Unknown sampler source: {source.get('name', source)}")


def _coerce_config(config, config_type):
    if isinstance(config, config_type):
        return config
    if isinstance(config, Mapping):
        valid = set(config_type.__dataclass_fields__)
        extra = set(config) - valid
        if extra:
            raise ValueError(
                f"Unknown {config_type.__name__} option(s): {', '.join(sorted(extra))}"
            )
        return config_type(**dict(config))
    raise TypeError(f"config must be {config_type.__name__} or mapping")


def _validate_count(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_dynamics_config(steps: int, sample_interval: int) -> None:
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if int(sample_interval) <= 0:
        raise ValueError("sample_interval must be positive")


def _attach_info(
    atoms: Atoms, source: str, frame_id: str, split: Split, metadata: Mapping[str, Any]
) -> None:
    atoms.info.update({"source": source, "frame_id": frame_id, "split": split})
    atoms.info.update(dict(metadata))


def _metastable_items(context: SamplingContext) -> list[tuple[Any, Atoms]]:
    items: list[tuple[Any, Atoms]] = []
    for index, atoms in enumerate(context.metastable_structures):
        items.append((atoms.info.get("metastable_id", index), atoms))
    for index, record in enumerate(context.metastable_records):
        atoms = record.get("atoms")
        if isinstance(atoms, Atoms):
            items.append(
                (record.get("id", atoms.info.get("metastable_id", index)), atoms)
            )
    return items


def _output_path(context: SamplingContext, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() and context.output_dir is not None:
        path = context.output_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _frames_from_collected(
    collected: Sequence[tuple[Atoms, int | None]],
    source: str,
    split: Split,
    seed: int,
    sampler: str,
    sample_interval: int,
    count: int | None,
) -> list[GeneratedFrame]:
    selected = list(collected if count is None else collected[:count])
    frames = []
    for index, (atoms, actual_step) in enumerate(selected):
        item = atoms.copy()
        step = actual_step if actual_step is not None else (index + 1) * sample_interval
        frame_id = f"{source}-{index:04d}"
        metadata = {
            "seed": seed,
            "sampler": sampler,
            "index": index,
            "step": step,
            "sample_interval": sample_interval,
        }
        _attach_info(item, source, frame_id, split, metadata)
        frames.append(GeneratedFrame(item, source, frame_id, split, metadata))
    return frames


def _contour_exploration_class():
    from ase.md.contour_exploration import ContourExploration

    return ContourExploration


def _npt_berendsen_class():
    return NPTBerendsen


__all__ = [
    "ContourSamplerConfig",
    "GeneratedFrame",
    "MetastableInterpolationConfig",
    "NptMdSamplerConfig",
    "RattleSamplerConfig",
    "SamplingContext",
    "generate_contour_frames",
    "generate_frames_from_source",
    "generate_metastable_interpolation_frames",
    "generate_npt_md_frames",
    "generate_rattle_frames",
]
