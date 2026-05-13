"""Thin wrappers around pymultibinit training functionality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class MultibinitTrainingResult:
    """Summary of a delegated pymultibinit training run."""

    model_config: Optional[str] = None
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    artifacts: Optional[dict] = None

    def to_dict(self):
        return asdict(self)


def train_multibinit_model(
    ddb, hist, config=None, output_dir="multibinit_training", metadata=None, **kwargs
):
    """Delegate MULTIBINIT model training to pymultibinit.

    atomchain intentionally does not implement MULTIBINIT binary handling. The
    canonical training entry point belongs in pymultibinit, which should call
    the multibinit executable rather than using ABINIT library mode.
    """
    train_func = _load_pymultibinit_training_api()
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = train_func(
        ddb=str(ddb),
        hist=str(hist),
        config=str(config) if config is not None else None,
        output_dir=str(outdir),
        **kwargs,
    )
    result = _coerce_training_result(raw, outdir)
    metadata_path = (
        Path(metadata)
        if metadata is not None
        else outdir / "atomchain_training_result.yaml"
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result.to_dict(), handle, sort_keys=True)
    return result


def validate_trained_multibinit_model(
    model_config=None, artifacts=None, require_init=False, output_dir=None
):
    """Validate files returned by pymultibinit training."""
    missing = []
    paths = []
    if model_config is not None:
        paths.append(model_config)
    for value in (artifacts or {}).values():
        if isinstance(value, (str, Path)):
            paths.append(value)
    base = Path(output_dir) if output_dir is not None else None
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute() and base is not None:
            candidate = base / candidate
        if not candidate.exists():
            missing.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            "Missing trained MULTIBINIT output files: " + ", ".join(missing)
        )
    if require_init and model_config is not None:
        from atomchain.init_model import init_calc

        init_calc("multibinit", model_path=str(model_config))
    return True


def _load_pymultibinit_training_api():
    try:
        from pymultibinit.training import train_multibinit_model as train_func

        return train_func
    except Exception as first_exc:
        try:
            from pymultibinit import train_multibinit_model as train_func

            return train_func
        except Exception as second_exc:
            raise ImportError(
                "pymultibinit training API is required. Install/update pymultibinit with MULTIBINIT training support."
            ) from second_exc or first_exc


def _coerce_training_result(raw, outdir):
    if isinstance(raw, MultibinitTrainingResult):
        return raw
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if isinstance(raw, dict):
        return MultibinitTrainingResult(
            model_config=raw.get("model_config") or raw.get("config"),
            output_dir=raw.get("output_dir", str(outdir)),
            log_file=raw.get("log_file"),
            artifacts=raw.get(
                "artifacts",
                {
                    k: v
                    for k, v in raw.items()
                    if k not in {"model_config", "config", "output_dir", "log_file"}
                },
            ),
        )
    return MultibinitTrainingResult(
        model_config=str(raw) if raw is not None else None,
        output_dir=str(outdir),
        artifacts={},
    )
