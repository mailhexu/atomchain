"""
Initialize model for calculation.
"""

import os
import urllib.error
import urllib.request

MACE_R2SCAN_MODEL_URL = (
    "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model"
)
MACE_R2SCAN_MODEL_PATH = "~/.config/mace/mace-mh-1.model"


def _ensure_mace_r2scan_model(model_path=None):
    """Return a local MACE R2SCAN model path, downloading it when missing."""
    mace_model_file = os.path.expanduser(model_path or MACE_R2SCAN_MODEL_PATH)
    if os.path.exists(mace_model_file):
        return mace_model_file

    model_dir = os.path.dirname(mace_model_file)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    try:
        print(
            "MACE R2SCAN model file not found. Downloading "
            f"{MACE_R2SCAN_MODEL_URL} to {mace_model_file}..."
        )
        urllib.request.urlretrieve(MACE_R2SCAN_MODEL_URL, mace_model_file)
    except (OSError, urllib.error.URLError) as exc:
        if os.path.exists(mace_model_file):
            os.remove(mace_model_file)
        raise RuntimeError(
            "MACE R2SCAN model file is not available and automatic download failed. "
            f"Download it manually from {MACE_R2SCAN_MODEL_URL} and save it as "
            f"{mace_model_file}."
        ) from exc
    return mace_model_file


def _get_torch_device(torch_module):
    """Return the best available torch device for MACE calculators."""
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def init_calc(model_type="mace", model_path=None):
    """
    Initialize calculator for calculation.

    param:
    =====
    model_type: str (default: "mace")
        Supported models: 'matgl', 'm3gnet', 'chgnet', 'deepmd', 'mace', 'xq', 'multibinit' (or 'mb')
    model_path: str (default: None).
        Path to the model file. Either a directory or a file.
        Required for 'deepmd' and 'multibinit' calculators.
        For 'multibinit': path to configuration file (e.g., 'config.conf')

    return:
    =====
    calc: calculator object

    Examples:
    =========
    # MACE (default)
    calc = init_calc(model_type="mace")

    # CHGNet
    calc = init_calc(model_type="chgnet")

    # MULTIBINIT with config file
    calc = init_calc(model_type="multibinit", model_path="config.conf")

    # MULTIBINIT with short alias
    calc = init_calc(model_type="mb", model_path="config.conf")
    """
    if model_type.lower() == "matgl":
        import matgl
        from matgl.ext.ase import M3GNetCalculator as M3GCalc

        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calc = M3GCalc(potential=pot, stress_weight=1.0)
    elif model_type.lower() == "m3gnet":
        from m3gnet.models import M3GNet, Potential
        from m3gnet.models import M3GNetCalculator as M3GCalc

        if model_path is None:
            potential = Potential(M3GNet.load())
        else:
            potential = Potential(M3GNet.from_dir(model_path))
        calc = M3GCalc(potential=potential, compute_stress=True)
    elif model_type.lower() == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        calc = CHGNetCalculator(model=None)
    elif model_type.lower() == "deepmd":
        from deepmd.calculator import DP

        calc = DP(model=model_path)
    elif model_type.lower() == "mace":
        import torch
        from mace.calculators import mace_mp

        device = _get_torch_device(torch)
        calc = mace_mp(
            model="medium", dispersion=False, default_dtype="float32", device=device
        )
    elif model_type.lower() == "mace-r2scan":
        import torch
        from mace.calculators import mace_mp

        mace_model_file = _ensure_mace_r2scan_model(model_path=model_path)
        device = _get_torch_device(torch)
        calc = mace_mp(
            mace_model_file,
            device=device,
            default_dtype="float64",
            dispersion=False,
            dispersion_xc="pbe",
            head="matpes_r2scan",
        )

    elif model_type.lower() == "xq":
        from atomic_potential_xq.calculator import XQCalculator

        calc = XQCalculator()
    elif model_type.lower() in ["multibinit", "mb"]:
        # MULTIBINIT effective potential calculator (requires pymultibinit >= 0.2.0)
        # Validate model_path is provided
        if model_path is None:
            raise ValueError(
                "MULTIBINIT requires a configuration file. "
                "Please provide model_path, e.g., init_calc(model_type='multibinit', model_path='config.conf')"
            )

        # Check if configuration file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Configuration file not found: {model_path}")

        # Import and initialize MULTIBINIT calculator
        try:
            from pymultibinit.calculator import MultibinitCalculator
        except ImportError as e:
            raise ImportError(
                "pymultibinit is required for MULTIBINIT calculator but is not installed. "
                "Please install pymultibinit."
            ) from e

        try:
            calc = MultibinitCalculator.from_config_file(model_path)
        except OSError as e:
            raise RuntimeError(
                f"Failed to initialize MULTIBINIT calculator from {model_path}: {e}"
            ) from e
    else:
        raise ValueError(
            "model_type not recognized. The current supported models are: "
            "'matgl', 'm3gnet', 'chgnet', 'deepmd', 'mace', 'xq', 'multibinit', 'mb'"
        )
    return calc
