"""
Initialize model for calculation.
"""

import os


def init_calc(model_type="chgnet", model_path=None):
    """
    Initialize calculator for calculation.

    param:
    =====
    model_type: str (default: "chgnet")
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
    # CHGNet (default)
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
        from mace.calculators import mace_mp

        calc = mace_mp(
            model="medium", dispersion=False, default_dtype="float32", device="cpu"
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

        calc = MultibinitCalculator.from_config_file(model_path)
    else:
        raise ValueError(
            "model_type not recognized. The current supported models are: "
            "'matgl', 'm3gnet', 'chgnet', 'deepmd', 'mace', 'xq', 'multibinit', 'mb'"
        )
    return calc
