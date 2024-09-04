"""
Initialize model for calculation.
"""


def init_calc(model_type="chgnet", model_path=None):
    """
    Initialize calculator for calculation.
    param:
    =====
    model_type: str (default: "chgnet")
    model_path: str (default: None).
      path to the model file. Either a directory or a file.

    return:
    =====
    calc: calculator object
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
    else:
        raise ValueError(
            "model_type not recognized. The current supported models are: 'matgl', 'm3gnet', 'chgnet', 'deepmd'"
        )
    return calc
