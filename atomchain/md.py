"""
Molecular Dynamics with ML potentials
"""

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase import units
from atomchain.init_model import init_calc


def md_nve_velocity_verlet(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NVE molecular dynamics using Velocity Verlet integrator.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Initial temperature in Kelvin for velocity initialization (default: 300 K)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.VelocityVerlet: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_nve_velocity_verlet
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_nve_velocity_verlet(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    dyn = VelocityVerlet(catoms, timestep_ase, trajectory=trajectory, logfile=logfile, loginterval=loginterval)

    # Run MD
    dyn.run(steps)

    return catoms, dyn


def md_nvt_langevin(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    friction=0.002,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NVT molecular dynamics using Langevin thermostat.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Target temperature in Kelvin (default: 300 K)
        friction (float): Friction coefficient in 1/fs (default: 0.002)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.Langevin: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_nvt_langevin
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_nvt_langevin(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     friction=0.002,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    dyn = Langevin(
        catoms,
        timestep_ase,
        temperature_K=temperature,
        friction=friction,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )

    # Run MD
    dyn.run(steps)

    return catoms, dyn


def md_nvt_berendsen(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    taut=100.0,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NVT molecular dynamics using Berendsen thermostat.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Target temperature in Kelvin (default: 300 K)
        taut (float): Time constant for temperature coupling in fs (default: 100 fs)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.NVTBerendsen: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_nvt_berendsen
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_nvt_berendsen(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     taut=100.0,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    taut_ase = taut * units.fs
    dyn = NVTBerendsen(
        catoms,
        timestep_ase,
        temperature_K=temperature,
        taut=taut_ase,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )

    # Run MD
    dyn.run(steps)

    return catoms, dyn


def md_nvt_andersen(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    andersen_prob=0.01,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NVT molecular dynamics using Andersen thermostat.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Target temperature in Kelvin (default: 300 K)
        andersen_prob (float): Probability of collision per timestep (default: 0.01)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.Andersen: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_nvt_andersen
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_nvt_andersen(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     andersen_prob=0.01,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    dyn = Andersen(
        catoms,
        timestep_ase,
        temperature_K=temperature,
        andersen_prob=andersen_prob,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )

    # Run MD
    dyn.run(steps)

    return catoms, dyn


def md_nvt_bussi(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    taut=100.0,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NVT molecular dynamics using Bussi stochastic velocity rescaling thermostat.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Target temperature in Kelvin (default: 300 K)
        taut (float): Time constant for temperature coupling in fs (default: 100 fs)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.Bussi: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_nvt_bussi
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_nvt_bussi(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     taut=100.0,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    taut_ase = taut * units.fs
    dyn = Bussi(
        catoms,
        timestep_ase,
        temperature_K=temperature,
        taut=taut_ase,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )

    # Run MD
    dyn.run(steps)

    return catoms, dyn


def md_npt_berendsen(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    pressure=1.01325,
    taut=100.0,
    taup=1000.0,
    compressibility=4.57e-5,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NPT molecular dynamics using Berendsen thermostat and barostat.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Target temperature in Kelvin (default: 300 K)
        pressure (float): Target pressure in bar (default: 1.01325 bar = 1 atm)
        taut (float): Time constant for temperature coupling in fs (default: 100 fs)
        taup (float): Time constant for pressure coupling in fs (default: 1000 fs)
        compressibility (float): Compressibility in 1/bar (default: 4.57e-5 for water)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.NPTBerendsen: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_npt_berendsen
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_npt_berendsen(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     pressure=1.01325,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    taut_ase = taut * units.fs
    taup_ase = taup * units.fs
    pressure_ase = pressure * units.bar
    
    dyn = NPTBerendsen(
        catoms,
        timestep_ase,
        temperature_K=temperature,
        pressure_au=pressure_ase,
        taut=taut_ase,
        taup=taup_ase,
        compressibility_au=compressibility,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )

    # Run MD
    dyn.run(steps)

    return catoms, dyn


def md_npt(
    atoms,
    calc=None,
    model_path=None,
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    pressure=1.01325,
    ttime=25.0,
    pfactor=None,
    trajectory=None,
    logfile=None,
    loginterval=1,
):
    """
    Run NPT molecular dynamics using Nose-Hoover-like dynamics.

    Args:
        atoms (ase.Atoms): Initial atomic structure
        calc (str or ase.Calculator): Calculator type ('multibinit', 'chgnet', etc.) or calculator object
        model_path (str): Path to model file (required for 'multibinit', 'deepmd', etc.)
        timestep (float): MD timestep in femtoseconds (default: 1.0 fs)
        steps (int): Number of MD steps (default: 1000)
        temperature (float): Target temperature in Kelvin (default: 300 K)
        pressure (float): Target pressure in bar (default: 1.01325 bar = 1 atm)
        ttime (float): Characteristic time for temperature in fs (default: 25 fs)
        pfactor (float): Pressure coupling constant (default: None, auto-calculated)
        trajectory (str): Trajectory filename (default: None, no trajectory saved)
        logfile (str): Log filename (default: None, prints to stdout)
        loginterval (int): Interval for logging (default: 1)

    Returns:
        ase.Atoms: Final atomic structure
        ase.md.NPT: MD dynamics object

    Example:
        >>> from ase.io import read
        >>> from atomchain.md import md_npt
        >>> atoms = read('structure.vasp')
        >>> final_atoms, dyn = md_npt(
        ...     atoms=atoms,
        ...     calc='multibinit',
        ...     model_path='config.conf',
        ...     timestep=1.0,
        ...     steps=1000,
        ...     temperature=300.0,
        ...     pressure=1.01325,
        ...     trajectory='md.traj'
        ... )
    """
    catoms = atoms.copy()

    # Initialize calculator
    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")
    catoms.calc = calc

    # Set initial velocities
    MaxwellBoltzmannDistribution(catoms, temperature_K=temperature)

    # Create dynamics
    timestep_ase = timestep * units.fs
    ttime_ase = ttime * units.fs
    pressure_ase = pressure * units.bar
    
    dyn = NPT(
        catoms,
        timestep_ase,
        temperature_K=temperature,
        externalstress=pressure_ase,
        ttime=ttime_ase,
        pfactor=pfactor,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )

    # Run MD
    dyn.run(steps)

    return catoms, dyn
