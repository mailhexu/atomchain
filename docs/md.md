# Molecular Dynamics

Run molecular dynamics simulations with ML potentials.

## Available Functions

| Function | Ensemble | Thermostat/Barostat |
|----------|----------|---------------------|
| `md_nve_velocity_verlet()` | NVE | None (energy conserving) |
| `md_nvt_langevin()` | NVT | Langevin |
| `md_nvt_berendsen()` | NVT | Berendsen |
| `md_nvt_andersen()` | NVT | Andersen |
| `md_nvt_bussi()` | NVT | Bussi |
| `md_npt_berendsen()` | NPT | Berendsen |
| `md_npt()` | NPT | Nose-Hoover |

## Quick Start

### NVE (Microcanonical)

```python
from ase.io import read
from atomchain.md import md_nve_velocity_verlet

atoms = read('structure.vasp')

final_atoms, dyn = md_nve_velocity_verlet(
    atoms=atoms,
    calc='multibinit',
    model_path='config.conf',
    timestep=1.0,           # fs
    steps=1000,
    temperature=300.0,      # K (initial)
    trajectory='md.traj',
    logfile='md.log'
)
```

### NVT (Canonical)

```python
from atomchain.md import md_nvt_langevin

final_atoms, dyn = md_nvt_langevin(
    atoms=atoms,
    calc='multibinit',
    model_path='config.conf',
    timestep=1.0,           # fs
    steps=1000,
    temperature=300.0,      # K (target)
    friction=0.002,         # 1/fs
    trajectory='md.traj',
    logfile='md.log'
)
```

### NPT (Isothermal-Isobaric)

```python
from atomchain.md import md_npt_berendsen

final_atoms, dyn = md_npt_berendsen(
    atoms=atoms,
    calc='multibinit',
    model_path='config.conf',
    timestep=1.0,           # fs
    steps=1000,
    temperature=300.0,      # K
    pressure=1.01325,       # bar (1 atm)
    taut=100.0,             # fs
    taup=1000.0,            # fs
    trajectory='md.traj',
    logfile='md.log'
)
```

## Common Parameters

All MD functions share these parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `atoms` | ase.Atoms | Initial structure | Required |
| `calc` | str or Calculator | Calculator type or object | None (CHGNet) |
| `model_path` | str | Path to model file | None |
| `timestep` | float | Time step in fs | 1.0 |
| `steps` | int | Number of MD steps | 1000 |
| `temperature` | float | Temperature in K | 300.0 |
| `trajectory` | str | Trajectory filename | None |
| `logfile` | str | Log filename | None |
| `loginterval` | int | Logging interval | 1 |

## Thermostat-Specific Parameters

### Langevin
- `friction` (float): Friction coefficient in 1/fs (default: 0.002)

### Berendsen (NVT)
- `taut` (float): Temperature coupling time in fs (default: 100)

### Andersen
- `andersen_prob` (float): Collision probability per step (default: 0.01)

### Bussi
- `taut` (float): Temperature coupling time in fs (default: 100)

## Barostat-Specific Parameters

### NPTBerendsen
- `pressure` (float): Target pressure in bar (default: 1.01325)
- `taut` (float): Temperature coupling time in fs (default: 100)
- `taup` (float): Pressure coupling time in fs (default: 1000)
- `compressibility` (float): Compressibility in 1/bar (default: 4.57e-5)

### NPT (Nose-Hoover)
- `pressure` (float): Target pressure in bar (default: 1.01325)
- `ttime` (float): Temperature characteristic time in fs (default: 25)
- `pfactor` (float): Pressure coupling constant (default: None, auto)

## Output Files

### Trajectory File
Contains atomic positions, velocities, forces, and cell parameters at each step.

Read with ASE:
```python
from ase.io import read
traj = read('md.traj', index=':')  # All frames
```

### Log File
Standard ASE MD log format:
```
Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
0.0000       -3654.78     -3655.00       0.22      340.0
0.0010       -3654.78     -3654.97       0.19      294.2
...
```

## Choosing a Thermostat

| Thermostat | Use Case | Pros | Cons |
|------------|----------|------|------|
| **Langevin** | General NVT, equilibration | Good temperature control, stochastic | Non-deterministic |
| **Berendsen** | Fast equilibration | Fast convergence | Not canonical ensemble |
| **Andersen** | Canonical sampling | Proper ensemble | Discontinuous velocities |
| **Bussi** | Production runs | Canonical, smooth | Requires tuning |

## Choosing a Barostat

| Barostat | Use Case | Notes |
|----------|----------|-------|
| **Berendsen** | Fast equilibration | Not proper NPT ensemble |
| **Nose-Hoover** | Production runs | Proper NPT ensemble |

## Complete Example

```python
from ase.io import read, write
from atomchain.md import md_nve_velocity_verlet, md_nvt_langevin

# Load structure
atoms = read('BaHfO3.vasp')

# Equilibrate with NVT (500 steps)
atoms_eq, _ = md_nvt_langevin(
    atoms=atoms,
    calc='multibinit',
    model_path='config.conf',
    timestep=1.0,
    steps=500,
    temperature=300.0,
    friction=0.002,
    trajectory='equilibration.traj',
    logfile='equilibration.log'
)

# Production run with NVE (5000 steps)
atoms_final, _ = md_nve_velocity_verlet(
    atoms=atoms_eq,
    calc='multibinit',
    model_path='config.conf',
    timestep=1.0,
    steps=5000,
    temperature=300.0,
    trajectory='production.traj',
    logfile='production.log'
)

# Save final structure
write('final.vasp', atoms_final)
```

## Supported Calculators

All MD functions support any calculator from `init_calc()`:
- `multibinit` - ABINIT MULTIBINIT effective potentials
- `chgnet` - CHGNet materials property prediction
- `m3gnet` - M3GNet graph neural networks
- `deepmd` - DeePMD-kit potentials
- `mace` - MACE neural networks
- Custom ASE calculators (pass calculator object)

## Tips

1. **Timestep selection**: Use 1 fs for most systems, 0.5 fs for light elements (H)
2. **Equilibration**: Run NVT for 500-1000 steps before production
3. **Temperature control**: Langevin works well for most cases
4. **Energy conservation**: Check NVE log file - drift should be minimal
5. **Pressure control**: NPT barostats need longer equilibration (1000+ steps)
