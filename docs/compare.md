# Trajectory Comparison

Compare calculated properties (energy, forces, stress) between two trajectory files. Ideal for validating machine learning potentials against DFT references or comparing different ML models.

## CLI Usage

```bash
# Basic comparison
mlcompare dft.traj ml.traj -o comparison.png

# With custom labels
mlcompare reference.traj predicted.traj --labels "DFT" "CHGNet" --show

# Normalize energies and save as PDF
mlcompare traj1.traj traj2.traj --normalize-energy --format pdf -o results.pdf

# Quiet mode (no console output)
mlcompare ref.traj pred.traj --quiet
```

### Options
- `trajectory1` - First trajectory file (reference)
- `trajectory2` - Second trajectory file (predicted)
- `--output, -o` - Output plot file (default: `trajectory_comparison.png`)
- `--labels LABEL1 LABEL2` - Custom labels for trajectories
- `--normalize-energy` - Normalize energies relative to minimum
- `--show` - Display plot interactively
- `--format {png,pdf,svg}` - Output format (overrides file extension)
- `--quiet, -q` - Suppress console statistics output

## Python API

```python
from atomchain.compare import compare_trajectories

# Basic comparison
results = compare_trajectories(
    'dft.traj',
    'ml.traj',
    output='comparison.png'
)

# With custom options
results = compare_trajectories(
    'reference.traj',
    'predicted.traj',
    labels=('DFT', 'CHGNet'),
    normalize_energy=True,
    show=True,
    quiet=True
)

# Access results
print(f"Energy R²: {results['metrics']['energy']['r2']:.4f}")
print(f"Energy RMSE: {results['metrics']['energy']['rmse']:.4f} eV")
print(f"Forces RMSE: {results['metrics']['forces']['rmse']:.4f} eV/Å")

# Access raw data
energies_1 = results['energy_1']  # Array of energies from trajectory 1
forces_rms_1 = results['forces_1']  # RMS forces per structure
```

### Parameters
- `trajectory1` - Path to first trajectory file
- `trajectory2` - Path to second trajectory file
- `labels` - Optional tuple of (label1, label2) for plot axes (default: `('Trajectory 1', 'Trajectory 2')`)
- `normalize_energy` - Normalize energies relative to minimum (default: `False`)
- `output` - Output file path for plot (default: `'trajectory_comparison.png'`)
- `show` - Display plot interactively (default: `False`)
- `quiet` - Suppress console output (default: `False`)

### Returns
Dictionary with comparison results:
- `energy_1`, `energy_2` - Energy arrays (eV)
- `forces_1`, `forces_2` - RMS forces per structure (eV/Å)
- `hydro_1`, `hydro_2` - Hydrostatic stress arrays (GPa)
- `shear_1`, `shear_2` - Shear stress arrays (GPa)
- `metrics` - Dict of statistical metrics:
  - `energy` - Energy metrics (`r2`, `rmse`, `mae`)
  - `forces` - Forces metrics
  - `hydro` - Hydrostatic stress metrics
  - `shear` - Shear stress metrics

## Output

### Console Statistics

```
============================================================
Trajectory Comparison Statistics
============================================================
Trajectory 1: DFT
Trajectory 2: CHGNet
Structures compared: 100

Property          R²      RMSE        MAE         
------------------------------------------------------------
Energy (eV)       0.9985  0.0234      0.0189
RMS Forces (eV/Å) 0.9872  0.0456      0.0321
Hydro Stress (GPa)0.9756  0.1234      0.0987
Shear Stress (GPa)0.9691  0.0876      0.0654
============================================================
```

### Comparison Plots

The tool generates a 2×2 grid of scatter plots:

1. **Energy** - Per-structure energy comparison
2. **RMS Forces** - Root-mean-square forces per structure
3. **Hydrostatic Stress** - Pressure (average of normal stresses)
4. **Shear Stress** - Maximum shear stress

Each subplot includes:
- Scatter plot with identity line (y=x)
- R², RMSE, MAE statistics
- Data range information

## Workflow Integration

### ML Model Validation Pipeline

```python
from atomchain.rattle import generate_rattle_dataset
from atomchain.batch import calculate_trajectory_batch
from atomchain.compare import compare_trajectories

# Step 1: Generate test structures
print("Generating test structures...")
test_structures = generate_rattle_dataset(
    'POSCAR',
    n_struct=50,
    stdev=0.05,
    supercell=[2, 2, 2],
    output='test_structures.traj'
)

# Step 2: Calculate with DFT (reference) - in practice, you'd have pre-calculated DFT
# Here we use CHGNet as a proxy for DFT
print("Calculating DFT reference...")
dft_results = calculate_trajectory_batch(
    'test_structures.traj',
    calculator='chgnet',  # Replace with actual DFT calculator
    output='dft_reference.traj'
)

# Step 3: Calculate with ML potential
print("Calculating with ML potential...")
ml_results = calculate_trajectory_batch(
    'test_structures.traj',
    calculator='mace',  # Your ML potential
    output='ml_predicted.traj'
)

# Step 4: Compare results
print("Comparing results...")
comparison = compare_trajectories(
    'dft_reference.traj',
    'ml_predicted.traj',
    labels=('DFT', 'MACE'),
    normalize_energy=True,
    output='validation_results.pdf',
    show=True
)

# Step 5: Analyze metrics
print(f"\nValidation Results:")
print(f"  Energy R² = {comparison['metrics']['energy']['r2']:.4f}")
print(f"  Forces RMSE = {comparison['metrics']['forces']['rmse']:.4f} eV/Å")

# Check if model meets accuracy requirements
energy_threshold = 0.01  # 10 meV per atom
forces_threshold = 0.10  # 0.1 eV/Å

energy_rmse = comparison['metrics']['energy']['rmse']
forces_rmse = comparison['metrics']['forces']['rmse']

if energy_rmse < energy_threshold and forces_rmse < forces_threshold:
    print("✓ ML model passes validation criteria")
else:
    print("✗ ML model needs improvement")
    if energy_rmse >= energy_threshold:
        print(f"  Energy RMSE ({energy_rmse:.4f}) exceeds threshold ({energy_threshold})")
    if forces_rmse >= forces_threshold:
        print(f"  Forces RMSE ({forces_rmse:.4f}) exceeds threshold ({forces_threshold})")
```

### Comparing Multiple Models

```python
from atomchain.compare import compare_trajectories

# Compare different ML potentials against DFT reference
models = ['chgnet', 'mace', 'm3gnet']
results = {}

for model in models:
    comparison = compare_trajectories(
        'dft_reference.traj',
        f'{model}_predicted.traj',
        labels=('DFT', model.upper()),
        output=f'comparison_{model}.png',
        quiet=True
    )
    results[model] = comparison['metrics']

# Print summary comparison
print("\nModel Comparison Summary:")
print(f"{'Model':<10} {'Energy R²':>12} {'Energy RMSE':>15} {'Forces RMSE':>15}")
print("-" * 55)
for model in models:
    metrics = results[model]
    print(f"{model.upper():<10} {metrics['energy']['r2']:>12.4f} "
          f"{metrics['energy']['rmse']:>15.6f} "
          f"{metrics['forces']['rmse']:>15.6f}")
```

## Properties Compared

### Energy
- Per-structure total energy in eV
- Optionally normalized relative to minimum energy
- Useful for comparing relative stability predictions

### RMS Forces
- Root-mean-square of force magnitudes per structure
- Computed as: `RMS = sqrt(mean(||F_i||²))` over all atoms
- Units: eV/Å
- Indicates overall force prediction accuracy

### Hydrostatic Stress (Pressure)
- Average of normal stress components: `P = -(σ_xx + σ_yy + σ_zz) / 3`
- Converted from eV/Å³ to GPa
- Represents volumetric stress

### Shear Stress
- Maximum shear stress from principal stresses
- Computed from eigenvalues of stress tensor
- Converted from eV/Å³ to GPa
- Indicates deviatoric stress prediction accuracy

## Statistical Metrics

### R² (Coefficient of Determination)
- Range: -∞ to 1 (1 is perfect)
- Measures how well predictions match reference
- Values close to 1 indicate excellent agreement

### RMSE (Root Mean Square Error)
- Always positive
- Same units as the property being compared
- Penalizes large errors more than MAE

### MAE (Mean Absolute Error)
- Always positive
- Same units as the property being compared
- Less sensitive to outliers than RMSE

## Tips and Best Practices

### Energy Normalization
Enable `--normalize-energy` when:
- Comparing structures with different compositions
- Total energies have large absolute values
- You care about relative energies, not absolute values

### Mismatched Trajectory Lengths
If trajectories have different lengths:
- The tool automatically compares the first N structures (N = shorter length)
- A warning is printed unless `--quiet` is used
- This allows partial comparisons without error

### Large Datasets
For very large trajectories:
- Consider comparing a random subset first
- Use `--quiet` to reduce console output overhead
- Save plots as PDF for better quality with many data points

### Interactive Analysis
```python
# Load comparison results for further analysis
import matplotlib.pyplot as plt
import numpy as np

results = compare_trajectories('ref.traj', 'pred.traj', quiet=True)

# Find structures with largest errors
energy_errors = np.abs(results['energy_1'] - results['energy_2'])
worst_idx = np.argmax(energy_errors)
print(f"Structure {worst_idx} has largest energy error: {energy_errors[worst_idx]:.4f} eV")

# Custom analysis
forces_corr = np.corrcoef(results['forces_1'], results['forces_2'])[0, 1]
print(f"Forces correlation: {forces_corr:.4f}")
```

## Requirements

- ASE (Atomic Simulation Environment)
- NumPy
- Matplotlib

Trajectories must have pre-calculated properties (energy, forces, stress) attached. Use `mlbatch` to calculate properties if needed.

## See Also

- [Batch Processing](batch.md) - Calculate properties for trajectory files
- [Rattle Dataset Generation](rattle.md) - Generate training/test datasets
