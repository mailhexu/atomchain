"""
MULTIBINIT Relaxation Example: BaHfO3 (Barium Hafnate)
========================================================

Purpose:
    Demonstrate structure relaxation using MULTIBINIT effective potential
    for the prototypical ferroelectric BaHfO3.

What this example does:
    1. Loads a slightly distorted BaHfO3 structure
    2. Initializes MULTIBINIT calculator from configuration file
    3. Performs structure relaxation (atomic positions + cell)
    4. Saves relaxed structure and visualizes results

How to run:
    python multibinit_relaxation.py

Prerequisites:
    - pymultibinit installed and configured
    - BaHfO3_DDB and BaHfO3.xml files in current directory
      (Note: Despite the filename, these contain BaHfO3 data - Hf not Ti!)
    - Configuration file: BaHfO3_config.conf

Expected behavior:
    - Relaxation should converge in ~10-50 steps
    - Final energy should be lower than initial energy
    - Ferroelectric distortion should be optimized
    - Cell parameters should relax to equilibrium values

Real-world use case:
    This approach is used for:
    - Optimizing complex ferroelectric structures
    - Finding stable phases at different temperatures
    - Studying domain walls and defects
    - Long MD simulations (ns-μs timescales)
"""

from ase import Atoms
from ase.io import read, write
from atomchain.init_model import init_calc
from atomchain.relax import relax_with_ml
import os

# Check if required files exist
required_files = [
    "BaHfO3_config.conf",
    "BaHfO3_DDB", 
    "BaHfO3.xml"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("ERROR: Missing required files:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nThis example requires MULTIBINIT potential files.")
    print("Please refer to ABINIT documentation for generating DDB and XML files.")
    print("See: https://docs.abinit.org/tutorial/lattice_model/")
    exit(1)

# ============================================================================
# 1. Create or load initial structure
# ============================================================================

# Option A: Create BaHfO3 structure from scratch (cubic perovskite)
def create_batio3_structure():
    """
    Create BaHfO3 primitive cell in cubic perovskite structure.
    Slightly distorted to demonstrate relaxation.
    """
    a = 4.00  # Lattice parameter (Angstrom)
    
    # Cubic perovskite structure
    cell = [[a, 0, 0],
            [0, a, 0],
            [0, 0, a]]
    
    # Atomic positions (fractional coordinates)
    # Ba at corner, Ti at body center, O at face centers
    positions = [
        [0.0, 0.0, 0.0],  # Ba
        [0.5, 0.5, 0.5],  # Ti
        [0.5, 0.5, 0.0],  # O
        [0.5, 0.0, 0.5],  # O
        [0.0, 0.5, 0.5],  # O
    ]
    
    # Add small distortion to Ti position (ferroelectric displacement)
    positions[1] = [0.52, 0.52, 0.52]  # Ti displaced by ~0.08 Å
    
    atoms = Atoms(
        symbols=['Ba', 'Hf', 'O', 'O', 'O'],  # Note: DDB has Hf, not Ti
        scaled_positions=positions,
        cell=cell,
        pbc=True
    )
    
    return atoms

# Option B: Load structure from file (if available)
# atoms = read('BaHfO3_initial.cif')

# Create initial structure
atoms = create_batio3_structure()
print(f"Initial structure created: {len(atoms)} atoms")
print(f"Chemical formula: {atoms.get_chemical_formula()}")
print(f"Initial cell volume: {atoms.get_volume():.3f} Å³")

# Save initial structure
write('BaHfO3_initial.vasp', atoms)
print("Saved initial structure to: BaHfO3_initial.vasp")

# ============================================================================
# 2. Initialize MULTIBINIT calculator
# ============================================================================

print("\n" + "="*70)
print("Initializing MULTIBINIT calculator...")
print("="*70)

# Method 1: Using AtomChain's unified interface (recommended)
calc = init_calc(model_type="multibinit", model_path="BaHfO3_config.conf")

# Alternatively, you can use pymultibinit directly:
# from pymultibinit import MultibinitCalculator
# calc = MultibinitCalculator.from_config_file("BaHfO3_config.conf")

print("✓ MULTIBINIT calculator initialized successfully")
print(f"  Calculator type: {type(calc).__name__}")

# ============================================================================
# 3. Calculate initial energy
# ============================================================================

atoms.calc = calc
initial_energy = atoms.get_potential_energy()
initial_forces = atoms.get_forces()
max_force = (initial_forces**2).sum(axis=1).max()**0.5

print(f"\nInitial state:")
print(f"  Energy: {initial_energy:.6f} eV")
print(f"  Max force: {max_force:.6f} eV/Å")

# ============================================================================
# 4. Perform structure relaxation
# ============================================================================

print("\n" + "="*70)
print("Starting structure relaxation...")
print("="*70)

# Relax both atomic positions and cell
relaxed_atoms = relax_with_ml(
    atoms=atoms,
    calc=calc,
    fmax=0.01,              # Force convergence criterion (eV/Å)
    relax_cell=True,        # Relax cell parameters
    traj_file='BaHfO3_relax.traj'  # Save optimization trajectory
)

print("✓ Relaxation completed")

# ============================================================================
# 5. Analyze results
# ============================================================================

print("\n" + "="*70)
print("Relaxation Results")
print("="*70)

final_energy = relaxed_atoms.get_potential_energy()
final_forces = relaxed_atoms.get_forces()
max_force_final = (final_forces**2).sum(axis=1).max()**0.5

print(f"\nEnergetics:")
print(f"  Initial energy: {initial_energy:.6f} eV")
print(f"  Final energy:   {final_energy:.6f} eV")
print(f"  Energy change:  {final_energy - initial_energy:.6f} eV")

print(f"\nForces:")
print(f"  Initial max force: {max_force:.6f} eV/Å")
print(f"  Final max force:   {max_force_final:.6f} eV/Å")

print(f"\nCell parameters:")
print(f"  Initial volume: {atoms.get_volume():.3f} Å³")
print(f"  Final volume:   {relaxed_atoms.get_volume():.3f} Å³")
print(f"  Volume change:  {(relaxed_atoms.get_volume() - atoms.get_volume()):.3f} Å³")

# Calculate Ti displacement (ferroelectric order parameter)
initial_ti_pos = atoms.get_scaled_positions()[1]  # Ti is second atom
final_ti_pos = relaxed_atoms.get_scaled_positions()[1]
ti_displacement = final_ti_pos - [0.5, 0.5, 0.5]
print(f"\nTi displacement from center (fractional coords):")
print(f"  Initial: [{initial_ti_pos[0]-0.5:.4f}, {initial_ti_pos[1]-0.5:.4f}, {initial_ti_pos[2]-0.5:.4f}]")
print(f"  Final:   [{ti_displacement[0]:.4f}, {ti_displacement[1]:.4f}, {ti_displacement[2]:.4f}]")

# ============================================================================
# 6. Save results
# ============================================================================

# Save relaxed structure in multiple formats
write('BaHfO3_relaxed.vasp', relaxed_atoms)
write('BaHfO3_relaxed.cif', relaxed_atoms)

print(f"\n{'='*70}")
print("Output files:")
print(f"{'='*70}")
print("  BaHfO3_initial.vasp      - Initial structure")
print("  BaHfO3_relaxed.vasp      - Relaxed structure (VASP format)")
print("  BaHfO3_relaxed.cif       - Relaxed structure (CIF format)")
print("  BaHfO3_relax.traj        - Full optimization trajectory")

print("\n" + "="*70)
print("Example completed successfully!")
print("="*70)

print("\nNext steps:")
print("  1. Visualize trajectory: ase gui BaHfO3_relax.traj")
print("  2. Compare structures: ase gui BaHfO3_initial.vasp BaHfO3_relaxed.vasp")
print("  3. Run phonon calculation: python multibinit_phonon.py")
