"""
MULTIBINIT Phonon Calculation Example: BaHfO3 (Barium Hafnate)
================================================================

Purpose:
    Demonstrate phonon band structure and DOS calculation using MULTIBINIT
    effective potential for ferroelectric BaHfO3.

What this example does:
    1. Loads relaxed BaHfO3 structure (or creates one)
    2. Initializes MULTIBINIT calculator from configuration file
    3. Calculates phonon band structure and density of states
    4. Plots and saves results

How to run:
    python multibinit_phonon.py

Prerequisites:
    - pymultibinit installed and configured
    - BaHfO3_DDB and BaHfO3.xml files in current directory
      (Note: Despite the filename, these contain BaHfO3 data - Hf not Ti!)
    - Configuration file: BaHfO3_config.conf
    - (Optional) Relaxed structure from multibinit_relaxation.py

Expected behavior:
    - Phonon calculation should complete in seconds to minutes
    - Soft modes near Γ point indicate ferroelectric instability
    - High-frequency modes correspond to oxygen vibrations
    - Results saved as phonon_band.pdf and phonon_dos.pdf

Real-world use case:
    Phonon calculations with MULTIBINIT are used for:
    - Studying structural phase transitions
    - Identifying soft modes and instabilities
    - Calculating thermodynamic properties
    - Understanding ferroelectric mechanisms
    - Much faster than DFT for large supercells
"""

from ase import Atoms
from ase.io import read, write
from atomchain.init_model import init_calc
from atomchain.phonon import phonon_with_ml
import os
import numpy as np

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
# 1. Load structure
# ============================================================================

print("="*70)
print("MULTIBINIT Phonon Calculation for BaHfO3")
print("="*70)

# Try to load relaxed structure first, otherwise create/load initial
if os.path.exists('BaHfO3_ref.vasp'):
    atoms_list = read('BaHfO3_ref.vasp')
    atoms = atoms_list if isinstance(atoms_list, Atoms) else atoms_list[0]
    print("Loaded relaxed structure from: BaHfO3_relaxed.vasp")
else:
    print("Relaxed structure not found. Creating cubic BaHfO3 structure...")
    
    # Create BaHfO3 primitive cell
    a = 4.00  # Lattice parameter (Angstrom)
    cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    
    positions = [
        [0.0, 0.0, 0.0],  # Ba
        [0.5, 0.5, 0.5],  # Hf
        [0.5, 0.5, 0.0],  # O
        [0.5, 0.0, 0.5],  # O
        [0.0, 0.5, 0.5],  # O
    ]
    
    atoms = Atoms(
        symbols=['Ba', 'Hf', 'O', 'O', 'O'],  
        scaled_positions=positions,
        cell=cell,
        pbc=True
    )
    print("  Note: For best results, run multibinit_relaxation.py first")

print(f"Structure: {len(atoms)} atoms")
print(f"Chemical formula: {atoms.get_chemical_formula()}")
print(f"Cell volume: {atoms.get_volume():.3f} Å³")

# ============================================================================
# 2. Initialize MULTIBINIT calculator (optional - can pass directly to phonon_with_ml)
# ============================================================================

print("\n" + "="*70)
print("Setting up MULTIBINIT calculation...")
print("="*70)

# Option 1: Initialize calculator explicitly (if you need to reuse it)
# calc = init_calc(model_type="multibinit", model_path="BaHfO3_config.conf")

# Option 2: Pass model_type and model_path directly to phonon_with_ml (recommended)
# This is simpler and the calculator will be created automatically

print("✓ MULTIBINIT will be initialized automatically during phonon calculation")

# ============================================================================
# 3. Set up phonon calculation parameters
# ============================================================================

print("\n" + "="*70)
print("Setting up phonon calculation...")
print("="*70)

# Supercell for finite displacement method
# Larger supercells give more accurate phonon frequencies
# Trade-off: accuracy vs. computational cost
supercell_matrix = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]  # 2×2×2 supercell → 40 atoms

print(f"Supercell: {supercell_matrix[0][0]}×{supercell_matrix[1][1]}×{supercell_matrix[2][2]}")
print(f"Total atoms in supercell: {len(atoms) * np.prod([supercell_matrix[i][i] for i in range(3)])}")

# Displacement magnitude for finite differences
displacement = 0.01  # Angstrom

print(f"Finite displacement: {displacement} Å")

# ============================================================================
# 4. Calculate phonon band structure and DOS
# ============================================================================

print("\n" + "="*70)
print("Calculating phonon band structure and DOS...")
print("="*70)
print("This may take a few minutes depending on supercell size...")

# Perform phonon calculation
# Pass model_type and model_path directly - no need to create calculator first!
phonon_with_ml(
    atoms=atoms,
    calc="multibinit",              # Model type (can also use "mb" as shorthand)
    model_path="BaHfO3_config.conf", # Path to configuration file
    supercell=supercell_matrix,
    displacement=displacement,
    npoints=100,                    # Points along band path
    use_seek_path=True,             # Automatically find k-path
    save_folder='phonon_multibinit', # Output directory
    relax=False,
)

print("\n✓ Phonon calculation completed successfully!")

# ============================================================================
# 5. Analyze results
# ============================================================================

print("\n" + "="*70)
print("Results")
print("="*70)

output_dir = 'phonon_multibinit'

print(f"\nOutput files in '{output_dir}/':")
print("  band.yaml                - Phonon band structure data")
print("  total_dos.dat            - Phonon density of states")
print("  phonon_band.pdf          - Band structure plot")
print("  phonon_dos.pdf           - DOS plot")
print("  phonopy_disp.yaml        - Displacement configuration")
print("  force_constants.hdf5     - Force constants matrix")

print("\n" + "="*70)
print("Physical Interpretation for BaHfO3:")
print("="*70)
print("""
Expected features in phonon spectrum:

1. Soft modes near Γ point (zone center):
   - If present, indicate ferroelectric instability
   - These modes drive the cubic → tetragonal transition
   - Frequency should be small or imaginary in cubic phase

2. Three acoustic branches:
   - Start at zero frequency at Γ point
   - Correspond to long-wavelength sound waves

3. Optical branches:
   - Higher frequency modes
   - Lower frequency: Ba, Ti motions (~100-200 cm⁻¹)
   - Higher frequency: O motions (~400-800 cm⁻¹)

4. Phonon DOS peaks:
   - Low frequency: Heavy Ba motion
   - Mid frequency: Ti motion and mixed modes
   - High frequency: Light O motion

Note: MULTIBINIT captures anharmonic effects that DFT-based
harmonic approximations miss, important for ferroelectrics!
""")

# ============================================================================
# 6. Next steps
# ============================================================================

print("="*70)
print("Next steps:")
print("="*70)
print("""
1. Visualize phonon band structure:
   ase gui phonon_multibinit/phonon_band.pdf

2. Check for imaginary frequencies (soft modes):
   Look for negative values in band.yaml

3. Calculate thermodynamic properties:
   - Helmholtz free energy
   - Heat capacity
   - Entropy
   Using phonopy tools with the generated force constants

4. Analyze specific phonon modes:
   Use phonopy to visualize atomic displacements
   phonopy --anime=MODE_NUMBER
""")

print("\n" + "="*70)
print("Example completed successfully!")
print("="*70)
