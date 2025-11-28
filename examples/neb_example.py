"""
NEB calculation example for finding transition states and energy barriers.

This example demonstrates how to use atomchain's NEB functionality to:
1. Find the minimum energy path between two atomic configurations
2. Calculate activation energy barriers
3. Visualize the energy profile

Purpose:
    Shows basic usage of calculate_neb() function and mlneb CLI tool

How to run:
    python neb_example.py

What it does:
    - Creates two Al FCC structures (perfect and slightly compressed)
    - Performs CI-NEB calculation to find the transition path
    - Generates energy profile plot
    - Saves the converged NEB path

Expected behavior:
    - Should complete in ~1-2 minutes on a laptop
    - Forward barrier should be ~0.01-0.1 eV for this simple case
    - Creates neb_path.traj and barrier_profile.png
"""

from ase.build import bulk
from atomchain.neb import calculate_neb

# Create initial structure: Perfect Al FCC
initial = bulk('Al', 'fcc', a=4.05, cubic=True)
print(f"Initial structure: {initial.get_chemical_formula()}")
print(f"Initial cell volume: {initial.get_volume():.2f} Å³")

# Create final structure: Slightly compressed along one axis
final = initial.copy()
cell = final.get_cell().array.copy()
cell[0] *= 0.95  # 5% compression along x-axis
final.set_cell(cell, scale_atoms=True)
print(f"\nFinal structure: {final.get_chemical_formula()}")
print(f"Final cell volume: {final.get_volume():.2f} Å³")
print(f"Volume change: {(final.get_volume() - initial.get_volume()):.2f} Å³")

# Perform NEB calculation
print("\n" + "="*60)
print("Running NEB calculation...")
print("="*60)

results = calculate_neb(
    initial=initial,
    final=final,
    calculator='chgnet',  # Use CHGNet ML potential
    nimages=7,            # 7 total images (5 intermediate)
    fmax=0.05,            # Force convergence criterion (eV/Å)
    optimizer='FIRE',     # Optimizer algorithm
    max_steps=500,        # Maximum optimization steps
    output='neb_path.traj',       # Save converged path
    plot='barrier_profile.png',   # Save energy profile plot
    show=False,           # Don't show plot interactively
)

# Print results
print("\n" + "="*60)
print("NEB Results")
print("="*60)
print(f"Converged: {results['converged']}")
print(f"Number of images: {len(results['images'])}")
print(f"Transition state: Image {results['ts_index']}")
print(f"\nEnergy Barriers:")
print(f"  Forward:  {results['barrier_forward']:.4f} eV")
print(f"  Reverse:  {results['barrier_reverse']:.4f} eV")
print(f"\nReaction Energy:")
print(f"  ΔE = {results['reaction_energy']:.4f} eV")
print(f"\nEnergy Profile:")
for i, energy in enumerate(results['energies']):
    marker = " <-- TS" if i == results['ts_index'] else ""
    print(f"  Image {i}: {energy:.6f} eV{marker}")

print("\n" + "="*60)
print("Output files:")
print("  - neb_path.traj: Converged NEB path trajectory")
print("  - barrier_profile.png: Energy profile visualization")
print("="*60)

# To use from command line instead:
print("\nCommand-line equivalent:")
print("  mlneb initial.vasp final.vasp \\")
print("        --calc chgnet --nimages 7 --fmax 0.05 \\")
print("        -o neb_path.traj -p barrier_profile.png")
