#!/usr/bin/env python3
"""
Quick test to verify phonon plotting with labels and vertical lines.

This script demonstrates the new plotting features:
- High-symmetry point labels on x-axis (Γ, X, M, etc.)
- Gray vertical lines at high-symmetry points
- Clean band structure visualization

Usage:
    # From the atomchain repository root:
    cd examples/02_phonon
    python test_plot_features.py
    
    # Or install atomchain first:
    pip install -e /path/to/atomchain
    python test_plot_features.py
    
Note: Requires an existing phonon calculation in phonon_save/ directory
"""

from atomchain.phonon.plotphonopy import plot_phonon

# Test plotting with the updated features
print("Testing phonon plot with labels and vertical lines...")
print("This assumes you have phonon data in phonon_save/ directory")
print()

try:
    plot_phonon(
        path="phonon_save",
        kpath="GXMG",  # Simple cubic path
        figname="test_phonon_plot.png",
        show=True,
        color="blue"
    )
    print("\n✓ Plot generated successfully!")
    print("Check test_phonon_plot.png for:")
    print("  - High-symmetry point labels (Γ, X, M, G) on x-axis")
    print("  - Gray vertical lines at each high-symmetry point")
    print("  - Clean band structure visualization")
except FileNotFoundError as e:
    print(f"\n✗ Error: {e}")
    print("\nTo use this test:")
    print("1. First run a phonon calculation (see examples/02_phonon/)")
    print("2. Make sure phonon_save/ directory exists with phonopy_params.yaml")
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
