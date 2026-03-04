#!/bin/bash
# ============================================================================
# CLI Example: Structure Relaxation with MULTIBINIT
# ============================================================================
# 
# This script demonstrates using AtomChain's CLI tools with MULTIBINIT
# for structure relaxation of BaHfO3.
#
# Usage:
#   bash cli_relaxation.sh
#
# Prerequisites:
#   - BaHfO3_config.conf, BaHfO3_DDB, BaHfO3.xml in current directory
#   - BaHfO3_initial.vasp (initial structure)
#   - mlrelax command available in PATH

set -e  # Exit on error

echo "========================================================================"
echo "AtomChain CLI: MULTIBINIT Relaxation"
echo "========================================================================"
echo ""

# Check if required files exist
if [ ! -f "BaHfO3_config.conf" ]; then
    echo "ERROR: BaHfO3_config.conf not found"
    exit 1
fi

if [ ! -f "BaHfO3_initial.vasp" ]; then
    echo "ERROR: BaHfO3_initial.vasp not found"
    echo "Run: python multibinit_relaxation.py first to create initial structure"
    exit 1
fi

echo "Input files:"
echo "  Structure: BaHfO3_initial.vasp"
echo "  Config:    BaHfO3_config.conf"
echo ""

# ============================================================================
# Relaxation with MULTIBINIT calculator
# ============================================================================

echo "========================================================================"
echo "Running structure relaxation..."
echo "========================================================================"
echo ""

# Note: The mlrelax CLI currently doesn't directly support passing
# config files for MULTIBINIT. This is a Python API-focused feature.
# 
# For CLI usage, you would typically need to:
# 1. Use the Python API (recommended)
# 2. Or create a wrapper script

echo "For MULTIBINIT relaxation, use the Python API:"
echo ""
echo "  from atomchain.init_model import init_calc"
echo "  from atomchain.relax import relax_with_ml"
echo "  from ase.io import read"
echo ""
echo "  atoms = read('BaHfO3_initial.vasp')"
echo "  calc = init_calc('multibinit', model_path='BaHfO3_config.conf')"
echo "  relaxed = relax_with_ml(atoms, calc=calc)"
echo ""

echo "Or run the provided Python script:"
echo "  python multibinit_relaxation.py"
echo ""

# ============================================================================
# Alternative: Using pre-initialized calculator
# ============================================================================

echo "========================================================================"
echo "Note: CLI Integration"
echo "========================================================================"
echo ""
echo "The mlrelax and mlphonon CLI tools are designed for simpler use cases"
echo "where you can specify the calculator type directly."
echo ""
echo "For MULTIBINIT (which requires config files), the Python API provides"
echo "more flexibility and is the recommended approach."
echo ""
echo "Example with other calculators:"
echo "  mlrelax structure.vasp --calc chgnet"
echo "  mlphonon structure.vasp --calc mace"
echo ""

echo "========================================================================"
echo "For MULTIBINIT, use Python scripts in this directory:"
echo "========================================================================"
echo "  1. multibinit_relaxation.py  - Full relaxation example"
echo "  2. multibinit_phonon.py      - Phonon calculation example"
echo ""
