#!/bin/bash
# ============================================================================
# CLI Example: Phonon Calculation with MULTIBINIT
# ============================================================================
# 
# This script demonstrates using AtomChain's CLI tools with MULTIBINIT
# for phonon band structure calculation of BaHfO3.
#
# Usage:
#   bash cli_phonon.sh
#
# Prerequisites:
#   - BaHfO3_config.conf, BaHfO3_DDB, BaHfO3.xml in current directory
#   - BaHfO3_relaxed.vasp (relaxed structure)
#   - mlphonon command available in PATH

set -e  # Exit on error

echo "========================================================================"
echo "AtomChain CLI: MULTIBINIT Phonon Calculation"
echo "========================================================================"
echo ""

# Check if required files exist
if [ ! -f "BaHfO3_config.conf" ]; then
    echo "ERROR: BaHfO3_config.conf not found"
    exit 1
fi

if [ ! -f "BaHfO3_relaxed.vasp" ]; then
    echo "WARNING: BaHfO3_relaxed.vasp not found"
    echo "Using BaHfO3_initial.vasp instead (not recommended)"
    
    if [ ! -f "BaHfO3_initial.vasp" ]; then
        echo "ERROR: No structure file found"
        echo "Run: python multibinit_relaxation.py first"
        exit 1
    fi
    STRUCTURE="BaHfO3_initial.vasp"
else
    STRUCTURE="BaHfO3_relaxed.vasp"
fi

echo "Input files:"
echo "  Structure: $STRUCTURE"
echo "  Config:    BaHfO3_config.conf"
echo ""

# ============================================================================
# Phonon calculation with MULTIBINIT
# ============================================================================

echo "========================================================================"
echo "Running phonon calculation..."
echo "========================================================================"
echo ""

# Note: The mlphonon CLI currently doesn't directly support passing
# config files for MULTIBINIT. This is a Python API-focused feature.

echo "For MULTIBINIT phonon calculations, use the Python API:"
echo ""
echo "  from atomchain.init_model import init_calc"
echo "  from atomchain.phonon import phonon_with_ml"
echo "  from ase.io import read"
echo ""
echo "  atoms = read('$STRUCTURE')"
echo "  calc = init_calc('multibinit', model_path='BaHfO3_config.conf')"
echo "  phonon_with_ml(atoms, calc=calc, supercell=[2, 2, 2])"
echo ""

echo "Or run the provided Python script:"
echo "  python multibinit_phonon.py"
echo ""

# ============================================================================
# Show example output
# ============================================================================

echo "========================================================================"
echo "Expected output files:"
echo "========================================================================"
echo "  phonon_multibinit/band.yaml           - Band structure data"
echo "  phonon_multibinit/total_dos.dat       - Density of states"
echo "  phonon_multibinit/phonon_band.pdf     - Band structure plot"
echo "  phonon_multibinit/phonon_dos.pdf      - DOS plot"
echo "  phonon_multibinit/force_constants.hdf5 - Force constants"
echo ""

echo "========================================================================"
echo "Note: CLI vs Python API"
echo "========================================================================"
echo ""
echo "The mlphonon CLI works well for simple cases:"
echo "  mlphonon structure.vasp --calc chgnet --supercell 2 2 2"
echo ""
echo "For MULTIBINIT (requiring config files), use the Python API for"
echo "better control and flexibility."
echo ""
