#!/usr/bin/env python
"""
Convert between different structure file formats using ASE.

This module provides functionality to convert atomic structures between
various file formats supported by ASE (e.g., VASP, LAMMPS, CIF, XYZ, etc.).
"""

from __future__ import annotations

import argparse
from typing import Optional
from ase.io import read, write


def convert_structure(
    input_file: str,
    output_file: str,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
    index: str = ":",
):
    """
    Convert structure files between different formats.

    Args:
        input_file: Path to input file
        output_file: Path to output file
        input_format: Input format (None = auto-detect from extension)
        output_format: Output format (None = auto-detect from extension)
        index: Frame index for trajectory files (default: ":" = all frames)
               Examples: "0" (first), "-1" (last), "::2" (every 2nd)

    Example:
        >>> # Convert POSCAR to CIF
        >>> convert_structure('POSCAR', 'structure.cif')
        
        >>> # Convert trajectory to LAMMPS data
        >>> convert_structure('md.traj', 'data.lammps', index='-1')
        
        >>> # Convert with explicit formats
        >>> convert_structure('input', 'output', 
        ...                   input_format='vasp', 
        ...                   output_format='xyz')
    """
    # Read structure(s)
    structures = read(input_file, index=index, format=input_format)
    
    # Write structure(s)
    write(output_file, structures, format=output_format)
    
    # Print summary
    if isinstance(structures, list):
        print(f"Converted {len(structures)} structures:")
    else:
        print(f"Converted 1 structure:")
    
    print(f"  Input:  {input_file}")
    if input_format:
        print(f"          (format: {input_format})")
    print(f"  Output: {output_file}")
    if output_format:
        print(f"          (format: {output_format})")


def mlconvert_cli():
    """
    Command-line interface for structure file format conversion.

    This CLI tool converts atomic structures between different file formats
    supported by ASE, including VASP, LAMMPS, CIF, XYZ, trajectory files, etc.

    Example:
        $ mlconvert POSCAR structure.cif
        $ mlconvert md.traj data.lammps --index -1
        $ mlconvert input.xyz output.vasp --input-format xyz --output-format vasp
        $ mlconvert trajectory.traj all_frames.xyz --index ":"
    """
    parser = argparse.ArgumentParser(
        description="Convert structure files between different formats using ASE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats (auto-detected from file extension):
  VASP:      POSCAR, CONTCAR, .vasp
  LAMMPS:    .lammps, data.*, *.data
  CIF:       .cif
  XYZ:       .xyz
  Trajectory: .traj
  PDB:       .pdb
  And many more formats supported by ASE

Index syntax for trajectory files:
  ":"      All frames (default)
  "0"      First frame
  "-1"     Last frame
  "::2"    Every 2nd frame
  "0:10"   First 10 frames

Examples:
  # Convert POSCAR to CIF
  mlconvert POSCAR structure.cif

  # Extract last frame from trajectory to LAMMPS
  mlconvert md.traj final.lammps --index -1

  # Convert all frames to XYZ
  mlconvert trajectory.traj all_frames.xyz

  # Specify formats explicitly
  mlconvert input output --input-format vasp --output-format xyz
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input structure file"
    )
    parser.add_argument(
        "output_file",
        help="Output structure file"
    )
    parser.add_argument(
        "--input-format",
        "-if",
        help="Input format (default: auto-detect from extension)",
        default=None,
    )
    parser.add_argument(
        "--output-format",
        "-of",
        help="Output format (default: auto-detect from extension)",
        default=None,
    )
    parser.add_argument(
        "--index",
        "-i",
        help='Frame index for trajectory files (default: ":" = all frames)',
        default=":",
    )

    args = parser.parse_args()

    try:
        convert_structure(
            input_file=args.input_file,
            output_file=args.output_file,
            input_format=args.input_format,
            output_format=args.output_format,
            index=args.index,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(mlconvert_cli())
