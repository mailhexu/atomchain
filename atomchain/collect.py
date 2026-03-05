"""
Module for collecting atomic structure files into a trajectory and JSON list.
"""

import json
import os
import argparse
from typing import List, Optional

from ase.io import read, write
from ase import Atoms


def collect_structures(
    root_dir: str, extension: str, output_traj: str, output_json: str
) -> None:
    """
    Recursively collect atomic structure files from a directory and save them
    to a trajectory file and a JSON list of paths.

    Args:
        root_dir: The root directory to search.
        extension: The file extension to look for (e.g., '.vasp', '.cif').
        output_traj: Path to the output .traj file.
        output_json: Path to the output .json file.
    """
    collected_atoms: List[Atoms] = []
    collected_paths: List[str] = []

    # Normalize extension (ensure it starts with .)
    # if not extension.startswith("."):
    #    extension = "." + extension
    # REMOVED normalization to allow matching filenames like POSCAR
    pass

    # Walk through directory
    for root, _, files in os.walk(root_dir):
        # Sort files to ensure deterministic order within directory
        for filename in sorted(files):
            if filename.endswith(extension):
                file_path = os.path.join(root, filename)
                try:
                    # Read structure
                    atoms = read(file_path)
                    # Handle case where read returns list (e.g. multiple frames)
                    # We only take the first one or all? Proposal implies "structures"
                    # usually implies one per file for things like VASP/CIF,
                    # but read() can return list.
                    # Let's assume one structure per file for now, or append all.
                    # The proposal says "put the structures... into a traj file".
                    # If a file has multiple, we probably want all.
                    # But the json path logic needs to match frames.
                    # If we have multiple frames in one file, do we put the path multiple times?
                    # The proposal says "paths of the structure files".
                    # Let's assume standard single-structure files for now or flattened list.
                    # If read returns a list, we add all and add path for each.

                    if isinstance(atoms, list):
                        for a in atoms:
                            collected_atoms.append(a)
                            collected_paths.append(file_path)
                    else:
                        collected_atoms.append(atoms)
                        collected_paths.append(file_path)

                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
                    continue

    if not collected_atoms:
        print(f"No valid structures found with extension '{extension}' in '{root_dir}'.")
        return

    # Write outputs
    write(output_traj, collected_atoms)
    print(f"Wrote {len(collected_atoms)} structures to {output_traj}")

    with open(output_json, "w") as f:
        json.dump(collected_paths, f, indent=2)
    print(f"Wrote paths to {output_json}")


def mlcollect_cli():
    """CLI entry point for mlcollect."""
    parser = argparse.ArgumentParser(
        description="Collect atomic structures from a directory into a trajectory."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--extension",
        "-e",
        required=True,
        help="File extension or suffix to look for (e.g., .vasp, POSCAR)",
    )
    parser.add_argument(
        "--output-traj",
        "-o",
        default="collected.traj",
        help="Output trajectory file (default: collected.traj)",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        default="collected_paths.json",
        help="Output JSON file with paths (default: collected_paths.json)",
    )

    args = parser.parse_args()

    collect_structures(
        args.root_dir, args.extension, args.output_traj, args.output_json
    )


if __name__ == "__main__":
    mlcollect_cli()
