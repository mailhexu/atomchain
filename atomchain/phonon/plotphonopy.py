#!/usr/bin/env python3
"""
Phonon band structure plotting utilities using ASE and Phonopy.

This module provides functions to plot phonon band structures using
ASE's built-in Brillouin zone path generation and modern Phonopy API.
"""

import os
import numpy as np
import matplotlib
# Use Agg backend for non-interactive plotting (fixes macOS crashes)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phonopy import load
from ase.cell import Cell
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


def group_band_path(bp, eps=1e-8, shift=0.15):
    """
    Group band path into segments, handling discontinuities.
    
    This function detects breaks in the k-path where consecutive points
    are disconnected and splits them into separate segments with visual shifts.
    
    Args:
        bp: ASE BandPath object
        eps: Threshold for detecting discontinuities (default: 1e-8)
        shift: Visual shift between segments (default: 0.15)
    
    Returns:
        xlist: List of x-coordinate arrays (one per segment)
        kptlist: List of k-point arrays (one per segment)
        Xs: Adjusted special point x-coordinates
        knames: Special point labels
    """
    xs, Xs, knames = bp.get_linear_kpoint_axis()
    kpts = bp.kpts
    
    # Detect discontinuities in the path
    m = xs[1:] - xs[:-1] < eps
    segments = [0] + list(np.where(m)[0] + 1) + [len(xs)]
    
    # Split into segments and add shifts
    xlist, kptlist = [], []
    for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
        kptlist.append(kpts[start:end])
        xlist.append(xs[start:end] + i * shift)
    
    # Adjust special point positions
    m = Xs[1:] - Xs[:-1] < eps
    s = np.where(m)[0] + 1
    for i in s:
        Xs[i:] += shift
    
    return xlist, kptlist, Xs, knames


def plot_phonon(
    path="./",
    kpath=None,
    npoints=100,
    color="blue",
    figname="phonon.pdf",
    show=True,
    units="cm-1",
):
    """
    Plot phonon band structure using ASE for k-path generation.

    This function loads phonopy calculation results and plots the phonon
    band structure along high-symmetry paths in the Brillouin zone.

    Args:
        path: Directory containing phonopy_params.yaml file (default: "phonon_save")
        kpath: String specifying k-path (e.g., "GXMG" for cubic).
               If None, ASE automatically determines appropriate path based on crystal structure.
        npoints: Number of points along the k-path for interpolation (default: 100)
        color: Color for the band structure lines (default: "blue")
        figname: Output filename for the plot (default: "phonon.pdf").
                 Set to None to skip saving.
        show: Whether to display the plot interactively (default: True)
        units: Frequency units - "cm-1" (default) or "THz"

    Returns:
        None

    Example:
        >>> # Automatic k-path detection
        >>> plot_phonon(path="phonon_save")

        >>> # Custom k-path for FCC structure
        >>> plot_phonon(path="phonon_save", kpath="GXWKGLUWLK")

        >>> # Just save, don't show
        >>> plot_phonon(path="phonon_save", kpath="GXMG", show=False)

        >>> # Use THz units
        >>> plot_phonon(path="phonon_save", units="THz")
    """
    # Load phonopy data
    phonopy_yaml = os.path.join(path, "phonopy_params.yaml")
    if not os.path.exists(phonopy_yaml):
        raise FileNotFoundError(
            f"Phonopy data file not found: {phonopy_yaml}\n"
            f"Make sure phonon calculation has been performed and output is in '{path}' directory."
        )

    print(f"[Phonopy] Loading phonopy data from {phonopy_yaml}")
    phonon = load(phonopy_yaml=phonopy_yaml)

    # Get primitive cell for k-path generation
    cell = Cell(phonon.primitive.cell)

    # Generate k-path using ASE
    # ASE's bandpath automatically determines Bravais lattice and appropriate path
    if kpath is None:
        print("[Phonopy] Automatically determining k-path based on crystal structure")
        bandpath = cell.bandpath(npoints=npoints)
    else:
        print(f"[Phonopy] Using specified k-path: {kpath}")
        kpath_normalized = kpath.replace("Γ", "G")
        bandpath = cell.bandpath(path=kpath_normalized, npoints=npoints)

    # Group band path into segments (handles discontinuities)
    xlist, kptlist, Xs, knames = group_band_path(bandpath)
    
    # Replace 'G' with Greek gamma symbol
    knames = [("Γ" if label == "G" else label) for label in knames]

    print(f"[Phonopy] K-path: {' → '.join(knames)}")
    print(f"[Phonopy] Path split into {len(xlist)} segment(s)")

    # Symmetrize force constants for better results
    phonon.symmetrize_force_constants()

    # Calculate phonons at each k-point using phonopy's dynamical matrix
    print("[Phonopy] Calculating phonon frequencies at k-points")
    all_kpoints = np.concatenate(kptlist)
    n_kpoints = len(all_kpoints)
    n_atoms = len(phonon.primitive)
    n_modes = n_atoms * 3

    frequencies = np.zeros((n_kpoints, n_modes))
    
    for i, q in enumerate(all_kpoints):
        dm = phonon.get_dynamical_matrix_at_q(q)
        eigenvalues, _ = np.linalg.eigh(dm)
        # Handle imaginary frequencies (negative eigenvalues)
        frequencies[i] = (
            np.sqrt(np.abs(eigenvalues))
            * np.sign(eigenvalues)
            * phonon.unit_conversion_factor
        )

    # Convert units if needed (phonopy returns THz by default)
    # Conversion: 1 THz = 33.35641 cm^-1
    if units.lower() == "cm-1":
        frequencies = frequencies * 33.35641
        ylabel = "Frequency (cm$^{-1}$)"
    else:
        ylabel = "Frequency (THz)"

    # Create k-path labels (match k-points to special points)
    special_points = bandpath.special_points
    kpath_labels = []
    current_pos = 0
    
    for i, kpts_segment in enumerate(kptlist):
        for name in knames:
            if name == "Γ":
                name_lookup = "G"
            else:
                name_lookup = name
            
            if name_lookup in special_points:
                spk = special_points[name_lookup]
                # Find matches in current segment (compare each k-point to special point)
                for kpt_idx, kpt in enumerate(kpts_segment):
                    if np.allclose(kpt, spk, atol=1e-5):
                        global_idx = current_pos + kpt_idx
                        kpath_labels.append((global_idx, name))
        current_pos += len(kpts_segment)
    
    # Remove duplicates and sort
    kpath_labels = sorted(set(kpath_labels), key=lambda x: x[0])

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot phonon bands by segment
    for seg_idx, x_segment in enumerate(xlist):
        # Get frequency data for this segment
        start_idx = sum(len(s) for s in xlist[:seg_idx])
        end_idx = start_idx + len(x_segment)
        freq_segment = frequencies[start_idx:end_idx]
        
        # Plot each band in this segment
        for band_idx in range(n_modes):
            ax.plot(x_segment, freq_segment[:, band_idx], color=color, linewidth=1.5)

    # Add vertical gray lines at high-symmetry points
    for pos in Xs:
        ax.axvline(x=pos, color="gray", linestyle="-", linewidth=0.8, alpha=0.6)

    # Convert label indices to x-coordinates
    label_x_positions = []
    for idx, name in kpath_labels:
        # Find which segment this index belongs to
        current = 0
        for seg_idx, x_seg in enumerate(xlist):
            if current + len(x_seg) > idx:
                local_idx = idx - current
                label_x_positions.append((x_seg[local_idx], name))
                break
            current += len(x_seg)

    # Set x-axis labels at high-symmetry points
    if label_x_positions:
        label_x, label_names = zip(*label_x_positions)
        ax.set_xticks(label_x)
        ax.set_xticklabels(label_names, fontsize=12)

    # Labels and formatting
    ax.set_xlabel("k-path", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Set x-limits to show full path
    all_x = np.concatenate(xlist)
    ax.set_xlim(all_x[0], all_x[-1])
    
    # Smart auto-scaling: include full frequency range (including imaginary modes)
    min_freq = np.min(frequencies)
    max_freq = np.max(frequencies)
    freq_range = abs(max_freq - min_freq)
    margin = 0.05 * freq_range  # 5% margin on each side
    
    # Ensure minimum margin based on units
    if units.lower() == "cm-1":
        margin = max(margin, 5.0)  # Minimum 5 cm-1 margin
    else:  # THz
        margin = max(margin, 0.01)  # Minimum 0.01 THz margin
    
    ax.set_ylim(min_freq - margin, max_freq + margin)
    
    # Add y=0 reference line for imaginary modes
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Debug info
    print(f"[Phonopy] X-coordinate range: [{all_x[0]:.3f}, {all_x[-1]:.3f}]")
    print(f"[Phonopy] Frequency range: [{min_freq:.3f}, {max_freq:.3f}] {units}")
    print(f"[Phonopy] Special point positions: {[f'{p:.3f}' for p in Xs]}")
    print(f"[Phonopy] Labels: {knames}")

    plt.tight_layout()

    # Save figure if requested
    if figname is not None:
        output_file = os.path.join(path, figname)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"[Phonopy] Plot saved to {output_file}")

    # Show plot if requested
    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    plot_phonon()
