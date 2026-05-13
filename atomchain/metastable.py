"""Metastable states exploration via imaginary phonon mode instabilities.

This module implements a systematic approach to discovering metastable
structures from a high-symmetry parent:

1. Compute phonons to find imaginary (unstable) modes
2. Label modes with irreps using symphon (BCS/Mulliken labels)
3. Generate all combinations of up to *nmax* imaginary modes
4. For each combination, displace along all order parameter
   directions (OPDs) — maximal and optionally non-maximal
5. Relax each displaced structure to a local minimum
6. Optionally compute phonons for each relaxed structure

The main entry point is :func:`explore_metastable_states`.

The CLI entry point is :func:`atomchain.metastable_cli.mlmetastable_cli`.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import spglib
from ase.io import write

from atomchain.commensurate import find_commensurate_matrix
from atomchain.init_model import init_calc
from atomchain.kpoints import get_high_symmetry_kpoints
from atomchain.modulate import (
    get_modulations_with_opd_info,
    get_multi_mode_modulations_with_info,
)
from atomchain.phonon.frozenphonon import calculate_phonon
from atomchain.phonon.irreps import get_all_labeled_modes, get_imaginary_modes
from atomchain.relax import relax_with_ml


def _get_spacegroup(atoms, symprec=0.1):
    """Return (spacegroup_number, international_symbol) or (None, None)."""
    cell = (
        atoms.get_cell(),
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return None, None
    return dataset.number, dataset.international


def _build_label_to_qpoint(atoms, symprec=1e-5):
    """Build mapping from high-symmetry k-point label to fractional coordinates."""
    hs_kpoints = get_high_symmetry_kpoints(atoms, symprec=symprec)
    mapping = {}
    for entry in hs_kpoints:
        label = entry["label"]
        qpoint = tuple(np.round(entry["qpoint"], 8))
        mapping[label] = qpoint
    mapping["Γ"] = (0.0, 0.0, 0.0)
    mapping["G"] = (0.0, 0.0, 0.0)
    mapping["Gamma"] = (0.0, 0.0, 0.0)
    return mapping


def _deduplicate_kpoints(kpoints):
    """Remove duplicate k-points (by rounding to 8 decimals)."""
    seen = set()
    unique = []
    for k in kpoints:
        key = tuple(np.round(k, 8))
        if key not in seen:
            seen.add(key)
            unique.append(k)
    return unique


def _deduplicate_modes(modes):
    """Remove duplicate modes by (kpoint_label, rounded frequency)."""
    seen = set()
    unique = []
    for m in modes:
        key = (m["kpoint_label"], round(m["frequency"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _build_combined_label(source_modes, opd_labels=None):
    """Build human-readable label like 'GM4-(a,0,0)/X5+(a,b)' from modes and OPD labels."""
    parts = []
    for i, m in enumerate(source_modes):
        irrep = m.get("bcs_label") or m.get("kpoint_label", "?")
        if opd_labels and i < len(opd_labels) and opd_labels[i]:
            parts.append(f"{irrep}{opd_labels[i]}")
        else:
            parts.append(irrep)
    return "/".join(parts)


def _energy_per_fu(energy, n_atoms, n_atoms_per_fu):
    """Convert total energy to per-formula-unit energy."""
    return energy / (n_atoms / n_atoms_per_fu)


def _deduplicate_results(results, energy_tol=1e-3):
    """Remove duplicate relaxed structures by (spacegroup, energy, n_atoms)."""
    seen = set()
    unique = []
    for r in results:
        sg = r.get("spacegroup_number")
        e = r.get("energy_per_fu")
        n = r.get("n_atoms")
        if sg is None or e is None:
            unique.append(r)
            continue
        bucket_e = round(e / energy_tol) * energy_tol
        key = (sg, bucket_e, n)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def explore_metastable_states(
    atoms,
    calc=None,
    model_path=None,
    nmax=2,
    amplitude=0.5,
    max_cell_size=64,
    phonon_kwargs=None,
    relax_kwargs=None,
    output_dir="metastable_results",
    phonon_ndim=None,
    symprec=1e-5,
    include_nonmaximal=True,
    deduplicate=True,
    compute_phonons=False,
) -> list[dict]:
    """Discover metastable structures from imaginary phonon modes.

    Given a high-symmetry parent structure, this function:

    1. Computes phonons to find imaginary (unstable) modes
    2. Labels each mode with its irrep (e.g. GM4-, R5-, X5+)
    3. Generates all combinations of up to *nmax* modes
    4. For each combination and each order parameter direction (OPD),
       displaces the parent structure along the mode eigenvectors
    5. Relaxes each displaced structure to a local energy minimum
    6. Optionally computes phonons for each relaxed metastable structure

    The method is based on the theory of isotropy subgroups: each
    combination of imaginary modes and OPD direction maps to a specific
    subgroup of the parent space group, producing a candidate metastable
    structure.

    Args:
        atoms:
            ASE Atoms object for the high-symmetry parent structure
            (e.g. cubic perovskite). Assumed to be one formula unit.
        calc:
            ASE calculator, calculator name string (``'mace-r2scan'``,
            ``'chgnet'``, ``'mace'``), or None (defaults to CHGNet).
        model_path:
            Optional path to a model file. Only used when *calc* is a
            string.
        nmax:
            Maximum number of imaginary modes to combine simultaneously.
            ``nmax=1`` considers single modes only; ``nmax=2`` also
            considers all pairs; ``nmax=3`` adds triplets. Higher values
            explore more structures but cost significantly more.  Default: 2.
        amplitude:
            Displacement amplitude in Angstrom for the initial modulation.
            Default: 0.5.
        max_cell_size:
            Maximum number of atoms in the commensurate supercell.
            Supercell matrices with larger cells are skipped.  Default: 64.
        phonon_kwargs:
            Extra keyword arguments passed to
            :func:`~atomchain.phonon.frozenphonon.calculate_phonon`
            for the parent phonon calculation.
        relax_kwargs:
            Extra keyword arguments passed to
            :func:`~atomchain.relax.relax_with_ml` for each relaxation.
            Common keys: ``fmax`` (force convergence, eV/Angstrom).
        output_dir:
            Directory for all output files (structures, phonon data,
            reports).  Default: ``"metastable_results"``.
        phonon_ndim:
            Supercell dimensions for the phonon calculation, as a 3x3
            matrix or length-3 list (diagonal).  Default: ``[[2,0,0],
            [0,2,0], [0,0,2]]``.
        symprec:
            Symmetry precision for spglib / phonopy operations.
            Default: 1e-5.
        include_nonmaximal:
            If True, also generate non-maximal OPD directions (e.g.
            ``(a,b,0)`` for a 3-fold degenerate mode), which can produce
            additional distinct local minima.  Default: True.
        deduplicate:
            If True, remove duplicate relaxed structures (same space
            group and similar energy).  Default: True.
        compute_phonons:
            If True, compute phonon band structures for each relaxed
            metastable structure. Results are cached in
            ``output_dir/metastable_phonon/``.  Default: False.

    Returns:
        dict with keys:

        - ``"results"``: list of result dicts, one per metastable
          structure found.  Each dict contains:

          =============== =================================================
          Key             Description
          =============== =================================================
          id              Integer identifier (1-based)
          energy          Total energy (eV)
          energy_per_fu   Energy per formula unit (eV)
          delta_e_per_fu  Energy relative to parent (eV/FU)
          spacegroup_name Relaxed space group name (e.g. ``"Pnma"``)
          spacegroup_number Relaxed space group number
          initial_spacegroup_name Space group before relaxation
          initial_spacegroup_number Space group number before relaxation
          combined_label  Mode + OPD label (e.g. ``"GM4-(a,0,0)"``)
          opd_label       Order parameter direction label
          opd_is_maximal  Whether the OPD is a maximal direction
          source_modes    List of source imaginary mode dicts
          supercell_matrix 3x3 supercell matrix used
          n_atoms         Number of atoms in the relaxed structure
          structure_file  Filename of the relaxed structure (VASP)
          phonon_yaml     Path to phonopy YAML (if compute_phonons)
          phonon_dir      Path to phonon cache dir (if compute_phonons)
          =============== =================================================

        - ``"imaginary_modes"``: list of imaginary mode dicts
        - ``"phonon_dir"``: path to parent phonon calculation directory

    Example::

        from ase.io import read
        from atomchain import explore_metastable_states

        atoms = read("BaTiO3.vasp")
        data = explore_metastable_states(
            atoms,
            calc="mace-r2scan",
            nmax=2,
            amplitude=0.5,
            relax_kwargs={"fmax": 0.05},
        )
        for r in data["results"]:
            print(f'{r["combined_label"]:30s} {r["spacegroup_name"]:10s} '
                  f'ΔE={r["delta_e_per_fu"]:.4f} eV/FU')
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(calc, str):
        calc = init_calc(model_type=calc, model_path=model_path)
    elif calc is None:
        calc = init_calc(model_type="chgnet")

    if phonon_kwargs is None:
        phonon_kwargs = {}

    if relax_kwargs is None:
        relax_kwargs = {}

    if phonon_ndim is None:
        phonon_ndim = np.diag([2, 2, 2])

    n_atoms_per_fu = len(atoms)

    phonon_save_dir = os.path.join(output_dir, "parent_phonon_save")

    print("[metastable] Step 1: Computing phonons...")
    calculate_phonon(
        atoms,
        calc=calc,
        ndim=phonon_ndim,
        phonon_save_dir=phonon_save_dir,
        parallel=False,
        **phonon_kwargs,
    )

    phonopy_yaml_path = os.path.join(phonon_save_dir, "phonopy_params.yaml")

    print("[metastable] Step 2: Computing parent energy...")
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    try:
        parent_energy_total = float(atoms_copy.get_potential_energy())
        parent_energy_per_fu = _energy_per_fu(
            parent_energy_total, len(atoms), n_atoms_per_fu
        )
        print(f"[metastable] Parent energy: {parent_energy_per_fu:.4f} eV/FU")
    except (TypeError, ValueError):
        parent_energy_per_fu = None
        print("[metastable] Parent energy: could not compute (mock calculator?)")

    print("[metastable] Step 3: Labeling modes...")
    all_labeled_modes = get_all_labeled_modes(phonopy_yaml_path, symprec=symprec)

    print("[metastable] Step 4: Filtering imaginary modes...")
    imaginary_modes = get_imaginary_modes(all_labeled_modes)

    if len(imaginary_modes) == 0:
        print("[metastable] No imaginary modes found. Returning empty results.")
        return {
            "results": [],
            "imaginary_modes": [],
            "phonon_dir": phonon_save_dir,
        }

    imaginary_modes = _deduplicate_modes(imaginary_modes)
    print(
        f"[metastable] Found {len(imaginary_modes)} unique imaginary mode(s) after deduplication."
    )

    label_to_qpoint = _build_label_to_qpoint(atoms, symprec=symprec)

    print("[metastable] Step 5: Generating mode combinations...")
    results = []
    result_id = 1

    for r in range(1, min(nmax, len(imaginary_modes)) + 1):
        for combo in itertools.combinations(imaginary_modes, r):
            combo_labels = "/".join(
                m.get("bcs_label") or m["kpoint_label"] for m in combo
            )
            print(f"[metastable] Processing combination of {r} mode(s): {combo_labels}")

            qpoints_in_combo = []
            valid = True
            for m in combo:
                qpoint = label_to_qpoint.get(m["kpoint_label"])
                if qpoint is None:
                    print(
                        f"[metastable] Warning: could not find qpoint for label "
                        f"'{m['kpoint_label']}'. Skipping."
                    )
                    valid = False
                    break
                qpoints_in_combo.append(np.array(qpoint))

            if not valid:
                continue

            unique_kpoints = _deduplicate_kpoints(qpoints_in_combo)
            supercell_matrix = find_commensurate_matrix(unique_kpoints, max_cell_size)
            if supercell_matrix is None:
                print("[metastable] No commensurate supercell found. Skipping.")
                continue

            print(
                f"[metastable] Commensurate supercell matrix: {supercell_matrix.tolist()}"
            )

            if r == 1:
                mode = combo[0]
                qpoint = unique_kpoints[0].tolist()
                mod_info_list = get_modulations_with_opd_info(
                    phonopy_yaml_path,
                    qpoint=qpoint,
                    band_index=mode["band_index"],
                    supercell_matrix=supercell_matrix,
                    amplitude=amplitude,
                    symprec=symprec,
                    include_nonmaximal=include_nonmaximal,
                )

                for info in mod_info_list:
                    mod_atoms = info["atoms"]
                    opd_label = info["opd_label"]
                    init_sg_number, init_sg_name = _get_spacegroup(mod_atoms)

                    print(
                        f"[metastable] Relaxing {mode.get('bcs_label', '?')}{opd_label} "
                        f"(maximal={info['is_maximal']})..."
                    )

                    relax_params = dict(
                        calc=calc, sym=True, relax_cell=True, fmax=0.001
                    )
                    relax_params.update(relax_kwargs)
                    relaxed = relax_with_ml(mod_atoms, **relax_params)

                    energy = relaxed.get_potential_energy()
                    energy_per_fu = _energy_per_fu(energy, len(relaxed), n_atoms_per_fu)
                    delta_e = (
                        energy_per_fu - parent_energy_per_fu
                        if parent_energy_per_fu is not None
                        else None
                    )
                    sg_number, sg_name = _get_spacegroup(relaxed)

                    struct_filename = f"metastable_{result_id:03d}.vasp"
                    struct_filepath = os.path.join(output_dir, struct_filename)
                    write(struct_filepath, relaxed)

                    m_label = mode["kpoint_label"]
                    m_qpoint = label_to_qpoint.get(m_label)
                    source_modes = [
                        {
                            "kpoint": list(m_qpoint) if m_qpoint is not None else None,
                            "kpoint_label": m_label,
                            "band_index": mode["band_index"],
                            "frequency": mode["frequency"],
                            "bcs_label": mode.get("bcs_label"),
                        }
                    ]

                    combined_label = _build_combined_label(
                        source_modes, opd_labels=[opd_label]
                    )

                    result = {
                        "id": result_id,
                        "energy": float(energy),
                        "energy_per_fu": float(energy_per_fu),
                        "delta_e_per_fu": float(delta_e),
                        "initial_spacegroup_number": init_sg_number,
                        "initial_spacegroup_name": init_sg_name,
                        "spacegroup_number": sg_number,
                        "spacegroup_name": sg_name,
                        "source_modes": source_modes,
                        "combined_label": combined_label,
                        "opd_label": opd_label,
                        "opd_is_maximal": info["is_maximal"],
                        "polarization_direction": info.get("polarization_direction"),
                        "supercell_matrix": supercell_matrix.tolist(),
                        "n_atoms": len(relaxed),
                        "structure_file": struct_filename,
                        "is_still_imaginary": None,
                    }

                    results.append(result)
                    result_id += 1

                    delta_str = f"{delta_e:.4f}" if delta_e is not None else "N/A"
                    maximal_str = "maximal" if info["is_maximal"] else "non-maximal"
                    print(
                        f"[metastable] Result {result_id - 1}: ΔE={delta_str} eV/FU, "
                        f"spacegroup={sg_name} ({sg_number}), n_atoms={len(relaxed)}, "
                        f"label={combined_label} [{maximal_str}]"
                    )

            else:
                mode_specs = []
                for m in combo:
                    qpoint = label_to_qpoint.get(m["kpoint_label"])
                    mode_specs.append((qpoint, m["band_index"]))

                mod_info_list, _ = get_multi_mode_modulations_with_info(
                    phonopy_yaml_path,
                    mode_specs=mode_specs,
                    supercell_matrix=supercell_matrix,
                    amplitude=amplitude,
                    symprec=symprec,
                    include_nonmaximal=include_nonmaximal,
                )

                if len(mod_info_list) == 0:
                    print("[metastable] No OPD structures generated. Skipping.")
                    continue

                print(
                    f"[metastable] Generated {len(mod_info_list)} modulated structure(s)."
                )

                for i, mod_info in enumerate(mod_info_list):
                    mod_atoms = mod_info["atoms"]
                    opd_labels = mod_info["opd_labels"]
                    all_maximal = mod_info["opd_is_maximal"]
                    init_sg_number, init_sg_name = _get_spacegroup(mod_atoms)

                    print(
                        f"[metastable] Relaxing modulated structure "
                        f"{i + 1}/{len(mod_info_list)}..."
                    )

                    relax_params = dict(
                        calc=calc, sym=True, relax_cell=True, fmax=0.001
                    )
                    relax_params.update(relax_kwargs)
                    relaxed = relax_with_ml(mod_atoms, **relax_params)

                    energy = relaxed.get_potential_energy()
                    energy_per_fu = _energy_per_fu(energy, len(relaxed), n_atoms_per_fu)
                    delta_e = (
                        energy_per_fu - parent_energy_per_fu
                        if parent_energy_per_fu is not None
                        else None
                    )
                    sg_number, sg_name = _get_spacegroup(relaxed)

                    struct_filename = f"metastable_{result_id:03d}.vasp"
                    struct_filepath = os.path.join(output_dir, struct_filename)
                    write(struct_filepath, relaxed)

                    source_modes = []
                    for mode in combo:
                        m_label = mode["kpoint_label"]
                        m_qpoint = label_to_qpoint.get(m_label)
                        source_modes.append(
                            {
                                "kpoint": list(m_qpoint)
                                if m_qpoint is not None
                                else None,
                                "kpoint_label": m_label,
                                "band_index": mode["band_index"],
                                "frequency": mode["frequency"],
                                "bcs_label": mode.get("bcs_label"),
                            }
                        )

                    combined_label = _build_combined_label(
                        source_modes, opd_labels=opd_labels
                    )

                    result = {
                        "id": result_id,
                        "energy": float(energy),
                        "energy_per_fu": float(energy_per_fu),
                        "delta_e_per_fu": float(delta_e),
                        "initial_spacegroup_number": init_sg_number,
                        "initial_spacegroup_name": init_sg_name,
                        "spacegroup_number": sg_number,
                        "spacegroup_name": sg_name,
                        "source_modes": source_modes,
                        "combined_label": combined_label,
                        "opd_label": "/".join(opd_labels) if opd_labels else None,
                        "opd_is_maximal": all_maximal,
                        "polarization_direction": None,
                        "supercell_matrix": supercell_matrix.tolist(),
                        "n_atoms": len(relaxed),
                        "structure_file": struct_filename,
                        "is_still_imaginary": None,
                    }

                    results.append(result)
                    result_id += 1

                    delta_str = f"{delta_e:.4f}" if delta_e is not None else "N/A"
                    maximal_str = "maximal" if all_maximal else "non-maximal"
                    print(
                        f"[metastable] Result {result_id - 1}: ΔE={delta_str} eV/FU, "
                        f"spacegroup={sg_name} ({sg_number}), n_atoms={len(relaxed)}, "
                        f"label={combined_label} [{maximal_str}]"
                    )

    if deduplicate:
        n_before = len(results)
        results = _deduplicate_results(results)
        n_after = len(results)
        if n_before != n_after:
            print(f"[metastable] Deduplicated {n_before} → {n_after} results.")

    for i, r in enumerate(results):
        r["id"] = i + 1

    if compute_phonons and results:
        print(
            f"[metastable] Computing phonons for {len(results)} metastable structure(s)..."
        )
        _compute_phonons_for_results(
            results, calc, output_dir, phonon_ndim, phonon_kwargs
        )

    print(f"[metastable] Done. Found {len(results)} metastable state(s).")
    return {
        "results": results,
        "imaginary_modes": imaginary_modes,
        "phonon_dir": phonon_save_dir,
    }


def _compute_phonons_for_results(results, calc, output_dir, phonon_ndim, phonon_kwargs):
    """Compute and cache phonons for each metastable structure."""
    for r in results:
        struct_file = r.get("structure_file")
        if not struct_file:
            continue
        struct_path = os.path.join(output_dir, struct_file)
        if not os.path.exists(struct_path):
            continue

        phonon_subdir = os.path.join(
            output_dir,
            "metastable_phonon",
            os.path.splitext(struct_file)[0],
        )
        r["phonon_dir"] = phonon_subdir

        phonon_yaml = os.path.join(phonon_subdir, "phonopy_params.yaml")
        if os.path.exists(phonon_yaml):
            print(f"[metastable] Phonon cache hit for {struct_file}, skipping.")
            r["phonon_yaml"] = phonon_yaml
            continue

        print(f"[metastable] Computing phonons for {struct_file}...")
        from ase.io import read as ase_read

        struct_atoms = ase_read(struct_path)

        try:
            calculate_phonon(
                struct_atoms,
                calc=calc,
                ndim=phonon_ndim,
                phonon_save_dir=phonon_subdir,
                parallel=False,
                **phonon_kwargs,
            )
            r["phonon_yaml"] = os.path.join(phonon_subdir, "phonopy_params.yaml")
        except Exception as e:
            print(
                f"[metastable] Warning: phonon calculation failed for {struct_file}: {e}"
            )
            r["phonon_yaml"] = None
