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
import pickle

import numpy as np
import spglib
from ase.build import make_supercell
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

_CHECKPOINT_VERSION = "2026-05-15-reference-supercell-v2"


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


def _attach_qpoints_to_modes(modes, label_to_qpoint):
    """Add q-point coordinates to mode dictionaries when labels are known."""
    enriched = []
    for mode in modes:
        mode_with_qpoint = dict(mode)
        qpoint = label_to_qpoint.get(mode.get("kpoint_label"))
        mode_with_qpoint["kpoint"] = list(qpoint) if qpoint is not None else None
        enriched.append(mode_with_qpoint)
    return enriched


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


def _get_spacegroup_or_raise(atoms, stage, symprec=0.1):
    """Return spacegroup info, raising if spglib cannot identify one."""
    sg_number, sg_name = _get_spacegroup(atoms, symprec=symprec)
    if sg_number is None or sg_name is None:
        raise RuntimeError(f"could not determine {stage} spacegroup")
    return sg_number, sg_name


def _checkpoint_filename(output_dir, checkpoint_path=None):
    """Return the checkpoint file path for a metastable run."""
    return checkpoint_path or os.path.join(output_dir, "checkpoint.pkl")


def _load_checkpoint(path, metadata=None):
    """Load checkpointed candidate results, returning an empty list on absence."""
    if not path or not os.path.exists(path):
        return []
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        checkpoint_metadata = data.get("metadata")
        if metadata is not None and checkpoint_metadata != metadata:
            print(
                "[metastable] Checkpoint metadata does not match current run; starting fresh."
            )
            return []
        return list(data.get("results", []))
    if metadata is not None:
        print("[metastable] Legacy checkpoint has no metadata; starting fresh.")
        return []
    return list(data)


def _write_checkpoint(path, results, metadata=None):
    """Write completed candidate results to the checkpoint file."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint_results = []
    for result in results:
        checkpoint_result = dict(result)
        checkpoint_result.pop("atoms", None)
        checkpoint_result.pop("initial_atoms", None)
        checkpoint_results.append(checkpoint_result)
    with open(path, "wb") as handle:
        pickle.dump({"metadata": metadata or {}, "results": checkpoint_results}, handle)


def _checkpoint_metadata(
    atoms,
    nmax,
    amplitude,
    max_cell_size,
    phonon_ndim,
    symprec,
    include_nonmaximal,
):
    """Build metadata used to decide whether a checkpoint can be reused."""
    return {
        "formula": atoms.get_chemical_formula(mode="metal", empirical=True),
        "checkpoint_version": _CHECKPOINT_VERSION,
        "n_atoms": len(atoms),
        "nmax": int(nmax),
        "amplitude": float(amplitude),
        "max_cell_size": int(max_cell_size),
        "phonon_ndim": np.asarray(phonon_ndim, dtype=int).tolist(),
        "symprec": float(symprec),
        "include_nonmaximal": bool(include_nonmaximal),
    }


def _source_modes_for_combo(combo, label_to_qpoint):
    """Build serializable source-mode metadata for a mode combination."""
    source_modes = []
    for mode in combo:
        m_label = mode["kpoint_label"]
        m_qpoint = label_to_qpoint.get(m_label)
        source_modes.append(
            {
                "kpoint": list(m_qpoint) if m_qpoint is not None else None,
                "kpoint_label": m_label,
                "band_index": mode["band_index"],
                "frequency": mode["frequency"],
                "bcs_label": mode.get("bcs_label"),
            }
        )
    return source_modes


def _reference_supercell(atoms, supercell_matrix):
    """Build an undisplaced supercell matching the candidate supercell."""
    try:
        return make_supercell(atoms, np.asarray(supercell_matrix, dtype=int))
    except Exception:
        return None


def _max_supercell_determinant(atoms, max_cell_size):
    """Convert a maximum atom count to a maximum supercell determinant."""
    return int(max_cell_size) // len(atoms)


def _scale_modulation(mod_atoms, reference_atoms, target_amplitude, original_amplitude):
    """Scale a distorted candidate back toward its undisplaced supercell."""
    if (
        reference_atoms is None
        or original_amplitude == 0
        or len(reference_atoms) != len(mod_atoms)
    ):
        return mod_atoms.copy()
    scaled = reference_atoms.copy()
    scaled.set_cell(mod_atoms.get_cell(), scale_atoms=False)
    factor = target_amplitude / original_amplitude
    disp = mod_atoms.get_scaled_positions() - reference_atoms.get_scaled_positions()
    disp -= np.round(disp)
    scaled.set_scaled_positions(reference_atoms.get_scaled_positions() + disp * factor)
    scaled.set_pbc(mod_atoms.get_pbc())
    return scaled


def _get_max_force(atoms, calc):
    """Run a single-point force calculation and return the max force norm."""
    force_atoms = atoms.copy()
    force_atoms.calc = calc
    forces = np.asarray(force_atoms.get_forces(), dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError("calculator returned forces with unexpected shape")
    return float(np.max(np.linalg.norm(forces, axis=1)))


def _screen_amplitude_by_force(
    mod_atoms,
    reference_atoms,
    amplitude,
    calc,
    force_threshold=10.0,
    max_reductions=10,
):
    """Halve amplitude until the pre-relaxation single-point force is acceptable."""
    attempt_amplitude = amplitude
    reductions = 0
    while True:
        attempt_atoms = _scale_modulation(
            mod_atoms, reference_atoms, attempt_amplitude, amplitude
        )
        max_force = _get_max_force(attempt_atoms, calc)
        if max_force <= force_threshold:
            return attempt_atoms, attempt_amplitude, max_force, reductions
        if reductions >= max_reductions:
            raise RuntimeError(
                f"max force {max_force:.6g} eV/Angstrom remains above "
                f"{force_threshold:.6g} eV/Angstrom after {reductions} amplitude reductions"
            )
        next_amplitude = attempt_amplitude / 2.0
        print(
            f"[metastable] Pre-relaxation max force {max_force:.6g} eV/Angstrom "
            f"at amplitude {attempt_amplitude:g}; retrying amplitude {next_amplitude:g}."
        )
        attempt_amplitude = next_amplitude
        reductions += 1


def _failure_result(
    result_id,
    source_modes,
    combined_label,
    opd_label,
    opd_is_maximal,
    polarization_direction,
    supercell_matrix,
    initial_relpath,
    initial_atoms,
    error_stage,
    error_message,
    attempt,
    attempt_amplitude,
    relaxed=None,
    relaxed_relpath=None,
    init_sg_number=None,
    init_sg_name=None,
    pre_relax_max_force=None,
    force_screen_reductions=0,
):
    """Build a failed candidate result that can be reported like successes."""
    result = {
        "id": result_id,
        "status": "failed",
        "error_stage": error_stage,
        "error_message": str(error_message),
        "attempt": attempt,
        "amplitude": float(attempt_amplitude),
        "pre_relax_max_force": float(pre_relax_max_force)
        if pre_relax_max_force is not None
        else None,
        "force_screen_reductions": int(force_screen_reductions),
        "energy": None,
        "energy_per_fu": None,
        "delta_e_per_fu": None,
        "initial_spacegroup_number": init_sg_number,
        "initial_spacegroup_name": init_sg_name,
        "spacegroup_number": None,
        "spacegroup_name": None,
        "source_modes": source_modes,
        "combined_label": combined_label,
        "opd_label": opd_label,
        "opd_is_maximal": opd_is_maximal,
        "polarization_direction": polarization_direction,
        "supercell_matrix": np.asarray(supercell_matrix).tolist(),
        "n_atoms": len(relaxed) if relaxed is not None else len(initial_atoms),
        "initial_structure_file": initial_relpath,
        "relaxed_structure_file": relaxed_relpath,
        "structure_file": relaxed_relpath,
        "initial_atoms": initial_atoms,
        "atoms": relaxed,
        "is_still_imaginary": None,
    }
    return result


def _relax_candidate_with_retries(
    result_id,
    output_dir,
    mod_atoms,
    reference_atoms,
    source_modes,
    combined_label,
    opd_label,
    opd_is_maximal,
    polarization_direction,
    supercell_matrix,
    calc,
    relax_kwargs,
    amplitude,
    parent_energy_per_fu,
    n_atoms_per_fu,
):
    """Relax one candidate, retrying twice with smaller amplitudes on failure."""
    initial_relpath = os.path.join(
        "initial_structures", f"initial_{result_id:03d}.vasp"
    )
    relaxed_relpath = os.path.join(
        "relaxed_structures", f"relaxed_{result_id:03d}.vasp"
    )
    trajectory_relpath = os.path.join(
        "relaxation_trajectories", f"relax_{result_id:03d}.traj"
    )
    try:
        screened_atoms, screened_amplitude, screened_force, reductions = (
            _screen_amplitude_by_force(mod_atoms, reference_atoms, amplitude, calc)
        )
    except Exception as exc:
        write(os.path.join(output_dir, initial_relpath), mod_atoms)
        return _failure_result(
            result_id,
            source_modes,
            combined_label,
            opd_label,
            opd_is_maximal,
            polarization_direction,
            supercell_matrix,
            initial_relpath,
            mod_atoms,
            "pre_relaxation_force",
            exc,
            0,
            amplitude,
        )

    attempt_amplitudes = [screened_amplitude / (2**i) for i in range(3)]
    last_error = None
    last_initial_atoms = screened_atoms
    last_force = screened_force
    last_reductions = reductions

    for attempt, attempt_amplitude in enumerate(attempt_amplitudes, start=1):
        attempt_atoms = _scale_modulation(
            screened_atoms, reference_atoms, attempt_amplitude, screened_amplitude
        )
        last_initial_atoms = attempt_atoms
        try:
            last_force = _get_max_force(attempt_atoms, calc)
        except Exception:
            last_force = screened_force
        try:
            init_sg_number, init_sg_name = _get_spacegroup_or_raise(
                attempt_atoms, "initial"
            )
        except Exception as exc:
            write(os.path.join(output_dir, initial_relpath), attempt_atoms)
            return _failure_result(
                result_id,
                source_modes,
                combined_label,
                opd_label,
                opd_is_maximal,
                polarization_direction,
                supercell_matrix,
                initial_relpath,
                attempt_atoms,
                "initial_symmetry",
                exc,
                attempt,
                attempt_amplitude,
                pre_relax_max_force=last_force,
                force_screen_reductions=last_reductions,
            )

        relax_params = dict(
            calc=calc,
            sym=True,
            relax_cell=True,
            fmax=0.001,
        )
        relax_params.update(relax_kwargs)
        relax_params["traj_file"] = os.path.join(output_dir, trajectory_relpath)
        try:
            relaxed = relax_with_ml(attempt_atoms, **relax_params)
        except Exception as exc:
            last_error = exc
            print(
                f"[metastable] Relaxation attempt {attempt}/3 failed for "
                f"result {result_id} at amplitude {attempt_amplitude:g}: {exc}"
            )
            continue

        try:
            energy = relaxed.get_potential_energy()
            energy_per_fu = _energy_per_fu(energy, len(relaxed), n_atoms_per_fu)
            delta_e = (
                energy_per_fu - parent_energy_per_fu
                if parent_energy_per_fu is not None
                else None
            )
            sg_number, sg_name = _get_spacegroup_or_raise(relaxed, "relaxed")
        except Exception as exc:
            write(os.path.join(output_dir, initial_relpath), attempt_atoms)
            write(os.path.join(output_dir, relaxed_relpath), relaxed)
            return _failure_result(
                result_id,
                source_modes,
                combined_label,
                opd_label,
                opd_is_maximal,
                polarization_direction,
                supercell_matrix,
                initial_relpath,
                attempt_atoms,
                "final_symmetry",
                exc,
                attempt,
                attempt_amplitude,
                relaxed=relaxed,
                relaxed_relpath=relaxed_relpath,
                init_sg_number=init_sg_number,
                init_sg_name=init_sg_name,
                pre_relax_max_force=last_force,
                force_screen_reductions=last_reductions,
            )

        write(os.path.join(output_dir, initial_relpath), attempt_atoms)
        write(os.path.join(output_dir, relaxed_relpath), relaxed)
        return {
            "id": result_id,
            "status": "success",
            "error_stage": None,
            "error_message": None,
            "attempt": attempt,
            "amplitude": float(attempt_amplitude),
            "pre_relax_max_force": float(last_force)
            if last_force is not None
            else None,
            "force_screen_reductions": int(last_reductions),
            "energy": float(energy),
            "energy_per_fu": float(energy_per_fu),
            "delta_e_per_fu": float(delta_e) if delta_e is not None else None,
            "initial_spacegroup_number": init_sg_number,
            "initial_spacegroup_name": init_sg_name,
            "spacegroup_number": sg_number,
            "spacegroup_name": sg_name,
            "source_modes": source_modes,
            "combined_label": combined_label,
            "opd_label": opd_label,
            "opd_is_maximal": opd_is_maximal,
            "polarization_direction": polarization_direction,
            "supercell_matrix": np.asarray(supercell_matrix).tolist(),
            "n_atoms": len(relaxed),
            "initial_structure_file": initial_relpath,
            "relaxed_structure_file": relaxed_relpath,
            "trajectory_file": trajectory_relpath,
            "structure_file": relaxed_relpath,
            "initial_atoms": attempt_atoms,
            "atoms": relaxed,
            "is_still_imaginary": None,
        }

    write(os.path.join(output_dir, initial_relpath), last_initial_atoms)
    return _failure_result(
        result_id,
        source_modes,
        combined_label,
        opd_label,
        opd_is_maximal,
        polarization_direction,
        supercell_matrix,
        initial_relpath,
        last_initial_atoms,
        "relaxation",
        last_error,
        3,
        attempt_amplitudes[-1],
        pre_relax_max_force=last_force,
        force_screen_reductions=last_reductions,
    )


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


def _assign_relaxed_groups(results, energy_tol=1e-3):
    """Annotate equivalent relaxed structures without removing candidates."""
    group_by_key = {}
    next_group_id = 1
    for result in results:
        sg = result.get("spacegroup_number")
        energy = result.get("energy_per_fu")
        n_atoms = result.get("n_atoms")
        if sg is None or energy is None:
            key = ("ungrouped", result.get("id"))
        else:
            bucket_e = round(energy / energy_tol) * energy_tol
            key = (sg, bucket_e, n_atoms)
        if key not in group_by_key:
            group_by_key[key] = next_group_id
            next_group_id += 1
        result["relaxed_group_id"] = group_by_key[key]

    group_members = {}
    for result in results:
        group_members.setdefault(result["relaxed_group_id"], []).append(result["id"])

    for result in results:
        members = group_members[result["relaxed_group_id"]]
        result["relaxed_group_members"] = members
        result["relaxed_group_size"] = len(members)
        result["relaxed_group_representative"] = result["id"] == members[0]
    return results


def _write_structure_directory_readmes(output_dir, results):
    """Write README files mapping generated structures to report rows."""
    for dirname, title, file_key in [
        (
            "initial_structures",
            "Initial Distorted Structures",
            "initial_structure_file",
        ),
        ("relaxed_structures", "Relaxed Structures", "relaxed_structure_file"),
    ]:
        dirpath = os.path.join(output_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        lines = [
            f"# {title}",
            "",
            "Files in this directory are linked to rows in `../report.yaml` and `../report.md`.",
            "",
            "| Result ID | File | Label | Relaxed Group | Source Modes | Supercell |",
            "|---|---|---|---|---|---|",
        ]
        for result in results:
            filename = result.get(file_key)
            if not filename:
                continue
            source_modes = ", ".join(
                f"{m.get('bcs_label') or m.get('kpoint_label')}:{m.get('band_index')}"
                for m in result.get("source_modes", [])
            )
            group = result.get("relaxed_group_id", "")
            if result.get("relaxed_group_size", 1) > 1:
                group = f"{group} ({result.get('relaxed_group_size')} equivalent)"
            lines.append(
                "| {id} | {file} | {label} | {group} | {modes} | {supercell} |".format(
                    id=result.get("id"),
                    file=os.path.basename(filename),
                    label=result.get("combined_label", ""),
                    group=group,
                    modes=source_modes,
                    supercell=result.get("supercell_matrix"),
                )
            )
        lines.append("")
        with open(os.path.join(dirpath, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


def _get_ase_bandpath_kpoints(phonopy_yaml_path):
    """Return ASE band-path special points in the phonopy primitive basis."""
    try:
        from ase.cell import Cell
        from phonopy import load

        phonon = load(phonopy_yaml=phonopy_yaml_path)
        bandpath = Cell(phonon.primitive.cell).bandpath(npoints=2)
    except Exception as exc:
        print(f"[metastable] Warning: could not get ASE bandpath k-points: {exc}")
        return []
    return [
        {"label": label, "qpoint": [float(x) for x in np.asarray(qpoint).flat]}
        for label, qpoint in sorted(bandpath.special_points.items())
    ]


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
    checkpoint_path=None,
    restart=True,
    relax_parent=True,
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
            If True, assign equivalent relaxed-structure groups using
            space group and similar energy. All generated candidates remain
            in the returned results. Default: True.
        compute_phonons:
            If True, compute phonon band structures for each relaxed
            metastable structure. Results are cached in
            ``output_dir/metastable_phonon/``.  Default: False.
        checkpoint_path:
            Optional checkpoint file path. Defaults to
            ``output_dir/checkpoint.pkl``.
        restart:
            If True, load the checkpoint by default and skip completed
            candidate IDs. Default: True.
        relax_parent:
            If True, relax the parent structure with symmetry and cell relaxation
            before computing parent phonons. Default: True.

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
        calc = init_calc(model_type="mace")

    if phonon_kwargs is None:
        phonon_kwargs = {}

    if relax_kwargs is None:
        relax_kwargs = {}

    if phonon_ndim is None:
        phonon_ndim = np.diag([2, 2, 2])

    if relax_parent:
        print("[metastable] Step 1: Relaxing parent structure...")
        parent_relax_params = dict(calc=calc, sym=True, relax_cell=True, fmax=0.001)
        parent_relax_params.update(relax_kwargs)
        atoms = relax_with_ml(atoms, **parent_relax_params)
        write(os.path.join(output_dir, "parent_relaxed.vasp"), atoms)

    n_atoms_per_fu = len(atoms)

    phonon_save_dir = os.path.join(output_dir, "parent_phonon_save")

    print("[metastable] Step 2: Computing phonons...")
    calculate_phonon(
        atoms,
        calc=calc,
        ndim=phonon_ndim,
        phonon_save_dir=phonon_save_dir,
        parallel=False,
        **phonon_kwargs,
    )

    phonopy_yaml_path = os.path.join(phonon_save_dir, "phonopy_params.yaml")

    print("[metastable] Step 3: Computing parent energy...")
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

    print("[metastable] Step 4: Labeling modes...")
    all_labeled_modes = get_all_labeled_modes(phonopy_yaml_path, symprec=symprec)
    ase_kpoints = _get_ase_bandpath_kpoints(phonopy_yaml_path)
    label_to_qpoint = _build_label_to_qpoint(atoms, symprec=symprec)
    bcs_kpoints = [
        {"label": label, "qpoint": [float(x) for x in np.asarray(qpoint).flat]}
        for label, qpoint in sorted(label_to_qpoint.items())
    ]

    print("[metastable] Step 5: Filtering imaginary modes...")
    imaginary_modes = get_imaginary_modes(all_labeled_modes)

    if len(imaginary_modes) == 0:
        print("[metastable] No imaginary modes found. Returning empty results.")
        return {
            "results": [],
            "imaginary_modes": [],
            "phonon_dir": phonon_save_dir,
            "ase_kpoints": ase_kpoints,
            "bcs_kpoints": bcs_kpoints,
            "bcs_labeled_modes": all_labeled_modes,
            "parent_atoms": atoms,
        }

    imaginary_modes = _attach_qpoints_to_modes(
        _deduplicate_modes(imaginary_modes), label_to_qpoint
    )
    print(
        f"[metastable] Found {len(imaginary_modes)} unique imaginary mode(s) after deduplication."
    )

    print("[metastable] Step 6: Generating mode combinations...")
    results = []
    result_id = 1
    initial_dir = os.path.join(output_dir, "initial_structures")
    relaxed_dir = os.path.join(output_dir, "relaxed_structures")
    trajectory_dir = os.path.join(output_dir, "relaxation_trajectories")
    os.makedirs(initial_dir, exist_ok=True)
    os.makedirs(relaxed_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)
    checkpoint_file = _checkpoint_filename(output_dir, checkpoint_path)
    checkpoint_metadata = _checkpoint_metadata(
        atoms,
        nmax,
        amplitude,
        max_cell_size,
        phonon_ndim,
        symprec,
        include_nonmaximal,
    )
    if restart:
        results = _load_checkpoint(checkpoint_file, metadata=checkpoint_metadata)
        if results:
            print(
                f"[metastable] Loaded {len(results)} checkpointed candidate(s) from {checkpoint_file}."
            )
    completed_ids = {result.get("id") for result in results}

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
            max_det = _max_supercell_determinant(atoms, max_cell_size)
            if max_det < 1:
                print(
                    f"[metastable] max_cell_size={max_cell_size} is smaller than "
                    f"the parent cell ({len(atoms)} atoms). Skipping."
                )
                continue
            supercell_matrix = find_commensurate_matrix(unique_kpoints, max_det)
            if supercell_matrix is None:
                print("[metastable] No commensurate supercell found. Skipping.")
                continue

            print(
                f"[metastable] Commensurate supercell matrix: {supercell_matrix.tolist()}"
            )
            reference_atoms = _reference_supercell(atoms, supercell_matrix)
            if reference_atoms is not None and len(reference_atoms) > max_cell_size:
                print(
                    f"[metastable] Commensurate supercell has {len(reference_atoms)} atoms, "
                    f"above max_cell_size={max_cell_size}. Skipping."
                )
                continue

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
                    if result_id in completed_ids:
                        print(
                            f"[metastable] Skipping checkpointed candidate {result_id}."
                        )
                        result_id += 1
                        continue

                    mod_atoms = info["atoms"]
                    mode_reference_atoms = info.get("reference_atoms", reference_atoms)
                    opd_label = info["opd_label"]

                    print(
                        f"[metastable] Relaxing {mode.get('bcs_label', '?')}{opd_label} "
                        f"(maximal={info['is_maximal']})..."
                    )

                    source_modes = _source_modes_for_combo(combo, label_to_qpoint)
                    combined_label = _build_combined_label(
                        source_modes, opd_labels=[opd_label]
                    )
                    result = _relax_candidate_with_retries(
                        result_id,
                        output_dir,
                        mod_atoms,
                        mode_reference_atoms,
                        source_modes,
                        combined_label,
                        opd_label,
                        info["is_maximal"],
                        info.get("polarization_direction"),
                        supercell_matrix,
                        calc,
                        relax_kwargs,
                        amplitude,
                        parent_energy_per_fu,
                        n_atoms_per_fu,
                    )

                    results.append(result)
                    _write_checkpoint(
                        checkpoint_file, results, metadata=checkpoint_metadata
                    )
                    result_id += 1

                    delta_e = result.get("delta_e_per_fu")
                    delta_str = f"{delta_e:.4f}" if delta_e is not None else "N/A"
                    maximal_str = "maximal" if info["is_maximal"] else "non-maximal"
                    print(
                        f"[metastable] Result {result_id - 1}: ΔE={delta_str} eV/FU, "
                        f"status={result.get('status')}, "
                        f"spacegroup={result.get('spacegroup_name')} ({result.get('spacegroup_number')}), "
                        f"n_atoms={result.get('n_atoms')}, "
                        f"label={combined_label} [{maximal_str}]"
                    )

            else:
                mode_specs = []
                for m in combo:
                    qpoint = label_to_qpoint.get(m["kpoint_label"])
                    mode_specs.append((qpoint, m["band_index"]))

                mod_info_list, multi_reference_atoms = (
                    get_multi_mode_modulations_with_info(
                        phonopy_yaml_path,
                        mode_specs=mode_specs,
                        supercell_matrix=supercell_matrix,
                        amplitude=amplitude,
                        symprec=symprec,
                        include_nonmaximal=include_nonmaximal,
                    )
                )

                if len(mod_info_list) == 0:
                    print("[metastable] No OPD structures generated. Skipping.")
                    continue

                print(
                    f"[metastable] Generated {len(mod_info_list)} modulated structure(s)."
                )

                for i, mod_info in enumerate(mod_info_list):
                    if result_id in completed_ids:
                        print(
                            f"[metastable] Skipping checkpointed candidate {result_id}."
                        )
                        result_id += 1
                        continue

                    mod_atoms = mod_info["atoms"]
                    mode_reference_atoms = multi_reference_atoms or reference_atoms
                    opd_labels = mod_info["opd_labels"]
                    all_maximal = mod_info["opd_is_maximal"]

                    print(
                        f"[metastable] Relaxing modulated structure "
                        f"{i + 1}/{len(mod_info_list)}..."
                    )

                    source_modes = _source_modes_for_combo(combo, label_to_qpoint)
                    combined_label = _build_combined_label(
                        source_modes, opd_labels=opd_labels
                    )
                    result = _relax_candidate_with_retries(
                        result_id,
                        output_dir,
                        mod_atoms,
                        mode_reference_atoms,
                        source_modes,
                        combined_label,
                        "/".join(opd_labels) if opd_labels else None,
                        all_maximal,
                        None,
                        supercell_matrix,
                        calc,
                        relax_kwargs,
                        amplitude,
                        parent_energy_per_fu,
                        n_atoms_per_fu,
                    )

                    results.append(result)
                    _write_checkpoint(
                        checkpoint_file, results, metadata=checkpoint_metadata
                    )
                    result_id += 1

                    delta_e = result.get("delta_e_per_fu")
                    delta_str = f"{delta_e:.4f}" if delta_e is not None else "N/A"
                    maximal_str = "maximal" if all_maximal else "non-maximal"
                    print(
                        f"[metastable] Result {result_id - 1}: ΔE={delta_str} eV/FU, "
                        f"status={result.get('status')}, "
                        f"spacegroup={result.get('spacegroup_name')} ({result.get('spacegroup_number')}), "
                        f"n_atoms={result.get('n_atoms')}, "
                        f"label={combined_label} [{maximal_str}]"
                    )

    if deduplicate:
        n_before = len(results)
        _assign_relaxed_groups(results)
        n_groups = len({result["relaxed_group_id"] for result in results})
        if n_before != n_groups:
            print(
                f"[metastable] Grouped {n_before} results into {n_groups} equivalent relaxed-structure group(s)."
            )
    else:
        _assign_relaxed_groups(results)

    _write_structure_directory_readmes(output_dir, results)

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
        "ase_kpoints": ase_kpoints,
        "bcs_kpoints": bcs_kpoints,
        "bcs_labeled_modes": all_labeled_modes,
        "checkpoint_file": checkpoint_file,
        "parent_atoms": atoms,
    }


def _compute_phonons_for_results(results, calc, output_dir, phonon_ndim, phonon_kwargs):
    """Compute and cache phonons for each metastable structure."""
    for r in results:
        if r.get("status", "success") != "success":
            continue
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
