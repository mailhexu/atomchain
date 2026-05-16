"""Report generation for metastable states exploration.

Generates machine-readable (YAML) and human-readable (Markdown + PNG)
reports from the output of :func:`~atomchain.metastable.explore_metastable_states`.

Public API:

- :func:`generate_report` — builds report dict, writes YAML, Markdown,
  and PNG files
- :func:`print_summary` — prints a text summary table to stdout
"""

from __future__ import annotations

import os
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spglib
import yaml
from ase.io import write


def _assign_groups_for_report(results):
    """Ensure report entries have relaxed-structure grouping metadata."""
    groups = {}
    next_group_id = 1
    for result in results:
        group_id = result.get("relaxed_group_id")
        if group_id is None:
            key = (
                result.get("spacegroup_number"),
                round(float(result.get("energy_per_fu", 0.0)), 3)
                if result.get("energy_per_fu") is not None
                else result.get("id"),
                result.get("n_atoms"),
            )
            if key not in groups:
                groups[key] = next_group_id
                next_group_id += 1
            result["relaxed_group_id"] = groups[key]

    members = {}
    for result in results:
        members.setdefault(result.get("relaxed_group_id"), []).append(result.get("id"))

    for result in results:
        group_members = members.get(result.get("relaxed_group_id"), [result.get("id")])
        result.setdefault("relaxed_group_members", group_members)
        result.setdefault("relaxed_group_size", len(group_members))
        result.setdefault(
            "relaxed_group_representative", result.get("id") == group_members[0]
        )
    return results


def _write_structure_directory_readmes(output_dir, report):
    """Write README files for report-linked initial and relaxed structures."""
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
            "Files in this directory map to rows in `../report.yaml` and `../report.md`.",
            "",
            "| Result ID | File | Label | Relaxed Group | Source Modes | Supercell |",
            "|---|---|---|---|---|---|",
        ]
        for result in report.get("results", []):
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


def _serialize_kpoints(kpoints):
    entries = []
    seen_qpoints = set()
    for item in kpoints or []:
        if isinstance(item, dict):
            label = item.get("label")
            qpoint = item.get("qpoint")
        else:
            label = getattr(item, "label", None)
            qpoint = getattr(item, "qpoint", None)
        qpoint_values = (
            [float(x) for x in np.asarray(qpoint, dtype=float).flat]
            if qpoint is not None
            else None
        )
        qpoint_key = (
            tuple(round(x, 8) for x in qpoint_values) if qpoint_values else None
        )
        if qpoint_key is not None and qpoint_key in seen_qpoints:
            continue
        if qpoint_key is not None:
            seen_qpoints.add(qpoint_key)
        entries.append(
            {
                "label": str(label),
                "qpoint": qpoint_values,
            }
        )
    return entries


def _serialize_labeled_modes(all_labeled_modes):
    serialized = []
    for label, data in sorted((all_labeled_modes or {}).items()):
        modes = []
        for mode in data.get("modes", []):
            modes.append(
                {
                    "band_index": int(mode["band_index"])
                    if mode.get("band_index") is not None
                    else None,
                    "frequency_THz": round(float(mode["frequency"]), 4)
                    if mode.get("frequency") is not None
                    else None,
                    "bcs_label": mode.get("bcs_label"),
                    "mulliken_label": mode.get("mulliken_label"),
                }
            )
        frequencies = data.get("frequencies")
        serialized.append(
            {
                "label": str(label),
                "frequencies_THz": [
                    round(float(x), 4) for x in np.asarray(frequencies).flat
                ]
                if frequencies is not None
                else [mode["frequency_THz"] for mode in modes],
                "modes": modes,
            }
        )
    return serialized


def _mode_labels(mode):
    labels = [mode.get("bcs_label"), mode.get("mulliken_label")]
    return [label for label in labels if label]


def _mode_index_summary(modes):
    parts = []
    for mode in modes:
        kpoint_label = mode.get("kpoint_label") or "?"
        band_index = mode.get("band_index")
        parts.append(
            kpoint_label if band_index is None else f"{kpoint_label}:{band_index}"
        )
    return ", ".join(parts)


def _mode_label_summary(modes):
    parts = []
    for mode in modes:
        labels = _mode_labels(mode)
        parts.append("/".join(labels) if labels else mode.get("kpoint_label") or "?")
    return ", ".join(parts)


def _get_spacegroup_info(atoms, symprec=0.1):
    """Return (international_name, number) for an ASE Atoms object."""
    cell = (
        atoms.get_cell(),
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    spg_dict = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if spg_dict is None:
        return "Unknown", 0
    return spg_dict.international, spg_dict.number


def _safe_filename_stem(value):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return stem or "structure"


def _plot_phonon_band_structure(phonon_dir, output_dir, formula=None):
    """Plot the parent phonon band structure and copy PNG to output_dir.

    Returns the PNG filename or None on failure.
    """
    try:
        from atomchain.phonon.plotphonopy import plot_phonon
    except ImportError:
        return None

    phonon_yaml = os.path.join(phonon_dir, "phonopy_params.yaml")
    if not os.path.exists(phonon_yaml):
        return None

    prefix = _safe_filename_stem(formula) if formula else "parent"
    figname = f"{prefix}_phonon_band_structure.png"
    try:
        plot_phonon(path=phonon_dir, figname=figname, show=False, units="THz")
        src = os.path.join(phonon_dir, figname)
        dst = os.path.join(output_dir, figname)
        if os.path.exists(src) and src != dst:
            shutil.copy2(src, dst)
        return figname
    except Exception as e:
        print(f"[report] Warning: could not plot phonon bands: {e}")
        return None


def _plot_metastable_phonon(result_id, phonon_dir, output_dir, formula=None):
    """Plot phonon bands for a single metastable structure.

    Returns the PNG filename (e.g. ``metastable_001_phonon.png``) or None.
    """
    phonon_yaml = os.path.join(phonon_dir, "phonopy_params.yaml")
    if not os.path.exists(phonon_yaml):
        return None

    prefix = _safe_filename_stem(formula) if formula else "metastable"
    figname = f"{prefix}_metastable_{result_id:03d}_phonon.png"
    try:
        from atomchain.phonon.plotphonopy import plot_phonon

        plot_phonon(path=phonon_dir, figname=figname, show=False, units="THz")
        src = os.path.join(phonon_dir, figname)
        dst = os.path.join(output_dir, figname)
        if os.path.exists(src) and src != dst:
            shutil.copy2(src, dst)
        return figname
    except Exception as e:
        print(f"[report] Warning: could not plot phonon for result {result_id}: {e}")
        return None


def _plot_energy_bar_chart(report, output_dir):
    """Generate a horizontal bar chart of ΔE/FU for all metastable structures."""
    results = [r for r in report["results"] if r.get("delta_e_per_fu") is not None]
    if not results:
        return None

    sorted_res = sorted(results, key=lambda x: x.get("delta_e_per_fu") or 0)
    labels = [r["combined_label"] for r in sorted_res]
    energies = [r.get("delta_e_per_fu") or 0 for r in sorted_res]
    colors = ["#e74c3c" if e > 0 else "#2980b9" for e in energies]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), 6))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, energies, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("\u0394E per formula unit (eV)", fontsize=11)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()
    calc = report.get("method", {}).get("calculator", "unknown")
    ax.set_title(
        f"{report['parent_structure']['formula']} metastable states ({calc})",
        fontsize=12,
    )
    plt.tight_layout()

    formula = report.get("parent_structure", {}).get("formula", "structure")
    figname = f"{_safe_filename_stem(formula)}_energy_bar_chart.png"
    figpath = os.path.join(output_dir, figname)
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figname


def _build_report_dict(exploration_data, parent_atoms, calc_name, params, output_dir):
    """Build the report data dict from exploration results.

    Args:
        exploration_data: Return value of :func:`explore_metastable_states`
            — either a dict (``{"results": ..., "imaginary_modes": ...,
            "phonon_dir": ...}``) or a plain list of result dicts
            (backward compatible).
        parent_atoms: ASE Atoms of the parent structure.
        calc_name: Calculator name string (e.g. ``"mace-r2scan"``).
        params: Dict of calculation parameters (nmax, amplitude, etc.).
        output_dir: Output directory path.

    Returns:
        Report dict with keys ``method``, ``parent_structure``,
        ``imaginary_modes``, ``results``, and optional plot filenames.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(exploration_data, dict):
        results = exploration_data.get("results", [])
        imaginary_modes = exploration_data.get("imaginary_modes", [])
        phonon_dir = exploration_data.get("phonon_dir")
        ase_kpoints = exploration_data.get("ase_kpoints", [])
        bcs_kpoints = exploration_data.get("bcs_kpoints", [])
        bcs_labeled_modes = exploration_data.get("bcs_labeled_modes", {})
    else:
        results = exploration_data
        imaginary_modes = []
        phonon_dir = None
        ase_kpoints = []
        bcs_kpoints = []
        bcs_labeled_modes = {}

    results = _assign_groups_for_report(list(results))

    parent_sg_name, parent_sg_number = _get_spacegroup_info(parent_atoms)

    report = {
        "method": {
            "calculator": calc_name,
            "nmax": params.get("nmax"),
            "amplitude": params.get("amplitude"),
            "max_cell_size": params.get("max_cell_size"),
            "fmax": params.get("fmax"),
            "phonon_supercell": params.get("phonon_ndim"),
        },
        "parent_structure": {
            "formula": parent_atoms.get_chemical_formula(mode="metal", empirical=True),
            "n_atoms": len(parent_atoms),
            "spacegroup": parent_sg_name,
            "spacegroup_number": parent_sg_number,
        },
        "imaginary_modes": [],
        "ase_kpoints": _serialize_kpoints(ase_kpoints),
        "bcs_kpoints": _serialize_kpoints(bcs_kpoints),
        "bcs_labeled_modes": _serialize_labeled_modes(bcs_labeled_modes),
        "results": [],
    }

    for m in imaginary_modes:
        kpoint = m.get("kpoint")
        if kpoint is not None:
            kpoint = [float(x) for x in np.asarray(kpoint).flat]
        report["imaginary_modes"].append(
            {
                "kpoint_label": m.get("kpoint_label", ""),
                "kpoint": kpoint,
                "band_index": m.get("band_index"),
                "frequency_THz": round(float(m["frequency"]), 4)
                if m.get("frequency") is not None
                else None,
                "bcs_label": m.get("bcs_label"),
                "mulliken_label": m.get("mulliken_label"),
                "degeneracy": m.get("degeneracy"),
            }
        )

    for r in results:
        if (
            "energy_per_fu" not in report["parent_structure"]
            and r.get("energy_per_fu") is not None
            and r.get("delta_e_per_fu") is not None
        ):
            parent_e_fu = r["energy_per_fu"] - r["delta_e_per_fu"]
            report["parent_structure"]["parent_energy_per_fu"] = float(parent_e_fu)

        atoms_obj = r.get("atoms")
        sg_name = r.get("spacegroup_name", "Unknown")
        sg_number = r.get("spacegroup_number", 0)
        init_sg_name = r.get("initial_spacegroup_name")
        init_sg_number = r.get("initial_spacegroup_number")

        if atoms_obj is not None and sg_number == 0:
            sg_name, sg_number = _get_spacegroup_info(atoms_obj)

        source_modes = r.get("source_modes", [])
        modes_for_report = []
        for m in source_modes:
            kpoint = m.get("kpoint")
            if kpoint is not None:
                kpoint = [float(x) for x in np.asarray(kpoint).flat]
            mode_entry = {
                "kpoint": kpoint,
                "kpoint_label": m.get("kpoint_label", ""),
                "band_index": int(m["band_index"])
                if m.get("band_index") is not None
                else None,
                "frequency_THz": round(float(m["frequency"]), 4)
                if m.get("frequency") is not None
                else None,
            }
            if "bcs_label" in m:
                mode_entry["bcs_label"] = m["bcs_label"]
            if "mulliken_label" in m:
                mode_entry["mulliken_label"] = m["mulliken_label"]
            modes_for_report.append(mode_entry)

        sm = r.get("supercell_matrix")
        if sm is not None:
            sm = [[float(x) for x in row] for row in np.asarray(sm).reshape(3, 3)]

        status = r.get("status", "success")
        legacy_structure_file = r.get("structure_file")
        if legacy_structure_file and os.path.dirname(legacy_structure_file):
            relaxed_structure_file = legacy_structure_file
        elif (
            status != "success"
            and r.get("atoms") is None
            and r.get("relaxed_structure_file") is None
        ):
            relaxed_structure_file = None
        else:
            relaxed_structure_file = r.get(
                "relaxed_structure_file",
                os.path.join("relaxed_structures", f"relaxed_{r['id']:03d}.vasp"),
            )

        entry = {
            "id": int(r["id"]),
            "status": status,
            "error_stage": r.get("error_stage"),
            "error_message": r.get("error_message"),
            "attempt": r.get("attempt"),
            "amplitude": float(r["amplitude"])
            if r.get("amplitude") is not None
            else None,
            "pre_relax_max_force": float(r["pre_relax_max_force"])
            if r.get("pre_relax_max_force") is not None
            else None,
            "force_screen_reductions": r.get("force_screen_reductions"),
            "delta_e_per_fu": float(r["delta_e_per_fu"])
            if r.get("delta_e_per_fu") is not None
            else None,
            "initial_spacegroup": f"{init_sg_name} (#{init_sg_number})"
            if init_sg_name
            else None,
            "relaxed_spacegroup": f"{sg_name} (#{sg_number})" if sg_name else None,
            "n_atoms": int(
                r.get("n_atoms", len(atoms_obj) if atoms_obj is not None else 0)
            ),
            "combined_label": r.get("combined_label", ""),
            "mode_indices": _mode_index_summary(modes_for_report),
            "mode_labels": _mode_label_summary(modes_for_report),
            "opd_label": r.get("opd_label"),
            "opd_is_maximal": r.get("opd_is_maximal"),
            "source_modes": modes_for_report,
            "supercell_matrix": sm,
            "initial_structure_file": r.get(
                "initial_structure_file",
                os.path.join("initial_structures", f"initial_{r['id']:03d}.vasp"),
            ),
            "relaxed_structure_file": relaxed_structure_file,
            "structure_file": relaxed_structure_file,
            "relaxed_group_id": r.get("relaxed_group_id"),
            "relaxed_group_size": r.get("relaxed_group_size"),
            "relaxed_group_members": r.get("relaxed_group_members"),
            "relaxed_group_representative": r.get("relaxed_group_representative"),
            "phonon_band_structure": None,
        }

        phonon_yaml = r.get("phonon_yaml")
        if phonon_yaml and os.path.exists(phonon_yaml):
            phonon_subdir = os.path.dirname(phonon_yaml)
            fig = _plot_metastable_phonon(
                entry["id"],
                phonon_subdir,
                output_dir,
                report["parent_structure"]["formula"],
            )
            entry["phonon_band_structure"] = fig

        report["results"].append(entry)

        initial_atoms = r.get("initial_atoms")
        if initial_atoms is not None:
            initial_path = os.path.join(output_dir, entry["initial_structure_file"])
            os.makedirs(os.path.dirname(initial_path), exist_ok=True)
            write(initial_path, initial_atoms, format="vasp")

        fname = entry["relaxed_structure_file"]
        if atoms_obj is not None:
            relaxed_path = os.path.join(output_dir, fname)
            relaxed_dir = os.path.dirname(relaxed_path)
            if relaxed_dir:
                os.makedirs(relaxed_dir, exist_ok=True)
            write(relaxed_path, atoms_obj, format="vasp")

    if phonon_dir:
        band_fig = _plot_phonon_band_structure(
            phonon_dir, output_dir, report["parent_structure"]["formula"]
        )
        if band_fig:
            report["phonon_band_structure"] = band_fig

    bar_fig = _plot_energy_bar_chart(report, output_dir)
    if bar_fig:
        report["energy_bar_chart"] = bar_fig

    report["relaxed_structure_groups"] = _build_group_summary(report["results"])
    _write_structure_directory_readmes(output_dir, report)

    return report


def _build_group_summary(results):
    groups = {}
    for result in results:
        group_id = result.get("relaxed_group_id")
        if group_id is None:
            continue
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "representative_id": result.get("id"),
                "member_ids": [],
                "spacegroup": result.get("relaxed_spacegroup"),
                "best_delta_e_per_fu": result.get("delta_e_per_fu"),
            },
        )
        group["member_ids"].append(result.get("id"))
        delta = result.get("delta_e_per_fu")
        if delta is not None and (
            group["best_delta_e_per_fu"] is None or delta < group["best_delta_e_per_fu"]
        ):
            group["best_delta_e_per_fu"] = delta
            group["representative_id"] = result.get("id")
    return sorted(groups.values(), key=lambda group: group["group_id"])


def _lowest_energy_results(results, limit=10):
    successful = [
        result for result in results if result.get("delta_e_per_fu") is not None
    ]
    failed = [result for result in results if result.get("delta_e_per_fu") is None]
    return sorted(successful, key=lambda x: x.get("delta_e_per_fu"))[:limit], failed


def _generate_markdown(report):
    """Generate a markdown report string from the report dict.

    Sections included:

    - **Method**: calculator, parameters
    - **Parent Structure**: formula, space group, energy
    - **Phonon Band Structure**: embedded image of parent phonons
    - **Imaginary Phonon Modes**: table of unstable modes
    - **Energy Landscape**: embedded bar chart
    - **Metastable Structures**: energy table, space group summary,
      and per-structure phonon band structure images (if available)

    Args:
        report: Report dict from :func:`generate_report` or
            :func:`_build_report_dict`.

    Returns:
        Markdown string.
    """
    parent = report["parent_structure"]
    method = report.get("method", {})
    imaginary = report.get("imaginary_modes", [])
    results = report["results"]
    calc = method.get("calculator", "unknown")

    lines = []

    lines.append(f"# Metastable States Report: {parent['formula']}")
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(f"- **Calculator**: {calc}")
    lines.append(f"- **Max mode combinations (nmax)**: {method.get('nmax', '?')}")
    lines.append(f"- **Displacement amplitude**: {method.get('amplitude', '?')} \u00c5")
    lines.append(f"- **Relaxation fmax**: {method.get('fmax', '?')} eV/\u00c5")
    lines.append(f"- **Max supercell size**: {method.get('max_cell_size', '?')} atoms")
    ph_sc = method.get("phonon_supercell")
    if ph_sc:
        lines.append(f"- **Phonon supercell**: {ph_sc}")
    lines.append("")

    lines.append("## Parent Structure")
    lines.append("")
    lines.append(f"- **Formula**: {parent['formula']}")
    lines.append(
        f"- **Space group**: {parent['spacegroup']} (#{parent['spacegroup_number']})"
    )
    lines.append(f"- **Atoms in primitive cell**: {parent['n_atoms']}")
    if "parent_energy_per_fu" in parent:
        lines.append(f"- **Energy**: {parent['parent_energy_per_fu']:.4f} eV/FU")
    lines.append("")

    ase_kpoints = report.get("ase_kpoints", [])
    bcs_kpoints = report.get("bcs_kpoints", [])
    bcs_labeled_modes = report.get("bcs_labeled_modes", [])
    if ase_kpoints or bcs_kpoints or bcs_labeled_modes:
        lines.append("## High-Symmetry Q-Point Coverage")
        lines.append("")
        lines.append(
            "The phonon band plot uses ASE band-path special points, while symmetry labels come from the BCS/symphon q-point list. These lists can differ."
        )
        lines.append("")
        if ase_kpoints:
            lines.append("### ASE Band-Path Special Points")
            lines.append("")
            lines.append("| Label | q-point |")
            lines.append("|---|---|")
            for item in ase_kpoints:
                qpoint = item.get("qpoint")
                qstr = ", ".join(f"{x:.6g}" for x in qpoint) if qpoint else ""
                lines.append(f"| {item.get('label')} | [{qstr}] |")
            lines.append("")
        if bcs_kpoints:
            lines.append("### BCS/Symphon Special Points")
            lines.append("")
            lines.append("| Label | q-point in input reciprocal basis |")
            lines.append("|---|---|")
            for item in bcs_kpoints:
                qpoint = item.get("qpoint")
                qstr = ", ".join(f"{x:.6g}" for x in qpoint) if qpoint else ""
                lines.append(f"| {item.get('label')} | [{qstr}] |")
            lines.append("")
        if bcs_labeled_modes:
            lines.append("### Symphon Mode Labels At BCS Points")
            lines.append("")
            lines.append(
                "| Q-point | Band | Frequency (THz) | BCS Label | Mulliken Label |"
            )
            lines.append("|---|---|---|---|---|")
            for point in bcs_labeled_modes:
                label = point.get("label")
                for mode in point.get("modes", []):
                    freq = mode.get("frequency_THz")
                    freq_str = f"{freq:.4f}" if freq is not None else ""
                    lines.append(
                        f"| {label} | {mode.get('band_index')} | {freq_str} | {mode.get('bcs_label') or ''} | {mode.get('mulliken_label') or ''} |"
                    )
            lines.append("")

    if report.get("phonon_band_structure"):
        lines.append("## Phonon Band Structure")
        lines.append("")
        lines.append(f"![Phonon band structure]({report['phonon_band_structure']})")
        lines.append("")

    if imaginary:
        lines.append("## Imaginary Phonon Modes")
        lines.append("")
        lines.append(
            f"The parent structure has **{len(imaginary)}** unstable phonon mode(s):"
        )
        lines.append("")
        lines.append(
            "| # | Q-point | q-point | Band index | Frequency (THz) | BCS Label | Mulliken Label | Degeneracy |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, m in enumerate(imaginary, 1):
            kpt = m.get("kpoint_label", "?")
            qpoint = m.get("kpoint")
            qstr = ", ".join(f"{x:.6g}" for x in qpoint) if qpoint else ""
            band = m.get("band_index", "?")
            freq = m.get("frequency_THz", 0)
            deg = m.get("degeneracy", "?")
            bcs_label = m.get("bcs_label") or ""
            mulliken_label = m.get("mulliken_label") or ""
            lines.append(
                f"| {i} | {kpt} | [{qstr}] | {band} | {freq:.3f} | {bcs_label} | {mulliken_label} | {deg} |"
            )
        lines.append("")

    if report.get("energy_bar_chart"):
        lines.append("## Energy Landscape")
        lines.append("")
        lines.append(f"![Energy bar chart]({report['energy_bar_chart']})")
        lines.append("")

    if results:
        sorted_results = sorted(results, key=lambda x: x.get("delta_e_per_fu") or 0)
        lowest_results, failed_results = _lowest_energy_results(results, limit=10)

        lines.append("## Metastable Structures")
        lines.append("")
        n_success = sum(1 for r in results if r.get("status", "success") == "success")
        n_failed = len(results) - n_success
        lines.append(
            f"**{len(results)}** candidate structures generated "
            f"({n_success} relaxed, {n_failed} failed)"
        )
        successful_results = [
            r for r in sorted_results if r.get("status", "success") == "success"
        ]
        if successful_results:
            best = successful_results[0]
            best_de = best.get("delta_e_per_fu")
            best_sg = best.get("relaxed_spacegroup", "Unknown")
            best_label = best.get("combined_label", "")
            de_str = f"{best_de:.4f}" if best_de is not None else "N/A"
            lines.append(
                f" - Ground state: **{best_sg}** at \u0394E = {de_str} eV/FU "
                f"({best_label})"
            )
        lines.append("")

        lines.append("### Lowest-Energy Structures")
        lines.append("")
        if len(lowest_results) < len(
            [r for r in results if r.get("delta_e_per_fu") is not None]
        ):
            lines.append(
                f"Showing the {len(lowest_results)} lowest-energy relaxed candidates. Full results are available in `report.yaml`."
            )
            lines.append("")
        if failed_results:
            lines.append(
                f"{len(failed_results)} failed candidate(s) are omitted from this lowest-energy table; see `report.yaml` for their error details."
            )
            lines.append("")
        if any(r.get("relaxed_group_size", 1) > 1 for r in lowest_results):
            lines.append(
                "Rows with the same relaxed group are symmetry/energy-equivalent relaxed structures; all generated candidates remain listed."
            )
            lines.append("")
        has_init_sg = any(r.get("initial_spacegroup") for r in lowest_results)
        if has_init_sg:
            lines.append(
                "| # | Status | Attempt | Amplitude | Group | \u0394E/FU (eV) | Initial SG | Relaxed SG | Mode Index | Mode Labels | Label | Initial File | Relaxed File | Atoms | Type | Error |"
            )
            lines.append(
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
            )
        else:
            lines.append(
                "| # | Status | Attempt | Amplitude | Group | \u0394E/FU (eV) | Relaxed SG | Mode Index | Mode Labels | Label | Initial File | Relaxed File | Atoms | Type | Error |"
            )
            lines.append(
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
            )

        for r in lowest_results:
            de = (
                f"{r['delta_e_per_fu']:.4f}"
                if r.get("delta_e_per_fu") is not None
                else "N/A"
            )
            init_sg = r.get("initial_spacegroup") or "\u2014"
            rel_sg = r.get("relaxed_spacegroup") or "\u2014"
            label = r.get("combined_label", "")
            mode_indices = r.get("mode_indices", "")
            mode_labels = r.get("mode_labels", "")
            n = r.get("n_atoms", "?")
            group = str(r.get("relaxed_group_id", ""))
            if r.get("relaxed_group_size", 1) > 1:
                group += f" ({r.get('relaxed_group_size')} eq.)"
            initial_file = r.get("initial_structure_file") or ""
            relaxed_file = (
                r.get("relaxed_structure_file") or r.get("structure_file") or ""
            )
            status = r.get("status", "success")
            attempt = r.get("attempt") or ""
            attempt_amplitude = r.get("amplitude")
            amplitude_str = (
                f"{attempt_amplitude:.6g}" if attempt_amplitude is not None else ""
            )
            error = r.get("error_stage") or ""
            if r.get("error_message"):
                error = (
                    f"{error}: {r.get('error_message')}"
                    if error
                    else r.get("error_message")
                )
            tp = (
                "maximal"
                if r.get("opd_is_maximal")
                else ("non-max" if r.get("opd_is_maximal") is False else "\u2014")
            )
            if has_init_sg:
                lines.append(
                    f"| {r['id']} | {status} | {attempt} | {amplitude_str} | {group} | {de} | {init_sg} | {rel_sg} | {mode_indices} | {mode_labels} | {label} | {initial_file} | {relaxed_file} | {n} | {tp} | {error} |"
                )
            else:
                lines.append(
                    f"| {r['id']} | {status} | {attempt} | {amplitude_str} | {group} | {de} | {rel_sg} | {mode_indices} | {mode_labels} | {label} | {initial_file} | {relaxed_file} | {n} | {tp} | {error} |"
                )
        lines.append("")

        if failed_results:
            lines.append("### Failed Candidates")
            lines.append("")
            lines.append("Failed candidates are omitted from the lowest-energy table.")
            lines.append("")
            lines.append("| # | Label | Mode Index | Mode Labels | Stage | Error |")
            lines.append("|---|---|---|---|---|---|")
            for r in failed_results:
                error = r.get("error_message") or ""
                lines.append(
                    f"| {r['id']} | {r.get('combined_label', '')} | {r.get('mode_indices', '')} | {r.get('mode_labels', '')} | {r.get('error_stage') or ''} | {error} |"
                )
            lines.append("")

        group_summaries = report.get("relaxed_structure_groups", [])
        if group_summaries:
            lines.append("### Relaxed Structure Groups")
            lines.append("")
            lines.append(
                "Equivalent relaxed structures are grouped by relaxed space group, rounded energy, and atom count."
            )
            lines.append("")
            lines.append(
                "| Group | Representative | Members | Space Group | Best \u0394E/FU (eV) |"
            )
            lines.append("|---|---|---|---|---|")
            for group_info in group_summaries:
                best = group_info.get("best_delta_e_per_fu")
                best_str = f"{best:.4f}" if best is not None else "N/A"
                members = ", ".join(str(i) for i in group_info.get("member_ids", []))
                lines.append(
                    f"| {group_info.get('group_id')} | {group_info.get('representative_id')} | {members} | {group_info.get('spacegroup')} | {best_str} |"
                )
            lines.append("")

        unique_sgs = sorted(
            set(
                str(r.get("relaxed_spacegroup")).split(" (")[0]
                for r in results
                if r.get("relaxed_spacegroup")
            )
        )
        if unique_sgs:
            lines.append("### Space Groups Found")
            lines.append("")
            sg_best = {}
            for r in results:
                sg = r.get("relaxed_spacegroup", "")
                sg_base = str(sg).split(" (")[0] if sg else "?"
                if sg_base not in sg_best or (
                    r.get("delta_e_per_fu") is not None
                    and (
                        sg_best[sg_base] is None
                        or r["delta_e_per_fu"] < sg_best[sg_base]
                    )
                ):
                    sg_best[sg_base] = r.get("delta_e_per_fu")
            lines.append("| Space Group | Best \u0394E/FU (eV) | Count |")
            lines.append("|------------|----------------|-------|")
            for sg in unique_sgs:
                count = sum(
                    1
                    for r in results
                    if str(r.get("relaxed_spacegroup") or "").startswith(sg)
                )
                best_e = sg_best.get(sg)
                de_str = f"{best_e:.4f}" if best_e is not None else "N/A"
                lines.append(f"| {sg} | {de_str} | {count} |")
            lines.append("")

        phonon_results = [r for r in sorted_results if r.get("phonon_band_structure")]
        if phonon_results:
            lines.append("### Phonon Band Structures")
            lines.append("")
            for r in phonon_results:
                rel_sg = r.get("relaxed_spacegroup", "?")
                label = r.get("combined_label", "")
                de = (
                    f"{r['delta_e_per_fu']:.4f}"
                    if r.get("delta_e_per_fu") is not None
                    else "N/A"
                )
                fig = r["phonon_band_structure"]
                lines.append(
                    f"**#{r['id']}** {rel_sg} — {label} (\u0394E = {de} eV/FU)"
                )
                lines.append("")
                lines.append(f"![Phonon #{r['id']}]({fig})")
                lines.append("")

    return "\n".join(lines)


def generate_report(exploration_data, parent_atoms, calc_name, params, output_dir):
    """Generate a complete report from metastable exploration results.

    Writes three files to *output_dir*:

    - ``report.yaml`` — machine-readable data (all results + metadata)
    - ``report.md`` — human-readable markdown report with embedded
      image references
    - ``phonon_band_structure.png`` — parent phonon band structure
    - ``energy_bar_chart.png`` — bar chart of ΔE/FU values
    - ``metastable_NNN_phonon.png`` — phonon bands per structure
      (only when phonon data is available)

    Args:
        exploration_data: Return value of
            :func:`~atomchain.metastable.explore_metastable_states`.
        parent_atoms: ASE Atoms of the parent (high-symmetry) structure.
        calc_name: Calculator name (e.g. ``"mace-r2scan"``).
        params: Dict of calculation parameters. Recognized keys:
            ``nmax``, ``amplitude``, ``max_cell_size``, ``fmax``,
            ``phonon_ndim``.
        output_dir: Output directory path.

    Returns:
        Report dict (same as written to report.yaml).
    """
    report = _build_report_dict(
        exploration_data, parent_atoms, calc_name, params, output_dir
    )

    report_path = os.path.join(output_dir, "report.yaml")
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    md = _generate_markdown(report)
    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] Markdown report written to {md_path}")

    return report


def print_summary(report, parent_atoms):
    """Print a text summary of the report to stdout.

    Shows parent info, imaginary modes, and a table of all metastable
    structures sorted by ID with their initial/relaxed space groups and
    energies.
    """
    parent_info = report["parent_structure"]
    results = report["results"]
    method = report.get("method", {})

    lines = []
    lines.append("=" * 90)
    lines.append("Metastable States Exploration Report")
    lines.append("=" * 90)
    lines.append(
        f"Parent: {parent_info['formula']} ({parent_info['spacegroup']}, "
        f"{parent_info['n_atoms']} atoms)"
    )
    if "parent_energy_per_fu" in parent_info:
        lines.append(f"Parent energy: {parent_info['parent_energy_per_fu']:.4f} eV/FU")
    calc_name = method.get("calculator", "unknown")
    lines.append(f"Calculator: {calc_name}")
    lines.append(
        f"Parameters: nmax={method.get('nmax')}, amplitude={method.get('amplitude')}, "
        f"fmax={method.get('fmax')}"
    )

    imaginary = report.get("imaginary_modes", [])
    if imaginary:
        lines.append(f"\nImaginary modes ({len(imaginary)}):")
        for m in imaginary:
            freq = m.get("frequency_THz", 0)
            label = m.get("bcs_label", m.get("kpoint_label", "?"))
            deg = m.get("degeneracy", "?")
            lines.append(
                f"  {label} (band {m.get('band_index', '?')}): "
                f"{freq:.3f} THz, degeneracy={deg}"
            )

    lines.append(f"\nStructures found: {len(results)}")
    lines.append("-" * 90)
    delta_header = "\u0394E/FU"
    lines.append(
        f"{'#':<4} {delta_header:>8} {'Initial SG':<20} {'Relaxed SG':<20} "
        f"{'Label':<30} {'Type':<10}"
    )
    lines.append("-" * 90)

    for r in results:
        delta_str = (
            f"{r['delta_e_per_fu']:.4f}"
            if r.get("delta_e_per_fu") is not None
            else "N/A"
        )
        init_sg = r.get("initial_spacegroup") or ""
        rel_sg = r.get("relaxed_spacegroup") or ""
        label = r.get("combined_label", "")
        is_max = r.get("opd_is_maximal")
        type_str = ""
        if is_max is True:
            type_str = "maximal"
        elif is_max is False:
            type_str = "non-max"

        lines.append(
            f"{r['id']:<4} {delta_str:>8} {init_sg:<20} {rel_sg:<20} "
            f"{label:<30} {type_str:<10}"
        )

    lines.append("=" * 90)
    print("\n".join(lines))
