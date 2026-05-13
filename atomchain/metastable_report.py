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
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spglib
import yaml
from ase.io import write


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


def _plot_phonon_band_structure(phonon_dir, output_dir):
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

    figname = "phonon_band_structure.png"
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


def _plot_metastable_phonon(result_id, phonon_dir, output_dir):
    """Plot phonon bands for a single metastable structure.

    Returns the PNG filename (e.g. ``metastable_001_phonon.png``) or None.
    """
    phonon_yaml = os.path.join(phonon_dir, "phonopy_params.yaml")
    if not os.path.exists(phonon_yaml):
        return None

    figname = f"metastable_{result_id:03d}_phonon.png"
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
    results = report["results"]
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

    figname = "energy_bar_chart.png"
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
    else:
        results = exploration_data
        imaginary_modes = []
        phonon_dir = None

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
        "results": [],
    }

    for m in imaginary_modes:
        report["imaginary_modes"].append(
            {
                "kpoint_label": m.get("kpoint_label", ""),
                "band_index": m.get("band_index"),
                "frequency_THz": round(float(m["frequency"]), 4)
                if m.get("frequency") is not None
                else None,
                "bcs_label": m.get("bcs_label"),
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
            modes_for_report.append(mode_entry)

        sm = r.get("supercell_matrix")
        if sm is not None:
            sm = [[float(x) for x in row] for row in np.asarray(sm).reshape(3, 3)]

        entry = {
            "id": int(r["id"]),
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
            "opd_label": r.get("opd_label"),
            "opd_is_maximal": r.get("opd_is_maximal"),
            "source_modes": modes_for_report,
            "supercell_matrix": sm,
            "structure_file": r.get("structure_file", f"metastable_{r['id']:03d}.vasp"),
            "phonon_band_structure": None,
        }

        phonon_yaml = r.get("phonon_yaml")
        if phonon_yaml and os.path.exists(phonon_yaml):
            phonon_subdir = os.path.dirname(phonon_yaml)
            fig = _plot_metastable_phonon(entry["id"], phonon_subdir, output_dir)
            entry["phonon_band_structure"] = fig

        report["results"].append(entry)

        fname = entry["structure_file"]
        if atoms_obj is not None:
            write(os.path.join(output_dir, fname), atoms_obj, format="vasp")

    if phonon_dir:
        band_fig = _plot_phonon_band_structure(phonon_dir, output_dir)
        if band_fig:
            report["phonon_band_structure"] = band_fig

    bar_fig = _plot_energy_bar_chart(report, output_dir)
    if bar_fig:
        report["energy_bar_chart"] = bar_fig

    return report


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
        lines.append("| # | Label | k-point | Band | Frequency (THz) | Degeneracy |")
        lines.append("|---|-------|---------|------|-----------------|------------|")
        for i, m in enumerate(imaginary, 1):
            label = m.get("bcs_label", m.get("kpoint_label", "?"))
            kpt = m.get("kpoint_label", "?")
            band = m.get("band_index", "?")
            freq = m.get("frequency_THz", 0)
            deg = m.get("degeneracy", "?")
            lines.append(f"| {i} | {label} | {kpt} | {band} | {freq:.3f} | {deg} |")
        lines.append("")

    if report.get("energy_bar_chart"):
        lines.append("## Energy Landscape")
        lines.append("")
        lines.append(f"![Energy bar chart]({report['energy_bar_chart']})")
        lines.append("")

    if results:
        sorted_results = sorted(results, key=lambda x: x.get("delta_e_per_fu") or 0)

        lines.append("## Metastable Structures")
        lines.append("")
        lines.append(f"**{len(results)}** distinct metastable structures found")
        if results:
            best = sorted_results[0]
            best_de = best.get("delta_e_per_fu")
            best_sg = best.get("relaxed_spacegroup", "Unknown")
            best_label = best.get("combined_label", "")
            de_str = f"{best_de:.4f}" if best_de is not None else "N/A"
            lines.append(
                f" - Ground state: **{best_sg}** at \u0394E = {de_str} eV/FU "
                f"({best_label})"
            )
        lines.append("")

        lines.append("### Energy Table (sorted by energy)")
        lines.append("")
        has_init_sg = any(r.get("initial_spacegroup") for r in sorted_results)
        if has_init_sg:
            lines.append(
                "| # | \u0394E/FU (eV) | Initial SG | Relaxed SG | Label | Atoms | Type |"
            )
            lines.append(
                "|---|-------------|-----------|-----------|-------|-------|------|"
            )
            for r in sorted_results:
                de = (
                    f"{r['delta_e_per_fu']:.4f}"
                    if r.get("delta_e_per_fu") is not None
                    else "N/A"
                )
                init_sg = r.get("initial_spacegroup") or "\u2014"
                rel_sg = r.get("relaxed_spacegroup") or "\u2014"
                label = r.get("combined_label", "")
                n = r.get("n_atoms", "?")
                tp = (
                    "maximal"
                    if r.get("opd_is_maximal")
                    else ("non-max" if r.get("opd_is_maximal") is False else "\u2014")
                )
                lines.append(
                    f"| {r['id']} | {de} | {init_sg} | {rel_sg} | {label} | {n} | {tp} |"
                )
        else:
            lines.append("| # | \u0394E/FU (eV) | Relaxed SG | Label | Atoms | Type |")
            lines.append("|---|-------------|-----------|-------|-------|------|")
            for r in sorted_results:
                de = (
                    f"{r['delta_e_per_fu']:.4f}"
                    if r.get("delta_e_per_fu") is not None
                    else "N/A"
                )
                rel_sg = r.get("relaxed_spacegroup") or "\u2014"
                label = r.get("combined_label", "")
                n = r.get("n_atoms", "?")
                tp = (
                    "maximal"
                    if r.get("opd_is_maximal")
                    else ("non-max" if r.get("opd_is_maximal") is False else "\u2014")
                )
                lines.append(f"| {r['id']} | {de} | {rel_sg} | {label} | {n} | {tp} |")
        lines.append("")

        unique_sgs = sorted(
            set(
                r.get("relaxed_spacegroup", "").split(" (")[0]
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
                sg_base = sg.split(" (")[0] if sg else "?"
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
                    1 for r in results if r.get("relaxed_spacegroup", "").startswith(sg)
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
