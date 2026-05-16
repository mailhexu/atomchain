import itertools

import numpy as np
from ase import Atoms
from phonopy import load as phonopy_load
from phonopy.structure.cells import Supercell
from spgrep_modulation.isotropy import IsotropyEnumerator
from spgrep_modulation.modulation import Modulation


def phonopy_atoms_to_ase(phonopy_atoms) -> Atoms:
    return Atoms(
        symbols=list(phonopy_atoms.symbols),
        scaled_positions=phonopy_atoms.scaled_positions,
        cell=phonopy_atoms.cell,
        pbc=True,
    )


def _compute_polarization_direction(eigenvectors, coefficients):
    pol = np.zeros(3)
    for i, c in enumerate(coefficients):
        pol += c * np.sum(eigenvectors[i], axis=0).real
    return pol


def _canonicalize_direction(direction, tol=0.1, ratio_tol=0.15):
    d = np.abs(direction)
    d_max = np.max(d)
    if d_max < 1e-12:
        return "(0,0,0)"
    d = d / d_max
    nonzero_mask = d >= tol
    nonzero_vals = d[nonzero_mask]
    if len(nonzero_vals) == 0:
        return "(0,0,0)"
    sorted_vals = np.sort(np.unique(np.round(nonzero_vals, 1)))[::-1]
    groups = []
    for val in sorted_vals:
        matched = False
        for g in groups:
            if abs(val - g) < ratio_tol:
                matched = True
                break
        if not matched:
            groups.append(val)
    groups.sort(reverse=True)
    val_to_letter = {}
    for i, g in enumerate(groups):
        val_to_letter[g] = chr(ord("a") + i)

    def get_letter(v):
        if v < tol:
            return "0"
        for g, letter in val_to_letter.items():
            if abs(v - g) < ratio_tol:
                return letter
        return chr(ord("a") + len(val_to_letter))

    parts = [get_letter(v) for v in d]
    return "(" + ",".join(parts) + ")"


def _canonicalize_direction_general(direction, tol=0.1):
    d = np.abs(np.array(direction, dtype=float))
    d_max = np.max(d)
    if d_max < 1e-12:
        return tuple([0] * len(direction))
    d = d / d_max
    result = []
    for v in d:
        if v < tol:
            result.append(0)
        else:
            result.append(round(v, 2))
    return tuple(result)


def _find_eigenspace_index(md, target_freq, degeneracy_tolerance):
    for idx, (eigenvalue, _, _) in enumerate(md.eigenspaces):
        freq = md.eigvals_to_frequencies(eigenvalue)
        if abs(freq - target_freq) < degeneracy_tolerance:
            return idx
    return None


def _build_modulation(md, freq_idx, coefficients, amplitude):
    amplitudes = np.abs(np.array(coefficients, dtype=float))
    arguments = np.angle(np.array(coefficients, dtype=float))
    modulation = md.get_modulated_supercell_and_modulation(
        freq_idx, amplitudes, arguments, return_cell=False
    )
    modulation = np.real(modulation)
    max_val = np.max(np.abs(modulation))
    if max_val < 1e-12:
        return None, None
    scaled = amplitude / max_val * modulation
    cell = md.apply_modulation_to_supercell(scaled)
    return cell, scaled


def _is_maximal_opd_label(label):
    if label is None:
        return False
    inner = label.strip("()")
    parts = inner.split(",")
    nonzero = [p for p in parts if p != "0"]
    if len(nonzero) == 0:
        return False
    return all(p == nonzero[0] for p in nonzero)


def _generate_all_directions(dim, pol_matrix, include_nonmaximal=True):
    if dim == 1 or pol_matrix is None:
        return []

    targets = []

    if dim == 3:
        targets.extend(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        if include_nonmaximal:
            targets.extend(
                [
                    [1, 0.5, 0],
                    [1, 0, 0.5],
                    [0, 1, 0.5],
                    [1, 0.5, 0.5],
                    [0.5, 1, 0.5],
                    [0.5, 0.5, 1],
                    [1, 0.5, 0.3],
                ]
            )
    elif dim == 2:
        targets.extend(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )
        if include_nonmaximal:
            targets.extend(
                [
                    [1, 0.5],
                    [0.5, 1],
                    [1, 0.3],
                ]
            )
    else:
        for axis in range(dim):
            v = np.zeros(dim)
            v[axis] = 1.0
            targets.append(v.tolist())
        v = np.ones(dim)
        targets.append(v.tolist())

    is_gamma = np.linalg.matrix_rank(pol_matrix.T) >= 3
    if is_gamma:
        eigvec_dirs = []
        for target in targets:
            try:
                c = np.linalg.lstsq(
                    pol_matrix.T, np.array(target, dtype=float), rcond=None
                )[0]
                if not np.all(np.abs(c) < 1e6):
                    continue
                eigvec_dirs.append(c)
            except np.linalg.LinAlgError:
                continue
        return eigvec_dirs
    else:
        return [np.array(t, dtype=float) for t in targets]


def get_modulations_with_opd_info(
    phonopy_yaml_or_object,
    qpoint,
    band_index,
    supercell_matrix,
    amplitude=0.5,
    symprec=1e-5,
    degeneracy_tolerance=1e-3,
    include_nonmaximal=True,
) -> list[dict]:
    if isinstance(phonopy_yaml_or_object, str):
        phonon = phonopy_load(phonopy_yaml_or_object)
    else:
        phonon = phonopy_yaml_or_object

    qpoint = np.array(qpoint)
    supercell_matrix = np.array(supercell_matrix)

    phonon.run_qpoints([qpoint], with_eigenvectors=True)
    qdict = phonon.get_qpoints_dict()
    freqs = qdict["frequencies"][0]
    target_freq = freqs[band_index]

    md = Modulation.with_supercell_and_symmetry_search(
        dynamical_matrix=phonon.dynamical_matrix,
        supercell_matrix=supercell_matrix.tolist(),
        qpoint=qpoint.tolist(),
        factor=phonon.unit_conversion_factor,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
    )
    base_atoms = phonopy_atoms_to_ase(Supercell(phonon.primitive, supercell_matrix))

    freq_idx = _find_eigenspace_index(md, target_freq, degeneracy_tolerance)
    if freq_idx is None:
        return []

    eigenvalue, eigenvectors, irrep = md.eigenspaces[freq_idx]
    dim = irrep.shape[1]
    is_gamma = np.allclose(qpoint, 0.0, atol=1e-6)

    pol_matrix = None
    if is_gamma and dim >= 2:
        pol_matrix = np.zeros((dim, 3))
        for i in range(dim):
            pol_matrix[i] = np.sum(eigenvectors[i], axis=0).real

    results = []
    seen_canonical = set()

    if is_gamma and dim >= 2:
        all_coeffs = _generate_all_directions(dim, pol_matrix, include_nonmaximal)
        for coeff in all_coeffs:
            cell, scaled = _build_modulation(md, freq_idx, coeff, amplitude)
            if cell is None:
                continue

            pol_dir = _compute_polarization_direction(eigenvectors, coeff)
            opd_label = _canonicalize_direction(pol_dir)
            canonical_key = opd_label

            is_max = _is_maximal_opd_label(opd_label)
            if canonical_key in seen_canonical:
                continue
            seen_canonical.add(canonical_key)

            results.append(
                {
                    "atoms": phonopy_atoms_to_ase(cell),
                    "reference_atoms": base_atoms.copy(),
                    "opd_label": opd_label,
                    "opd_vector_eigvec": np.real(coeff).tolist(),
                    "polarization_direction": pol_dir.tolist(),
                    "is_maximal": is_max,
                    "is_gamma": True,
                }
            )
    else:
        ie = IsotropyEnumerator(
            md.little_rotations, md.little_translations, md.qpoint, irrep
        )

        for j, opd in enumerate(ie.order_parameter_directions):
            if opd.shape[0] != 1:
                continue

            vec = opd[0].real
            cell, scaled = _build_modulation(md, freq_idx, vec, amplitude)
            if cell is None:
                continue

            opd_label = _canonicalize_direction(vec)
            canonical_key = opd_label
            if canonical_key in seen_canonical:
                continue
            seen_canonical.add(canonical_key)

            results.append(
                {
                    "atoms": phonopy_atoms_to_ase(cell),
                    "reference_atoms": base_atoms.copy(),
                    "opd_label": opd_label,
                    "opd_vector_eigvec": vec.tolist(),
                    "polarization_direction": None,
                    "is_maximal": True,
                    "is_gamma": False,
                }
            )

        if include_nonmaximal and dim >= 2:
            all_coeffs = _generate_all_directions(dim, pol_matrix, include_nonmaximal)
            for coeff in all_coeffs:
                cell, scaled = _build_modulation(md, freq_idx, coeff, amplitude)
                if cell is None:
                    continue

                opd_label = _canonicalize_direction(coeff)
                canonical_key = opd_label
                if canonical_key in seen_canonical:
                    continue
                seen_canonical.add(canonical_key)

                results.append(
                    {
                        "atoms": phonopy_atoms_to_ase(cell),
                        "reference_atoms": base_atoms.copy(),
                        "opd_label": opd_label,
                        "opd_vector_eigvec": np.real(coeff).tolist(),
                        "polarization_direction": None,
                        "is_maximal": False,
                        "is_gamma": False,
                    }
                )

    return results


def get_high_symmetry_modulations(
    phonopy_yaml_or_object,
    qpoint,
    band_index,
    supercell_matrix,
    amplitude=0.5,
    symprec=1e-5,
    degeneracy_tolerance=1e-3,
) -> list[Atoms]:
    results = get_modulations_with_opd_info(
        phonopy_yaml_or_object,
        qpoint,
        band_index,
        supercell_matrix,
        amplitude=amplitude,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        include_nonmaximal=True,
    )
    return [r["atoms"] for r in results]


def get_mode_displacements(
    phonopy_yaml_path,
    qpoint,
    band_index,
    supercell_matrix,
    amplitude=0.5,
    symprec=1e-5,
    degeneracy_tolerance=1e-3,
):
    phonon = phonopy_load(phonopy_yaml_path)
    qpoint = np.array(qpoint)
    smat = np.array(supercell_matrix)

    sc = Supercell(phonon.primitive, smat)
    base_frac = sc.scaled_positions.copy()
    base_ase = phonopy_atoms_to_ase(sc)

    phonon.run_qpoints([qpoint], with_eigenvectors=True)
    freqs = phonon.get_qpoints_dict()["frequencies"][0]
    target_freq = freqs[band_index]

    md = Modulation.with_supercell_and_symmetry_search(
        dynamical_matrix=phonon.dynamical_matrix,
        supercell_matrix=smat.tolist(),
        qpoint=qpoint.tolist(),
        factor=phonon.unit_conversion_factor,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
    )

    freq_idx = _find_eigenspace_index(md, target_freq, degeneracy_tolerance)
    if freq_idx is None:
        return [], base_ase

    mod_cells = md.get_high_symmetry_modulated_supercells(
        frequency_index=freq_idx,
        maximal_displacement=amplitude,
    )

    displacements = []
    for cell in mod_cells:
        disp = cell.scaled_positions - base_frac
        disp -= np.round(disp)
        displacements.append(disp)

    return displacements, base_ase


def get_multi_mode_modulations_with_info(
    phonopy_yaml_path,
    mode_specs,
    supercell_matrix,
    amplitude=0.5,
    symprec=1e-5,
    degeneracy_tolerance=1e-3,
    include_nonmaximal=True,
    max_combinations=100,
):
    phonon = phonopy_load(phonopy_yaml_path)
    smat = np.array(supercell_matrix).reshape(3, 3)

    sc = Supercell(phonon.primitive, smat)
    base_frac = sc.scaled_positions.copy()
    base_ase = phonopy_atoms_to_ase(sc)

    per_mode = []
    for qpoint, band_index in mode_specs:
        qpoint_arr = np.array(qpoint)
        info_list = get_modulations_with_opd_info(
            phonon,
            qpoint=qpoint_arr.tolist(),
            band_index=band_index,
            supercell_matrix=smat.tolist(),
            amplitude=amplitude,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            include_nonmaximal=include_nonmaximal,
        )
        if not info_list:
            return [], base_ase

        mode_entries = []
        for info in info_list:
            mod_atoms = info["atoms"]
            disp = mod_atoms.get_scaled_positions() - base_frac
            disp -= np.round(disp)
            mode_entries.append(
                {
                    "displacement": disp,
                    "opd_label": info["opd_label"],
                    "is_maximal": info["is_maximal"],
                }
            )
        per_mode.append(mode_entries)

    combos = list(itertools.product(*per_mode))
    if len(combos) > max_combinations:
        print(
            f"[modulate] Warning: {len(combos)} multi-mode combinations, "
            f"limiting to {max_combinations}"
        )
        combos = combos[:max_combinations]

    results = []
    cell_mat = base_ase.get_cell()
    symbols = list(base_ase.get_chemical_symbols())

    for combo in combos:
        total_disp = np.zeros_like(base_frac)
        opd_labels = []
        all_maximal = True
        for entry in combo:
            total_disp += entry["displacement"]
            opd_labels.append(entry["opd_label"])
            if not entry["is_maximal"]:
                all_maximal = False
        new_frac = (base_frac + total_disp) % 1.0
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=new_frac,
            cell=cell_mat,
            pbc=True,
        )
        results.append(
            {
                "atoms": atoms,
                "opd_labels": opd_labels,
                "opd_is_maximal": all_maximal,
            }
        )

    return results, base_ase


def get_multi_mode_modulations(
    phonopy_yaml_path,
    mode_specs,
    supercell_matrix,
    amplitude=0.5,
    symprec=1e-5,
    degeneracy_tolerance=1e-3,
    max_combinations=50,
):
    all_disps = []
    base_atoms = None

    for qpoint, band_index in mode_specs:
        disps, base = get_mode_displacements(
            phonopy_yaml_path,
            qpoint,
            band_index,
            supercell_matrix,
            amplitude=amplitude,
            symprec=symprec,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        if not disps:
            return []
        if base_atoms is None:
            base_atoms = base
        all_disps.append(disps)

    combos = list(itertools.product(*all_disps))
    if len(combos) > max_combinations:
        print(
            f"[modulate] Warning: {len(combos)} OPD combinations, "
            f"limiting to {max_combinations}"
        )
        combos = combos[:max_combinations]

    results = []
    base_frac = base_atoms.get_scaled_positions()
    cell = base_atoms.get_cell()
    symbols = list(base_atoms.get_chemical_symbols())

    for combo in combos:
        total_disp = np.zeros_like(base_frac)
        for disp in combo:
            total_disp += disp
        new_frac = (base_frac + total_disp) % 1.0
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=new_frac,
            cell=cell,
            pbc=True,
        )
        results.append(atoms)

    return results
