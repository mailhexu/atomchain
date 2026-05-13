from atomchain.kpoints import _patch_symphon


def _symphon_irreps():
    _patch_symphon()
    from symphon.irreps.highsym import get_all_irreps_phonopy
    from symphon.irreps.phonopy import IrRepsPhonopy

    return get_all_irreps_phonopy, IrRepsPhonopy


def label_phonon_modes(
    phonopy_params,
    qpoint,
    kpname=None,
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
):
    _, IrRepsPhonopy = _symphon_irreps()
    irr = IrRepsPhonopy(
        phonopy_params=phonopy_params,
        qpoint=qpoint,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
    )
    irr.run(kpname=kpname)

    freqs = irr.get_frequencies(unit="THz")
    bcs_labels = irr.get_bcs_labels()
    mulliken_labels = irr.get_mulliken_labels()

    modes = []
    for i in range(len(freqs)):
        modes.append(
            {
                "band_index": i,
                "frequency": float(freqs[i]),
                "bcs_label": bcs_labels[i] if i < len(bcs_labels) else None,
                "mulliken_label": mulliken_labels[i]
                if i < len(mulliken_labels)
                else None,
            }
        )

    return {"frequencies": freqs, "modes": modes}


def get_all_labeled_modes(
    phonopy_params,
    symprec=1e-5,
    degeneracy_tolerance=1e-4,
    log_level=0,
):
    get_all_irreps_phonopy, _ = _symphon_irreps()
    all_irreps = get_all_irreps_phonopy(
        phonopy_params=phonopy_params,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance,
        log_level=log_level,
    )

    result = {}
    for label, irr_obj in all_irreps.items():
        irr_obj.run(kpname=label)
        freqs = irr_obj.get_frequencies(unit="THz")
        bcs_labels = irr_obj.get_bcs_labels()
        mulliken_labels = irr_obj.get_mulliken_labels()

        modes = []
        for i in range(len(freqs)):
            modes.append(
                {
                    "band_index": i,
                    "frequency": float(freqs[i]),
                    "bcs_label": bcs_labels[i] if i < len(bcs_labels) else None,
                    "mulliken_label": mulliken_labels[i]
                    if i < len(mulliken_labels)
                    else None,
                }
            )

        result[label] = {"frequencies": freqs, "modes": modes}

    return result


def get_imaginary_modes(all_labeled_modes, threshold=0.0, deg_tolerance=1e-3):
    imaginary = []

    for kpoint_label, data in all_labeled_modes.items():
        modes = data["modes"]
        for mode in modes:
            freq = mode["frequency"]
            if freq < threshold:
                degeneracy = sum(
                    1 for m in modes if abs(m["frequency"] - freq) < deg_tolerance
                )
                imaginary.append(
                    {
                        "kpoint_label": kpoint_label,
                        "band_index": mode["band_index"],
                        "frequency": freq,
                        "bcs_label": mode.get("bcs_label"),
                        "mulliken_label": mode.get("mulliken_label"),
                        "degeneracy": degeneracy,
                    }
                )

    return imaginary
