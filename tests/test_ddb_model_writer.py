import yaml
from ase.build import bulk

from atomchain.ddb.model import (
    DdbDerivativeBlock,
    DdbDocument,
    DdbHeader,
    DdbPerturbation,
)
from atomchain.ddb.validate import compare_derivative_blocks, validate_ddb_metadata
from atomchain.ddb.writer import ddb_to_lines, write_ddb


def minimal_document():
    atoms = bulk("Al", "fcc", a=4.05)
    header = DdbHeader.from_atoms(atoms)
    document = DdbDocument(header=header, metadata={"test": True})
    document.add_derivative(
        DdbDerivativeBlock(
            qpoint=(0, 0, 0),
            perturbation_i=DdbPerturbation(
                kind="displacement", atom_index=0, cart_direction=0
            ),
            perturbation_j=DdbPerturbation(
                kind="displacement", atom_index=0, cart_direction=1
            ),
            value=1.25 + 0.5j,
            units="Ha/Bohr^2",
            source="test",
        )
    )
    document.add_derivative(
        DdbDerivativeBlock(
            qpoint=(0, 0, 0),
            perturbation_i=DdbPerturbation(kind="strain", voigt_index=0),
            perturbation_j=DdbPerturbation(kind="strain", voigt_index=5),
            value=2.0,
            units="Ha",
            source="stress_fd",
        )
    )
    return document


def test_document_metadata_is_deterministic():
    metadata1 = minimal_document().to_metadata()
    metadata2 = minimal_document().to_metadata()
    assert metadata1 == metadata2
    assert len(metadata1["derivative_blocks"]) == 2


def test_writer_emits_header_and_derivative_block(tmp_path):
    path = tmp_path / "out.ddb"
    write_ddb(minimal_document(), path)
    text = path.read_text(encoding="utf-8")
    assert " **** DERIVATIVE DATABASE ****" in text
    assert "+DDB, Version number" in text
    assert "     znucl" in text
    assert "2nd derivatives (non-stat.)" in text
    assert "qpt" in text
    assert (tmp_path / "out.ddb.yaml").exists()
    assert validate_ddb_metadata(tmp_path / "out.ddb.yaml") == []


def test_writer_emits_reference_total_energy_block(tmp_path):
    document = minimal_document()
    document.reference_energy_ha = -12.5
    path = tmp_path / "energy.ddb"
    write_ddb(document, path)
    text = path.read_text(encoding="utf-8")
    metadata = yaml.safe_load(
        (tmp_path / "energy.ddb.yaml").read_text(encoding="utf-8")
    )

    assert "Number of data blocks=    2" in text
    assert " Total energy                 - # elements :           1" in text
    assert "-1.25000000000000D+01" in text
    assert metadata["reference_energy_ha"] == -12.5


def test_writer_is_byte_stable(tmp_path):
    path1 = tmp_path / "one.ddb"
    path2 = tmp_path / "two.ddb"
    write_ddb(minimal_document(), path1)
    write_ddb(minimal_document(), path2)
    assert path1.read_bytes() == path2.read_bytes()


def test_validation_compare_detects_value_difference(tmp_path):
    path = tmp_path / "out.ddb"
    write_ddb(minimal_document(), path)
    metadata = yaml.safe_load((tmp_path / "out.ddb.yaml").read_text(encoding="utf-8"))
    changed = yaml.safe_load((tmp_path / "out.ddb.yaml").read_text(encoding="utf-8"))
    changed["derivative_blocks"][0]["value"]["real"] += 1.0
    assert compare_derivative_blocks(metadata, metadata) == []
    assert compare_derivative_blocks(changed, metadata)


def test_writer_emits_real_symmetry_in_abinit_column_order():
    document = minimal_document()
    document.header.symmetry_operations = [
        {
            "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [0.25, 0.5, 0.0],
        }
    ]
    lines = ddb_to_lines(document)
    assert "      nsym         1" in lines
    symrel = next(line for line in lines if "symrel" in line)
    assert symrel.endswith("    0    1    0   -1    0    0    0    0    1")
    tnons = next(line for line in lines if "tnons" in line)
    assert "2.50000000000000D-01" in tnons
    assert "5.00000000000000D-01" in tnons
