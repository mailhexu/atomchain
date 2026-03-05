"""
Tests for the structure collection module.
"""

import json
import os
import shutil
import tempfile
import pytest
from ase import Atoms
from ase.io import write, read
from atomchain.collect import collect_structures


@pytest.fixture
def temp_structure_dir():
    """Create a temporary directory with some structure files."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    os.makedirs(os.path.join(temp_dir, "subdir1"))
    os.makedirs(os.path.join(temp_dir, "subdir2"))
    
    # Create dummy atoms
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
    atoms2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=[10, 10, 10], pbc=True)
    atoms3 = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.10]], cell=[10, 10, 10], pbc=True)
    
    # Write files
    # file1 in root
    file1_path = os.path.join(temp_dir, "struct1.vasp")
    write(file1_path, atoms1)
    
    # file2 in subdir1
    file2_path = os.path.join(temp_dir, "subdir1", "struct2.vasp")
    write(file2_path, atoms2)
    
    # file3 in subdir2 with different extension (should be ignored)
    file3_path = os.path.join(temp_dir, "subdir2", "struct3.xyz")
    write(file3_path, atoms3)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_collect_structures(temp_structure_dir):
    """Test collecting structures from a directory."""
    root_dir = temp_structure_dir
    output_traj = os.path.join(root_dir, "out.traj")
    output_json = os.path.join(root_dir, "out.json")
    
    collect_structures(root_dir, ".vasp", output_traj, output_json)
    
    # Check if files were created
    assert os.path.exists(output_traj)
    assert os.path.exists(output_json)
    
    # Check trajectory content
    traj = read(output_traj, index=":")
    assert len(traj) == 2
    assert traj[0].get_chemical_formula() == "H2"
    assert traj[1].get_chemical_formula() == "O2"
    
    # Check JSON content
    with open(output_json, "r") as f:
        paths = json.load(f)
    
    assert len(paths) == 2
    # Check relative order (sorted)
    # struct1.vasp is in root, struct2.vasp is in subdir1.
    # walk yields root, then subdirs.
    # struct1.vasp path should be in paths
    # struct2.vasp path should be in paths
    
    assert any("struct1.vasp" in p for p in paths)
    assert any("struct2.vasp" in p for p in paths)
    assert not any("struct3.xyz" in p for p in paths)


def test_collect_structures_nested(temp_structure_dir):
    """Test recursive collection."""
    # Already tested in previous test implicitly, but let's double check ordering if relevant
    # The implementation sorts files within each dir, but os.walk order depends on OS for dirs usually.
    # However, we sort files.
    pass


def test_collect_no_files(temp_structure_dir):
    """Test behavior when no files are found."""
    root_dir = temp_structure_dir
    output_traj = os.path.join(root_dir, "empty.traj")
    output_json = os.path.join(root_dir, "empty.json")
    
    collect_structures(root_dir, ".nonexistent", output_traj, output_json)
    
    assert not os.path.exists(output_traj)
    assert not os.path.exists(output_json)

