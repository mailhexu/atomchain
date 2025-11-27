"""
Tests for init_model module.

Tests calculator initialization for different model types.

Purpose:
    Verify that init_calc() correctly initializes calculators for supported
    ML potential types and raises appropriate errors for unsupported types.

How to run:
    pytest tests/test_init_model.py
    or
    pytest  # runs all tests
"""

import pytest

from atomchain.init_model import init_calc


def test_init_calc_invalid_model_type():
    """Test that init_calc raises ValueError for unsupported model types."""
    with pytest.raises(ValueError, match="model_type not recognized"):
        init_calc(model_type="unsupported_model")


def test_init_calc_chgnet():
    """Test CHGNet calculator initialization.
    
    This test verifies that CHGNet calculator can be initialized.
    It may be skipped if CHGNet is not installed.
    """
    pytest.importorskip("chgnet")
    calc = init_calc(model_type="chgnet")
    assert calc is not None
    assert hasattr(calc, "calculate")


def test_init_calc_m3gnet():
    """Test M3GNet calculator initialization.
    
    This test verifies that M3GNet calculator can be initialized.
    It may be skipped if M3GNet is not installed.
    """
    pytest.importorskip("m3gnet")
    calc = init_calc(model_type="m3gnet")
    assert calc is not None
    assert hasattr(calc, "calculate")


def test_init_calc_matgl():
    """Test MatGL calculator initialization.
    
    This test verifies that MatGL calculator can be initialized.
    It may be skipped if MatGL is not installed.
    """
    pytest.importorskip("matgl")
    calc = init_calc(model_type="matgl")
    assert calc is not None
    assert hasattr(calc, "calculate")


def test_init_calc_mace():
    """Test MACE calculator initialization.
    
    This test verifies that MACE calculator can be initialized.
    It may be skipped if MACE is not installed.
    """
    pytest.importorskip("mace")
    calc = init_calc(model_type="mace")
    assert calc is not None
    assert hasattr(calc, "calculate")


def test_init_calc_deepmd():
    """Test DeepMD calculator initialization.
    
    This test verifies that DeepMD calculator can be initialized with a model path.
    It may be skipped if DeepMD is not installed.
    """
    pytest.importorskip("deepmd")
    # DeepMD requires a model path, so we expect it to fail without one
    with pytest.raises((ValueError, FileNotFoundError, TypeError)):
        init_calc(model_type="deepmd", model_path=None)
