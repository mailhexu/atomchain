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

import sys
import types

import pytest

from atomchain.init_model import (
    MACE_R2SCAN_MODEL_URL,
    _ensure_mace_r2scan_model,
    _get_torch_device,
    init_calc,
)


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


def test_mace_device_prefers_cuda_then_mps_then_cpu():
    """MACE device selection should support CUDA and Apple Silicon MPS."""
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
    )
    assert _get_torch_device(torch) == "cuda"

    torch.cuda.is_available = lambda: False
    assert _get_torch_device(torch) == "mps"

    torch.backends.mps.is_available = lambda: False
    assert _get_torch_device(torch) == "cpu"

    torch.backends = types.SimpleNamespace()
    assert _get_torch_device(torch) == "cpu"


def test_init_calc_mace_uses_mps_when_cuda_unavailable(monkeypatch):
    """MACE initialization should pass device='mps' on Apple Silicon."""
    calls = []

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
    )

    def fake_mace_mp(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(calculate=lambda *args, **kwargs: None)

    fake_mace = types.ModuleType("mace")
    fake_calculators = types.ModuleType("mace.calculators")
    fake_calculators.mace_mp = fake_mace_mp
    fake_mace.calculators = fake_calculators

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mace", fake_mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", fake_calculators)

    calc = init_calc(model_type="mace")

    assert calc is not None
    assert calls[0]["device"] == "mps"


def test_mace_r2scan_model_downloads_when_missing(tmp_path, monkeypatch):
    """Missing MACE R2SCAN model should be downloaded to the requested path."""
    model_path = tmp_path / "mace" / "mace-mh-1.model"
    calls = []

    def mock_urlretrieve(url, filename):
        calls.append((url, filename))
        model_path.write_text("model", encoding="utf-8")

    monkeypatch.setattr(
        "atomchain.init_model.urllib.request.urlretrieve", mock_urlretrieve
    )

    result = _ensure_mace_r2scan_model(model_path=str(model_path))

    assert result == str(model_path)
    assert model_path.exists()
    assert calls == [(MACE_R2SCAN_MODEL_URL, str(model_path))]


def test_mace_r2scan_model_download_allows_current_directory_path(
    tmp_path, monkeypatch
):
    """Custom model paths without a directory component should be valid."""
    model_path = tmp_path / "mace-mh-1.model"
    calls = []

    def mock_urlretrieve(url, filename):
        calls.append((url, filename))
        model_path.write_text("model", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "atomchain.init_model.urllib.request.urlretrieve", mock_urlretrieve
    )

    result = _ensure_mace_r2scan_model(model_path="mace-mh-1.model")

    assert result == "mace-mh-1.model"
    assert model_path.exists()
    assert calls == [(MACE_R2SCAN_MODEL_URL, "mace-mh-1.model")]


def test_mace_r2scan_model_download_failure_message(tmp_path, monkeypatch):
    """Failed automatic download should explain the exact URL and save path."""
    model_path = tmp_path / "mace" / "mace-mh-1.model"

    def mock_urlretrieve(url, filename):
        raise OSError("network unavailable")

    monkeypatch.setattr(
        "atomchain.init_model.urllib.request.urlretrieve", mock_urlretrieve
    )

    with pytest.raises(RuntimeError) as excinfo:
        _ensure_mace_r2scan_model(model_path=str(model_path))

    message = str(excinfo.value)
    assert MACE_R2SCAN_MODEL_URL in message
    assert str(model_path) in message


def test_init_calc_deepmd():
    """Test DeepMD calculator initialization.

    This test verifies that DeepMD calculator can be initialized with a model path.
    It may be skipped if DeepMD is not installed.
    """
    pytest.importorskip("deepmd")
    # DeepMD requires a model path, so we expect it to fail without one
    with pytest.raises((ValueError, FileNotFoundError, TypeError)):
        init_calc(model_type="deepmd", model_path=None)


def test_init_calc_multibinit_no_config():
    """Test that MULTIBINIT raises ValueError when model_path is None."""
    with pytest.raises(ValueError, match="MULTIBINIT requires a configuration file"):
        init_calc(model_type="multibinit", model_path=None)


def test_init_calc_multibinit_alias_no_config():
    """Test that MULTIBINIT alias 'mb' raises ValueError when model_path is None."""
    with pytest.raises(ValueError, match="MULTIBINIT requires a configuration file"):
        init_calc(model_type="mb", model_path=None)


def test_init_calc_multibinit_nonexistent_config():
    """Test that MULTIBINIT raises FileNotFoundError for non-existent config file."""
    with pytest.raises(FileNotFoundError, match="nonexistent.conf"):
        init_calc(model_type="multibinit", model_path="nonexistent.conf")


def test_init_calc_multibinit():
    """Test MULTIBINIT calculator initialization with config file.

    This test verifies that MULTIBINIT calculator can be initialized.
    It may be skipped if pymultibinit is not installed.

    Note: This test will fail if the referenced DDB and XML files don't exist,
    which is expected in a pure unit test environment. The test primarily
    validates the init_calc logic up to the point of calling pymultibinit.
    """
    pytest.importorskip("pymultibinit")
    import os

    # Get path to example config file
    test_dir = os.path.dirname(__file__)
    config_path = os.path.join(test_dir, "fixtures", "multibinit_example.conf")

    # This will fail because the DDB and XML files don't exist,
    # but it validates that init_calc correctly processes the request
    # and attempts to initialize the calculator
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        init_calc(model_type="multibinit", model_path=config_path)


def test_init_calc_multibinit_alias():
    """Test MULTIBINIT calculator initialization with 'mb' alias.

    This test verifies that the 'mb' alias works identically to 'multibinit'.
    It may be skipped if pymultibinit is not installed.
    """
    pytest.importorskip("pymultibinit")
    import os

    # Get path to example config file
    test_dir = os.path.dirname(__file__)
    config_path = os.path.join(test_dir, "fixtures", "multibinit_example.conf")

    # Same expectation as the full name test
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        init_calc(model_type="mb", model_path=config_path)
