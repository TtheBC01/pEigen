import os

import pytest

import peigen


def test_build_config_has_expected_keys():
    cfg = peigen.build_config()
    expected = {
        "version",
        "build_type",
        "blas_enabled",
        "blas_backend",
        "lapack_enabled",
        "lapack_backend",
        "openmp_enabled",
        "vectorization_enabled",
        "eigen_mpl2_only",
        "eigen_version",
    }
    assert expected.issubset(cfg.keys())
    assert isinstance(cfg["version"], str)
    assert isinstance(cfg["blas_enabled"], bool)
    assert cfg["eigen_mpl2_only"] is True
    assert cfg["eigen_version"].count(".") >= 1


def test_show_build_config_prints(capsys):
    peigen.show_build_config()
    captured = capsys.readouterr().out
    assert "pEigen build configuration" in captured
    assert "lapack_backend:" in captured


@pytest.mark.skipif(os.environ.get("PEIGEN_REQUIRE_LAPACK") != "1", reason="set PEIGEN_REQUIRE_LAPACK=1 to enforce")
def test_release_build_has_blas_and_lapack():
    """Release wheels and CI builds must link BLAS/LAPACK; a silent fallback would halve performance."""
    cfg = peigen.build_config()
    assert cfg["blas_enabled"], cfg
    assert cfg["lapack_enabled"], cfg
    assert cfg["lapack_backend"] != "none", cfg
