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
    }
    assert expected.issubset(cfg.keys())
    assert isinstance(cfg["version"], str)
    assert isinstance(cfg["blas_enabled"], bool)
    assert cfg["eigen_mpl2_only"] is True


def test_show_build_config_prints(capsys):
    peigen.show_build_config()
    captured = capsys.readouterr().out
    assert "pEigen build configuration" in captured
    assert "lapack_backend:" in captured
