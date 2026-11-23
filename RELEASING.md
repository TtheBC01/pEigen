# Releasing pEigen

Releases are published to PyPI by `.github/workflows/release.yml` when a `v*` tag is pushed.
The same workflow can be run without a tag as a build-only dry run.

## One-time setup

1. On PyPI, add a Trusted Publisher for the existing `peigen` project: repository
   `TtheBC01/pEigen`, workflow `release.yml`, environment `pypi`.
2. In the GitHub repository settings, create a `pypi` environment (optionally with required
   reviewers, so publishing waits for a manual approval).

## Release steps

1. Make sure CI is green on `main`.
2. Set the version in `pyproject.toml` and move the `Unreleased` heading in `CHANGELOG.md` to the
   version and date.
3. **Dry run:** in GitHub, open Actions → Release → *Run workflow* on `main`. This builds and
   tests every wheel and the sdist but publishes nothing. Wheel tests run with
   `PEIGEN_REQUIRE_LAPACK=1`, so a wheel that silently lost BLAS/LAPACK fails here. Download the
   artifacts if you want to inspect them.
4. Tag and push:

   ```bash
   git tag -a v2.0.0 -m "pEigen 2.0.0"
   git push origin v2.0.0
   ```

5. The workflow then:
   - checks that the tag matches the `pyproject.toml` version (and stops if not)
   - builds and tests wheels for CPython 3.10-3.15 on Linux x86_64/aarch64 and macOS
     arm64/x86_64, and the sdist
   - publishes everything to PyPI (after approval, if the `pypi` environment requires it)
   - creates a GitHub Release with the artifacts and generated notes

Build settings (Python versions, skipped targets, BLAS setup, test command) live under
`[tool.cibuildwheel]` in `pyproject.toml`; the workflow only chooses runners.

## Verify published artifacts

In a fresh environment:

```bash
python -m pip install --upgrade pip
python -m pip install "peigen[sparse]"
python -c "import peigen; peigen.show_build_config()"
```

`lapack_enabled` should be `True`, with `accelerate` (macOS) or `system-lapack` (Linux) as the
backend.
