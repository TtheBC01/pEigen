# Releasing pEigen

Releases are published to PyPI by `.github/workflows/release.yml` when a `v*` tag is pushed.
The same workflow can be run without a tag as a build-only dry run.

## One-time setup

The PyPI distribution is named `peigentux` (the import name is `peigen`). Uploads use PyPI
Trusted Publishing, so no password or API token is stored in GitHub.

1. Create an account on [pypi.org](https://pypi.org/account/register/) and enable two-factor
   authentication (PyPI requires it to publish).
2. Before the first release, while the project does not exist yet, add a *pending publisher*:
   PyPI → your account → **Publishing** → *Add a new pending publisher* → GitHub, with
   - PyPI project name: `peigentux`
   - Owner: `TtheBC01`
   - Repository name: `pEigen`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   The first successful upload creates the project under your account and turns the pending
   publisher into a normal one (later visible under the project's **Manage → Publishing**).
3. In the GitHub repository settings, create a `pypi` environment (optionally with required
   reviewers, so publishing waits for a manual approval).

If the publish job fails with `invalid-publisher`, the values above do not match the workflow
run; fix them on PyPI and use *Re-run failed jobs* on the release run, which reuses the built
wheels.

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
python -m pip install "peigentux[sparse]"
python -c "import peigen; peigen.show_build_config()"
```

`lapack_enabled` should be `True`, with `accelerate` (macOS) or `system-lapack` (Linux) as the
backend.
