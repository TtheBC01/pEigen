# Releasing pEigen

This project publishes to production PyPI on version tags.

## One-time setup

1. In PyPI, create the `peigen` project (or preferred package name).
2. Configure Trusted Publishing for this GitHub repository and workflow.
3. In GitHub repository settings, create `pypi` environment (optional protection rules).

## Release steps

1. Ensure tests pass on `main`.
2. Update version in `pyproject.toml`.
3. Update changelog/release notes.
4. Create and push tag:

```bash
git tag v2.0.0
git push origin v2.0.0
```

5. GitHub Actions `release.yml` will:
   - build wheel artifacts for macOS arm64/x86_64 and Linux x86_64
   - build sdist
   - publish all artifacts to PyPI
   - create a GitHub Release with attached artifacts

## Verify published artifacts

```bash
python -m pip install --upgrade pip
python -m pip install peigen
python -c "import peigen; print(peigen.__all__)"
```
