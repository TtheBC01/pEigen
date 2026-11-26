"""Run every copy-paste example published on the website (website/*.html, code blocks marked
data-example), so the site can't drift from the API. Skipped when the website folder isn't present
(e.g. when testing an installed sdist)."""

import html
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("scipy.sparse")

WEBSITE = pathlib.Path(__file__).resolve().parent.parent / "website"
BLOCK = re.compile(r'<code class="language-python" data-example>(.*?)</code>', re.S)


def _examples():
    if not WEBSITE.is_dir():
        return []
    found = []
    for page in sorted(WEBSITE.glob("*.html")):
        for i, block in enumerate(BLOCK.findall(page.read_text(encoding="utf-8"))):
            found.append(pytest.param(html.unescape(block), id=f"{page.stem}-{i}"))
    return found


EXAMPLES = _examples()


@pytest.mark.skipif(not EXAMPLES, reason="website/ not available")
@pytest.mark.parametrize("source", EXAMPLES)
def test_website_example_runs(source, tmp_path):
    script = tmp_path / "example.py"
    script.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-W", "error::FutureWarning", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_examples_were_found():
    if WEBSITE.is_dir():
        assert len(EXAMPLES) >= 10
