"""Check tutorial navigation and downloads without executing model workflows."""

from base64 import b64decode
import ast
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
import shutil

import pytest

pytest.importorskip("sphinx", reason="Documentation checks require Sphinx")
pytest.importorskip("myst_nb", reason="Notebook rendering requires MyST-NB")
pytest.importorskip("furo", reason="Tutorial page controls use Furo")
from sphinx.application import Sphinx


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchors = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.anchors.append(dict(attrs))


def test_tutorial_pages_and_notebook_downloads(tmp_path, monkeypatch):
    docs = Path(__file__).resolve().parents[1] / "docs"
    monkeypatch.syspath_prepend(str(docs))
    source = tmp_path / "source"
    tutorials = source / "tutorials"
    tutorials.mkdir(parents=True)
    theme_options = next(
        ast.literal_eval(node.value)
        for node in ast.parse((docs / "conf.py").read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "html_theme_options"
                for target in node.targets)
    )
    (source / "conf.py").write_text(
        'extensions = ["_notebook_downloads", "myst_nb"]\n'
        'nb_execution_mode = "off"\n'
        'html_theme = "furo"\n'
        f'html_theme_options = {theme_options!r}\n'
        f'templates_path = [{str(docs / "_templates")!r}]\n'
    )
    (source / "index.rst").write_text(
        "Documentation\n=============\n\n.. toctree::\n\n"
        "   installation\n   tutorials/index\n\n"
        ":download:`Notes <notes.txt>`\n"
    )
    (source / "installation.rst").write_text("Installation\n============\n")
    (source / "notes.txt").write_text("A regular download.\n")
    for name in (
        "models/index", "user_guide/photometry", "user_guide/instrument_profiles",
        "user_guide/instrument_provenance",
    ):
        page = source / f"{name}.rst"
        page.parent.mkdir(exist_ok=True)
        page.write_text(
            ":orphan:\n\n"
            + (".. _g280-calibration-status:\n\n" if name.endswith("instrument_provenance") else "")
            + "Reference\n=========\n"
        )
    shutil.copyfile(docs / "tutorials/index.rst", tutorials / "index.rst")
    notebooks = sorted((docs / "tutorials").glob("*.ipynb"))
    assert len(notebooks) == 5
    for notebook in notebooks:
        shutil.copyfile(notebook, tutorials / notebook.name)

    output = tmp_path / "html"
    warnings = StringIO()
    app = Sphinx(
        str(source), str(source), str(output), str(tmp_path / "doctrees"),
        "html", status=StringIO(), warning=warnings, warningiserror=True,
        freshenv=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0, warnings.getvalue()
    assert not warnings.getvalue()

    table = (output / "tutorials/index.html").read_text().split("<tbody>", 1)[1]
    rows = table.split("</tbody>", 1)[0].split("<tr")[1:]
    assert len(rows) == len(notebooks)
    for row in rows:
        cells = row.split("</td>")
        title_links = Links()
        title_links.feed(cells[0])
        assert len(title_links.anchors) == 1
        page = title_links.anchors[0]["href"]
        assert (output / "tutorials" / page).is_file()

        downloads = Links()
        downloads.feed(cells[2])
        assert len(downloads.anchors) == 1
        link = downloads.anchors[0]
        filename = link["download"]
        assert filename == f"{Path(page).stem}.ipynb"
        prefix, encoded = link["href"].split(",", 1)
        assert prefix == "data:application/x-ipynb+json;base64"
        original = (tutorials / filename).read_bytes()
        assert b64decode(encoded, validate=True) == original
        assert next((output / "_downloads").rglob(filename)).read_bytes() == original

        # Each notebook has its own download beside the other page controls.
        page_html = (output / "tutorials" / page).read_text()
        controls = page_html.split('class="content-icon-container"', 1)[1].split("<article", 1)[0]
        page_links = Links()
        page_links.feed(controls)
        buttons = [a for a in page_links.anchors if "notebook-download" in a.get("class", "").split()]
        assert len(buttons) == 1
        assert buttons[0]["download"] == filename
        assert buttons[0]["href"] == link["href"]
        assert buttons[0]["aria-label"] == "Download notebook"
        view_buttons = [a for a in page_links.anchors if "github-view" in a.get("class", "").split()]
        assert len(view_buttons) == 1
        assert view_buttons[0]["href"] == (
            f"https://github.com/brackham/speclib/blob/main/docs/tutorials/{filename}?plain=true"
        )
        assert view_buttons[0]["aria-label"] == "View this page on GitHub"
        assert 'class="github-icon"' in controls
        assert "edit-this-page" not in controls

    for page in ("index", "installation", "tutorials/index"):
        assert 'class="muted-link notebook-download"' not in (output / f"{page}.html").read_text()

    ordinary = Links()
    ordinary.feed((output / "index.html").read_text())
    link = next(a for a in ordinary.anchors if a.get("href", "").endswith("notes.txt"))
    assert "download" in link
    assert (output / link["href"]).read_text() == "A regular download.\n"
