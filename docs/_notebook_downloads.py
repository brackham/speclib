"""Make notebook downloads work in both hosted and local HTML documentation."""

from base64 import b64encode
from pathlib import Path

from sphinx import addnodes
from sphinx.writers.html5 import HTML5Translator


def notebook_data_url(source):
    encoded = b64encode(source.read_bytes()).decode("ascii")
    return f"data:application/x-ipynb+json;base64,{encoded}"


def visit_download_reference(translator, node):
    if translator.builder.download_support and node.get("filename", "").endswith(".ipynb"):
        _, source = translator.builder.env.relfn2path(
            node["reftarget"], node.get("refdoc")
        )
        source = Path(source)
        # Browsers can ignore file:// download links. A data URL also works
        # when the generated site is opened directly from disk.
        translator.body.append(translator.starttag(
            node, "a", "", **{
                "class": "reference download internal",
                "download": source.name,
                "href": notebook_data_url(source),
            }
        ))
        translator.context.append("</a>")
    else:
        HTML5Translator.visit_download_reference(translator, node)


def add_notebook_download(app, pagename, templatename, context, doctree):
    if doctree is None:
        return
    source = Path(app.env.doc2path(pagename))
    if source.suffix == ".ipynb":
        context["notebook_download"] = {
            "filename": source.name,
            "url": notebook_data_url(source),
        }


def setup(app):
    app.add_node(
        addnodes.download_reference,
        html=(visit_download_reference, HTML5Translator.depart_download_reference),
        override=True,
    )
    app.connect("html-page-context", add_notebook_download)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
