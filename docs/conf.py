# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import speclib

project = "speclib"
copyright = "2021–2026, Benjamin V. Rackham"
author = "Benjamin V. Rackham"
release = speclib.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "dollarmath",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
autodoc_typehints = "description"

# Every tutorial is deliberately lightweight and offline.  Executing notebooks
# here keeps rendered output synchronized with the source checkout without
# committing generated notebook output.
nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_raise_on_error = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"speclib {release}"
html_theme_options = {
    "source_repository": "https://github.com/brackham/speclib/",
    "source_branch": "main",
    "source_directory": "docs/",
}
