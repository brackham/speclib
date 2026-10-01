# Contributing to speclib

Contributions are welcome and appreciated. Useful contributions include bug reports, documentation improvements, tests, bug fixes, maintenance work, and feature additions.

## Before you start

Small fixes, including typos and documentation improvements, can be submitted directly. Substantial new features, API changes, new spectral libraries, and architectural changes should generally be discussed in an [issue](https://github.com/brackham/speclib/issues/new/choose) first. The issue chooser has forms for bugs, feature/API requests, and model/spectral-library requests, plus a blank issue option for other topics.

Keep each pull request to **one logical change**. Avoid combining unrelated fixes, refactors, features, or formatting changes.

## Local development

Use Python 3.12 for local development and keep changes compatible with Python 3.11, 3.12, and 3.13. With [Poetry](https://python-poetry.org/) installed, clone your fork and install the package and development dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/speclib.git
cd speclib
poetry env use python3.12
poetry install
```

For a fork, add the source repository as `upstream` once, then create a focused branch from its current `main`:

```bash
git remote add upstream https://github.com/brackham/speclib.git
git fetch upstream
git switch -c your-change upstream/main
```

Keep the branch reasonably current with `main` and resolve merge/rebase conflicts before review where practical.

### Tests

Run targeted tests while developing, then the full suite before submitting a PR:

```bash
poetry run pytest tests/test_core.py
poetry run pytest
```

Choose the test file relevant to your change. Add tests for new behavior and regression tests for bug fixes where practical; explain any testing gaps in the PR.

When practical, run the supported-version matrix before submitting or updating a PR:

```bash
poetry run tox
```

The matrix covers Python 3.11, 3.12, and 3.13. Install those interpreters to run all three environments; `tox.ini` skips interpreters that are unavailable.

### Documentation

User-facing behavior should be documented where appropriate in `docs/`, including API details, examples, and model-library notes. Sphinx and the documentation dependencies are installed by `poetry install`. From the repository root, build the HTML documentation and check links with:

```bash
poetry run sphinx-build -b html -W --keep-going docs docs/_build/html
poetry run sphinx-build -b linkcheck docs docs/_build/linkcheck
```

Open `docs/_build/html/index.html` to review the rendered pages. The build executes lightweight, offline tutorial notebooks. Link checking needs network access; report any failures caused by unavailable external sites. This guide is also included in the documentation through `docs/contributing.rst`.

## Preparing a pull request

- Describe what changed and why, link a related issue when applicable, and report the tests/checks you ran, including any skipped checks.
- Keep changes directly related to the PR's purpose and include relevant tests and documentation.
- Preserve existing public API behavior unless a change is intentional; describe intentional API changes explicitly in the issue and PR.
- For scientific, model-grid, or data changes, cite authoritative publications, DOIs, archives, or model inventories. Explain changes to quantities, units, parameter axes, interpolation, or scientific assumptions so they can be reviewed against those sources.

Use the PR template as a short review aid; mark sections or checklist items as not applicable when appropriate.

## Contribution credit and citation metadata

`speclib` distinguishes between creators/authors and contributors for citation metadata.

Creators/authors are people who have made substantial intellectual, scientific, architectural, or long-term maintenance contributions to the software and who should appear in the formal citation for a release. Contributors are people who have made useful code, documentation, testing, bug-fix, maintenance, or feature contributions that should be credited but do not necessarily imply authorship of the citable software release.

Pull requests, issue reports, bug fixes, documentation improvements, and small feature additions are gratefully acknowledged, but they do not automatically imply creator/authorship status on Zenodo or in `CITATION.cff`. Creator/authorship for a release is determined by the maintainer based on the nature, scope, and intellectual contribution of the work included in that release.

Contributors may be credited in GitHub contributor history, release notes, the changelog, and/or Zenodo contributor metadata.
