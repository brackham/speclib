# Contributing to speclib

Bug reports, ideas, documentation improvements, and code contributions are welcome.
Use [GitHub Issues](https://github.com/brackham/speclib/issues/new/choose) to report a bug or request a feature. For bugs, include a small example, the error message, and your Python and `speclib` versions. Please discuss substantial features or API changes in an issue before starting work.

## Development environment

Fork the repository, clone your fork, and create a branch for your change. Use Python 3.12 and [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/YOUR_USERNAME/speclib.git
cd speclib
poetry env use python3.12
poetry install
git switch -c your-change
```

Run `poetry install` again after pulling dependency changes. Keep code compatible with Python 3.11, 3.12, and 3.13.

## Tests and documentation

Run the tests relevant to your change while developing, then the full suite:

```bash
poetry run pytest tests/test_core.py
poetry run pytest
```

When practical, also run `poetry run tox` to test all supported Python versions. Tox skips interpreters that are not installed.

Build the documentation with warnings treated as errors:

```bash
poetry run sphinx-build -b html -W --keep-going docs docs/_build/html
poetry run sphinx-build -b linkcheck docs docs/_build/linkcheck
```

Open `docs/_build/html/index.html` to review the result. The HTML build executes the five offline tutorials. Link checking needs network access; mention unavailable external sites in your pull request.

## Pull requests

Keep each pull request focused on one change. Explain the problem, describe the result, and list the checks you ran. Include tests for bug fixes and new behavior, and update the relevant documentation. Cite scientific sources and explain any changes to units, model parameters, interpolation, or assumptions.

## Credit

Contributions are acknowledged through GitHub history, release notes, or citation metadata. The maintainer determines release authorship in `CITATION.cff` and Zenodo from the scientific, intellectual, and maintenance contributions to that release.
