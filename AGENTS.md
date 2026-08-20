# Repository Instructions

## Development Environment

- Use Poetry for dependency management and development commands.
- Use Python 3.12 as the primary local development version.
- Maintain compatibility with all supported Python versions: 3.11, 3.12, and 3.13.

## Testing

- Run targeted tests during development, focusing on the code being changed.
- Add or update tests for every bug fix and behavioral change.
- Before completing work, run the full test suite:

  ```bash
  poetry run pytest
  ```

- Before opening or updating a pull request, run the supported-version test matrix when practical:

  ```bash
  poetry run tox
  ```

## Change Scope

- Keep changes focused on the requested task.
- Do not perform unrelated refactoring or cleanup.
- Do not bump the package version or edit the changelog unless explicitly requested.

## Repository Operations

- Do not create commits, push branches, merge changes, open or modify issues, or open or modify pull requests unless explicitly requested.
