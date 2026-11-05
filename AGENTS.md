
# Repository Instructions

## Development Environment

- Use Poetry for dependency management and development commands.
- Use Python 3.12 as the primary local development version.
- Maintain compatibility with all supported Python versions: 3.11, 3.12, and 3.13.

## Scientific Integrity

- Preserve scientifically meaningful behavior and numerical accuracy.
- Do not introduce or extend scientific approximations without documenting their justification, assumptions, and limitations.
- Use authoritative primary literature and official calibration references for model-grid and instrument-profile changes.
- Do not silently extrapolate beyond supported scientific data or calibration ranges.
- Distinguish nominal approximations from empirically calibrated quantities.
- Preserve scientific provenance and reproducibility when modifying model grids, instrument profiles, or calibration data.
- Prefer focused regression tests that verify scientific behavior over excessive implementation-specific tests.

## Testing

- Run targeted tests during development, focusing on the behavior being changed.
- Add or update tests for bug fixes and behavioral changes. Avoid redundant tests that do not meaningfully improve coverage.
- For code changes, run the full test suite before completing work:

  ```bash
  poetry run pytest
  ```

- Before opening or updating a pull request, run the supported-version test matrix when practical, especially for changes affecting numerical behavior, dependencies, or compatibility:

  ```bash
  poetry run tox
  ```

- For documentation changes, run the relevant Sphinx build with warnings treated as errors when practical.
- Documentation-only changes generally do not require rerunning the full test suite.
- If a required validation command cannot run, document the reason, any alternative validation performed, and remaining limitations.
- Never report tests or checks as passed unless they were actually executed successfully.

## Documentation

- Update relevant documentation whenever public functionality or behavior changes.
- Follow the established style, terminology, and organization of the existing documentation.
- Keep API references, user guides, tutorials, and examples consistent with the implementation.
- Document scientific assumptions, limitations, and provenance whenever scientific functionality, model grids, or calibration data change.
- Keep documentation concise, accurate, and focused on what users need to understand and accomplish.
- Avoid adding implementation details or specialized functionality to the README unless essential for understanding or installing the package.
- Do not introduce unnecessary documentation files or duplicate information already maintained elsewhere.

## Change Scope and Changelog

- Keep changes focused on the requested task.
- Do not perform unrelated refactoring, cleanup, or formatting changes.
- Prefer simple, maintainable solutions consistent with existing package architecture and conventions.
- Do not bump the package version unless explicitly requested.
- Update the `[Unreleased]` section of `CHANGELOG.md` for meaningful user-facing changes, including:
  - New functionality and enhancements.
  - Bug fixes.
  - API changes and changes to existing behavior.
  - Scientific methodology or numerical behavior changes.
  - Changes to supported model grids, instrument profiles, or calibration data.
- Follow the existing changelog format and write concise entries from the user's perspective.
- Changelog entries are generally unnecessary for internal refactoring, tests, CI changes, or minor documentation corrections that do not affect user-facing behavior.
- Do not modify historical changelog entries unless correcting a factual error or explicitly instructed.

## Repository Operations

- Do not create commits, push branches, merge changes, open or modify issues, or open or modify pull requests unless explicitly requested.
- When explicitly authorized to create or update a pull request, complete its associated documentation and verification without requiring separate authorization.
- Do not modify unrelated branches or discard uncommitted work.
- Never merge a pull request unless explicitly instructed.

## Pull Requests

All pull requests must follow `.github/pull_request_template.md`, which is the authoritative source for PR description structure.

### Preparing the PR

- Read the PR template before creating or updating a pull request.
- Populate every required section with substantive, issue-specific information.
- Preserve the template's section headings and checklist structure.
- Do not leave template comments, empty sections, placeholders, or unfilled checklists.
- Do not prepend an informal summary to an unfilled template.
- Keep descriptions concise but sufficiently detailed for a maintainer to understand the change, its motivation, validation, and implications without inspecting the entire diff.

### Required Content

- **Summary:** Explain the principal change and its scope.
- **Motivation / related issue:** Describe the problem being solved and link the relevant issue. Use `Fixes #NNN` when merging the PR should close the issue, or `Refs #NNN` otherwise.
- **Testing:** Report actual validation commands and results, including test counts, supported Python versions, documentation checks, skipped tests, and limitations where applicable.
- **API / scientific impact:** Explicitly describe changes to public behavior, metadata, scientific assumptions, model grids, or calibration data. Cite authoritative scientific sources when applicable. State `None` when there is no relevant impact.
- **Documentation:** Identify documentation changes or explain why none were necessary.
- **Checklist:** Mark items complete only after verification. Explain items that are not applicable or could not be completed.

### Creating and Verifying the PR

- Use `gh pr create --body-file` or `gh pr edit --body-file` with a complete Markdown description rather than appending text to the default template.
- After creating or updating the PR, retrieve its body from GitHub and verify that all required sections are populated correctly.
- Confirm that the PR targets the intended base branch, references the correct issue, and contains no unrelated changes.
- Verify that the branch is reasonably current with the base branch and identify any merge conflicts.
- Report the PR URL, validation results, and any remaining concerns to the user.
- Do not merge the PR without explicit authorization.

## Release Preparation

- Development commits do not require version increments.
- Each newly published release must use a unique version; its wheel and source distribution must share that version.
- Never replace published artifacts with changed builds under the same version.
- Dependency and API changes must be reflected in the code and package metadata of the next published release.
- Increment the prerelease identifier when preparing a new prerelease.
- Ensure the changelog accurately summarizes changes included in the release.
