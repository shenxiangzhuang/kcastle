# Project documentation

Project-specific knowledge is organized by purpose:

## Architecture

Long-lived responsibilities, boundaries, and invariants.

- [Architecture index](architecture/README.md)
- [Workspace overview](architecture/overview.md)
- [Session](architecture/session.md)
- [Desktop](architecture/desktop.md)
- [Desktop app storage](architecture/app-storage.md)
- [TLA+ models and verification](architecture/tla/README.md)

## Development

Commands and operational workflows that may change with the repository.

- [Development workflow](development/workflow.md)
- [Release workflow](development/release.md)

Keep architectural contracts and their executable models in `architecture/`, and
operational workflows in `development/`. Update the relevant index when adding a
document; do not create a new category for a single speculative document.
