# Architecture

These documents describe current responsibilities, boundaries, and contracts.
`Status: accepted` means the design is adopted; verification results are recorded
by tests and CI for each commit. Filenames follow responsibilities; protocol versions
belong in the content and history.

| Document | Responsibility | Formal model |
| --- | --- | --- |
| [Workspace overview](overview.md) | Crate boundaries and dependency direction | — |
| [Session](session.md) | Durable facts, transactions, lifecycle, and runtime ownership | [Session tools](tla/session-tools/README.md) |
| [Desktop](desktop.md) | Projection, timing semantics, interaction, and rendering | — |
| [App storage](app-storage.md) | Product configuration and catalog persistence | — |

The [TLA+ guide](tla/README.md) indexes executable models and explains how to run
or add them. Each model links back to its architecture contract and records its
scope, assumptions, properties, and implementation mapping.

Update the relevant design document and model together when their contracts change.
Keep operational commands in the [development workflow](../development/workflow.md).
