# Desktop app storage

Status: accepted

## Decision

The desktop owns one app-level SQLite WAL database at `<data-root>/app.sqlite3`. It stores the
project registry, UI preferences, provider and model catalogs, credentials, model preferences, and
new-session defaults. These values belong to the desktop product and never enter the agent core's
session schema.

Every project has a deterministic data directory at `<data-root>/projects/<project-id>`. Its agent
history remains in `sessions/sessions.sqlite3`, whose schema and transaction semantics are owned
exclusively by `kcastle-agent`. The built-in Default project uses the reserved ID `default` and the
same directory shape as every other project.

Session configuration is copied into the session journal when a session is created or changed.
The app database therefore supplies defaults; the project session database remains the durable
record of the configuration actually used. Future project-specific defaults should be keyed by
`project_id` in the desktop database unless they become part of the agent runtime's canonical
session semantics.

## Layout

```text
<data-root>/
├── app.sqlite3
└── projects/
    ├── default/
    │   └── sessions/
    │       └── sessions.sqlite3
    └── <project-id>/
        └── sessions/
            └── sessions.sqlite3
```

Project storage paths are derived from validated stable IDs rather than persisted independently.
Workspace relocation changes the external workspace path without changing the project ID or its
session directory.

The app database, WAL, and shared-memory sidecars use mode `0600` on Unix because provider API keys
remain stored as application data. Moving credentials to the operating-system credential store is
a separate security boundary and does not change this database ownership model.
