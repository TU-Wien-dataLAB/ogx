# storage

Storage backends for OGX: key-value stores and SQL stores.

## Directory Structure

```text
storage/
  kvstore/             # Key-value store backends
    __init__.py
    config.py          # KVStore config classes
    kvstore.py         # KVStore factory and base implementation
    sqlite/            # SQLite KV backend (aiosqlite)
    redis/             # Redis KV backend
    postgres/          # PostgreSQL KV backend
    mongodb/           # MongoDB KV backend
  sqlstore/            # SQL store backends (SQLAlchemy-based)
    __init__.py
    sqlstore.py        # SqlStore factory and config
    sqlalchemy_sqlstore.py  # SQLAlchemy implementation
    authorized_sqlstore.py  # SqlStore with access control
  __init__.py
  datatypes.py         # Storage config types (StorageBackendType, KVStoreReference, etc.)
```

## KVStore

The `KVStore` interface provides simple key-value operations (`get`, `set`, `delete`, `keys`). Values are strings (typically JSON-serialized). Keys can be namespaced.

Backends: SQLite (default), Redis, PostgreSQL, MongoDB.

Used by: distribution registry, quota middleware, provider state persistence.

## SqlStore

The `SqlStore` interface provides typed table operations with column definitions, filtering, and pagination. Built on SQLAlchemy for portability.

Backends: SQLite (default), PostgreSQL.

Used by: inference store (chat completion logs), conversations, prompts.

### AuthorizedSqlStore

`authorized_sqlstore.py` wraps a `SqlStore` with two enforcement layers applied to every operation:

1. **Tenant isolation** -- when tenancy is enabled (`single` or `multi` mode), a `tenant_id` column is added to every table. Writes stamp the current user's `tenant_id`; all reads and mutations include a non-bypassable `WHERE tenant_id = ?` filter. In `multi` mode, missing tenant context results in default deny (empty results). Client-supplied `tenant_id` in data payloads is stripped and replaced with the authenticated value.

2. **ABAC access control** -- `owner_principal` and `access_attributes` columns support policy-based rules (e.g., `user is owner`). These operate within a tenant boundary.

The tenancy mode is set process-wide during startup via `set_default_tenancy_mode()` in `stack.py`.

## Configuration

Storage is configured in `StackConfig.storage` via `StorageConfig`. The `stores` field contains typed references (`KVStoreReference`, `SqlStoreReference`, `InferenceStoreReference`) that point to specific backend configurations.

See `datatypes.py` for all config types and `StorageBackendType` for the enum of supported backends.

### Optional Stores (null to disable)

Some store references are optional: setting the reference to `null` (not omitting
it) means OGX does not construct the store at all, and the API that depends on
it degrades gracefully. This is an explicit operator choice made in a run config;
the default for every optional reference remains an enabled reference, so
existing deployments are unaffected unless they opt in.

- **`inference`** -- when `null`, no `InferenceStore` is constructed: no
  `inference_store` table is created and no background write workers run. Chat
  completions still work (streaming and non-streaming); the chat completion
  history endpoints (`list`, `retrieve`, `messages`) report that persistence is
  not configured (HTTP 501) rather than returning an empty list or a 404.
- **`responses`** -- the Responses store already follows the same pattern.

Other stores (`datasets`, `eval`, `files`, `prompts`, `vector_io`) are not
affected by disabling the inference store, so persistence can be turned off for
one API independently of the rest of the storage layer.
