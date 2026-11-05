# Environment configuration

Keep all `.env` files and their templates in this directory. Do not create
`.env` files at the repository root. Local settings are Git-ignored; only
`.env.example` and `.env.*.example` templates are tracked.

For generic baseline/router launchers, copy `config/.env.example` to
`config/.env`. Collection launchers use the matching server file, such as
`config/.env.a800`, `config/.env.l20`, `config/.env.l20.thinking`, or
`config/.env.l40.thinking`. Tree-prophetkv launchers use
`config/.env.a800.tree-prophetkv` or `config/.env.l20.tree-prophetkv`.

Existing exported variables override file values. To select a template variant,
copy it to a local file here and set `UCM_ENV_FILE=config/.env.<variant>`.
Relative overrides resolve from the checkout root. See
[server environment syntax](../docs/deployment/SERVER_ENV.md).
