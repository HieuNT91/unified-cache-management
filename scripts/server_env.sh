#!/usr/bin/env bash
# Trusted local Bash assignments, one per line. Existing variables win.
ucm_load_server_env() {
    local _ucm_env_root
    _ucm_env_root="$(cd -- "$1" && pwd)" || return 1
    local _ucm_env_file="${UCM_ENV_FILE:-$_ucm_env_root/.env}"
    local _ucm_env_line _ucm_env_key _ucm_env_value _ucm_env_number=0
    if [[ "$_ucm_env_file" != /* ]]; then
        _ucm_env_file="$_ucm_env_root/$_ucm_env_file"
    fi
    if [[ ! -f "$_ucm_env_file" ]]; then
        if [[ -n "${UCM_ENV_FILE:-}" ]]; then
            echo "Missing UCM_ENV_FILE: $_ucm_env_file" >&2
            return 1
        fi
        return 0
    fi
    while IFS= read -r _ucm_env_line || [[ -n "$_ucm_env_line" ]]; do
        _ucm_env_number=$((_ucm_env_number + 1))
        _ucm_env_line="${_ucm_env_line%$'\r'}"
        [[ "$_ucm_env_line" =~ ^[[:space:]]*(#|$) ]] && continue
        if [[ "$_ucm_env_line" =~ ^[[:space:]]*(export[[:space:]]+)?([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$ ]]; then
            _ucm_env_key="${BASH_REMATCH[2]}"
            _ucm_env_value="${BASH_REMATCH[3]}"
            if [[ ! -v "$_ucm_env_key" ]]; then
                # Like sourcing a shell config, RHS expansions are intentional.
                # Never print values: local files may contain credentials.
                eval "export $_ucm_env_key=$_ucm_env_value" || return 1
            fi
        else
            echo "Expected KEY=VALUE at $_ucm_env_file:$_ucm_env_number" >&2
            return 1
        fi
    done < "$_ucm_env_file"
}
