#!/usr/bin/env bash
set -euo pipefail

default_install_extras() {
  local system_name="${1:-$(uname -s)}"
  if [[ "$system_name" == "Darwin" ]]; then
    printf '%s' "local-vision,local-embedding,zvec"
  else
    printf '%s' ""
  fi
}

resolve_install_extras() {
  local system_name="${1:-$(uname -s)}"
  if [[ "${TROVE_RUNTIME_INSTALL_EXTRAS+x}" == "x" ]]; then
    printf '%s' "$TROVE_RUNTIME_INSTALL_EXTRAS"
  else
    default_install_extras "$system_name"
  fi
}

if [[ "${TROVE_BOOTSTRAP_SOURCE_ONLY:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${TROVE_VENV_DIR:-$ROOT/.venv}"
BOOTSTRAP_PYTHON="${TROVE_BOOTSTRAP_PYTHON:-python3}"
INSTALL_EXTRAS="$(resolve_install_extras)"
CHECK_ONLY=0
NO_INSTALL=0
QUIET=0

usage() {
  cat <<USAGE
Usage: bash scripts/bootstrap_runtime.sh [--check] [--no-install] [--quiet]

Creates or verifies the repository-local Python runtime at .venv.
System Python is used only for the bootstrap step that creates .venv; all TROVE
product commands should run through ./scripts/trove-python afterwards.

Environment:
  TROVE_BOOTSTRAP_PYTHON       Python used only to create .venv (default: python3)
  TROVE_VENV_DIR               Override venv path (default: <repo>/.venv)
  TROVE_RUNTIME_INSTALL_EXTRAS Optional extras to install (default on macOS: local-vision,local-embedding,zvec; empty elsewhere)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --no-install) NO_INSTALL=1 ;;
    --quiet) QUIET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() {
  if [[ "$QUIET" != "1" ]]; then
    echo "$@"
  fi
}

PY="$VENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
  if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "missing project runtime: $PY" >&2
    exit 127
  fi
  log "Creating TROVE project runtime: $VENV_DIR"
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

if [[ "$CHECK_ONLY" != "1" && "$NO_INSTALL" != "1" ]]; then
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    log "Installing pip into TROVE project runtime"
    "$PY" -m ensurepip --upgrade
  fi
  log "Installing/updating TROVE runtime packages in .venv"
  "$PY" -m pip install --upgrade pip setuptools wheel
  if [[ -n "$INSTALL_EXTRAS" ]]; then
    "$PY" -m pip install -e "$ROOT[$INSTALL_EXTRAS]"
  else
    "$PY" -m pip install -e "$ROOT"
  fi
fi

"$ROOT/scripts/trove-python" scripts/runtime_doctor.py --json
