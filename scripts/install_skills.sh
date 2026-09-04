#!/usr/bin/env bash
# Install the canonical TROVE Skills as symlinks for Agents without Skill Hub.
# Idempotent: rerunning is safe; existing real directories are never
# overwritten. --uninstall removes only the symlinks this script created.
# Compatible with macOS bash 3.2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
SKILLS_DIR="$REPO_ROOT/skills"
TARGET="${HOME:?}/.agents/skills"
UNINSTALL=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_skills.sh [--target DIR] [--uninstall]

Link every canonical TROVE Skill (skills/*/SKILL.md in this repository) into
an agent-visible skills directory, defaulting to ~/.agents/skills.

Options:
  --target DIR   install into DIR instead of ~/.agents/skills
  --uninstall    remove only the symlinks this script created
  -h, --help     show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --target)
      [ $# -ge 2 ] || { echo "install_skills: --target requires a directory" >&2; exit 2; }
      TARGET="$2"
      shift 2
      ;;
    --target=*)
      TARGET="${1#*=}"
      shift
      ;;
    --uninstall)
      UNINSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "install_skills: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

installed=0
skipped=0
refused=0
removed=0
kept=0

[ "$UNINSTALL" -eq 1 ] || mkdir -p "$TARGET"

for skill_md in "$SKILLS_DIR"/*/SKILL.md; do
  [ -f "$skill_md" ] || continue
  name="$(basename "$(dirname "$skill_md")")"
  src="$SKILLS_DIR/$name"
  dst="$TARGET/$name"

  if [ "$UNINSTALL" -eq 1 ]; then
    if [ -L "$dst" ]; then
      case "$(readlink "$dst")" in
        "$SKILLS_DIR"/*)
          rm "$dst"
          echo "removed $name"
          removed=$((removed + 1))
          ;;
        *)
          echo "kept $name (symlink not created by this script)"
          kept=$((kept + 1))
          ;;
      esac
    elif [ -e "$dst" ]; then
      echo "kept $name (not a symlink)"
      kept=$((kept + 1))
    fi
    continue
  fi

  if [ -L "$dst" ]; then
    current="$(readlink "$dst")"
    if [ "$current" = "$src" ]; then
      echo "already installed: $name"
      skipped=$((skipped + 1))
    else
      echo "refusing to replace foreign symlink: $dst -> $current" >&2
      refused=$((refused + 1))
    fi
    continue
  fi
  if [ -e "$dst" ]; then
    echo "refusing to overwrite existing path: $dst" >&2
    refused=$((refused + 1))
    continue
  fi

  ln -s "$src" "$dst"
  echo "installed $name -> $src"
  installed=$((installed + 1))
done

if [ "$UNINSTALL" -eq 1 ]; then
  echo "uninstall: removed=$removed kept=$kept"
else
  echo "install: installed=$installed already=$skipped refused=$refused"
fi

[ "$refused" -eq 0 ]
