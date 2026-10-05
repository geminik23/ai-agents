#!/bin/sh
set -eu

if ! command -v zola >/dev/null 2>&1; then
  mkdir -p .bin
  if [ ! -x .bin/zola ]; then
    curl -L https://github.com/getzola/zola/releases/download/v0.23.6/zola-v0.23.6-x86_64-unknown-linux-gnu.tar.gz \
      | tar -xz -C .bin
    chmod +x .bin/zola
  fi
  PATH="$PWD/.bin:$PATH"
fi

# Match the CI pin before generating content or parsing version-specific settings.
ZOLA_VERSION="$(zola --version)"
if [ "$ZOLA_VERSION" != "zola 0.23.6" ]; then
  echo "error: website build requires Zola 0.23.6 (CI pin); found $ZOLA_VERSION" >&2
  echo "Install the pinned version: cargo install zola --version 0.23.6 --locked" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTENT_DIR="$SCRIPT_DIR/content/examples"

# Sync examples/README.md into Zola content with frontmatter prepended
echo "Syncing examples/README.md -> content/examples/_index.md"

mkdir -p "$CONTENT_DIR"

cat > "$CONTENT_DIR/_index.md" << 'FRONTMATTER'
+++
title = "Examples"
template = "examples.html"
description = "Run YAML agent examples on the ai-agents Rust runtime: conversations, local models, tools, memory, state workflows, and multi-agent routing."
sort_by = "weight"
[extra]
seo_title = "YAML Agent Examples for Rust"
+++

FRONTMATTER

tail -n +3 "$PROJECT_ROOT/examples/README.md" >> "$CONTENT_DIR/_index.md"

# Curated learning pages embed the repository YAML rather than maintaining copies.
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: website build requires Python 3.11 or newer for example pages" >&2
  exit 1
fi
python3 "$SCRIPT_DIR/scripts/generate_examples.py"
python3 "$SCRIPT_DIR/scripts/capture_execution_demo.py" --check
python3 "$SCRIPT_DIR/scripts/home_agent_demos.py"

echo "Done. Building site..."

cd "$SCRIPT_DIR"

CMD="${1:-build}"
shift 2>/dev/null || true

case "$CMD" in
  serve)
    exec zola serve "$@"
    ;;
  build)
    zola build "$@"
    echo "Site built → $SCRIPT_DIR/public/"
    ;;
  *)
    echo "Usage: $0 [build|serve] [extra zola flags]"
    exit 1
    ;;
esac
