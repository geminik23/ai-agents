#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTENT_DIR="$SCRIPT_DIR/content/examples"

# Sync examples/README.md into Zola content with frontmatter prepended
echo "Syncing examples/README.md -> content/examples/_index.md"

mkdir -p "$CONTENT_DIR"

cat > "$CONTENT_DIR/_index.md" << 'FRONTMATTER'
+++
title = "Examples"
template = "section.html"
description = "Browse example agents — from simple chatbots to complex workflows."
sort_by = "weight"
+++

FRONTMATTER

tail -n +3 "$PROJECT_ROOT/examples/README.md" >> "$CONTENT_DIR/_index.md"

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
