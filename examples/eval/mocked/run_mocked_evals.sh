#!/usr/bin/env sh
# Convenience helper for no-key mocked eval suite checks.
# This file lives next to the mocked eval suites because it is workflow convenience, not framework runtime code.
#
# Usage:
#   sh examples/eval/mocked/run_mocked_evals.sh
#   sh examples/eval/mocked/run_mocked_evals.sh --category state-machine
#   sh examples/eval/mocked/run_mocked_evals.sh --category tools --filter code_search
#   sh examples/eval/mocked/run_mocked_evals.sh --list
#
# Safety:
#   Mocked suites use fixture LLMs and do not call a real provider.
#   No API keys, network access, or provider cost is required.
#   Suites run serially to avoid shared fixture surprises.
set -eu

usage() {
  cat <<'EOF'
Run mocked eval suites for local and CI checks.

Usage:
  sh examples/eval/mocked/run_mocked_evals.sh
  sh examples/eval/mocked/run_mocked_evals.sh --category state-machine

Options:
  --category CATEGORY     Run only one folder under examples/eval/mocked, such as tools or state-machine.
  --filter TEXT           Run only suites whose category/name contains TEXT. Can be combined with --category.
  --output-root PATH      Output root. Default: target/eval/mocked/manual
  --list                  Print matching suite files and exit.
  -h, --help              Show this help.

Examples:
  sh examples/eval/mocked/run_mocked_evals.sh --category tools
  sh examples/eval/mocked/run_mocked_evals.sh --category state-machine --filter two_state
  sh examples/eval/mocked/run_mocked_evals.sh --list
  sh examples/eval/mocked/run_mocked_evals.sh --output-root target/eval/mocked/full
EOF
}

category_filter=""
filter=""
output_root="target/eval/mocked/manual"
list_only="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --category)
      if [ "$#" -lt 2 ]; then
        echo "error: --category requires a value" >&2
        exit 2
      fi
      category_filter="$2"
      shift 2
      ;;
    --filter)
      if [ "$#" -lt 2 ]; then
        echo "error: --filter requires a value" >&2
        exit 2
      fi
      filter="$2"
      shift 2
      ;;
    --output-root)
      if [ "$#" -lt 2 ]; then
        echo "error: --output-root requires a value" >&2
        exit 2
      fi
      output_root="$2"
      shift 2
      ;;
    --list)
      list_only="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

suite_dir="examples/eval/mocked"
if [ ! -d "$suite_dir" ]; then
  echo "error: missing $suite_dir" >&2
  exit 2
fi

found=0
passed=0
failed=0
failed_names=""

for suite in "$suite_dir"/*/*_mocked.yaml; do
  if [ ! -f "$suite" ]; then
    continue
  fi

  category=$(basename "$(dirname "$suite")")
  name=$(basename "$suite" .yaml)
  display_name="$category/$name"
  if [ -n "$category_filter" ] && [ "$category" != "$category_filter" ]; then
    continue
  fi
  if [ -n "$filter" ]; then
    case "$display_name" in
      *"$filter"*) ;;
      *) continue ;;
    esac
  fi

  found=$((found + 1))

  if [ "$list_only" = "true" ]; then
    echo "$suite"
    continue
  fi

  output="$output_root/$category/$name"
  echo "==> $display_name"

  if cargo run -q -p ai-agents-cli -- eval \
    --scenarios "$suite" \
    --output "$output"; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
    failed_names="$failed_names $display_name"
  fi

done

if [ "$found" -eq 0 ]; then
  echo "error: no mocked suites matched" >&2
  if [ -n "$category_filter" ]; then
    echo "selected category: $category_filter" >&2
  fi
  exit 2
fi

if [ "$list_only" = "true" ]; then
  exit 0
fi

echo
printf 'Summary: %s passed, %s failed, %s matched\n' "$passed" "$failed" "$found"

if [ "$failed" -ne 0 ]; then
  echo "Failed suites:$failed_names" >&2
  exit 1
fi
