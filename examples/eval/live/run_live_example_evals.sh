#!/usr/bin/env sh
# Convenience helper for intentional live example eval checks.
# This file lives next to the live eval suites because it is workflow convenience, not framework runtime code.
#
# Usage:
#   sh examples/eval/live/run_live_example_evals.sh --dry-config-check
#   sh examples/eval/live/run_live_example_evals.sh --dry-config-check --category tools
#   sh examples/eval/live/run_live_example_evals.sh --yes-live --category state-machine
#   sh examples/eval/live/run_live_example_evals.sh --yes-live --category tools --filter code_search
#   sh examples/eval/live/run_live_example_evals.sh --list
#
# Safety:
#   --dry-config-check parses suite config only and does not call a provider.
#   --yes-live runs real provider calls, requires credentials, and may incur cost.
#   Suites run serially to avoid shared fixture surprises.
set -eu

usage() {
  cat <<'EOF'
Run live example eval suites for local maintainer checks.

Usage:
  sh examples/eval/live/run_live_example_evals.sh --dry-config-check
  sh examples/eval/live/run_live_example_evals.sh --yes-live

Options:
  --dry-config-check       Validate every live suite and referenced agent without running scenarios or constructing providers.
  --yes-live              Run real live suites with --real-llm. Requires provider keys and may incur cost.
  --category CATEGORY     Run only one folder under examples/eval/live/examples, such as tools or state-machine.
  --filter TEXT           Run only suites whose category/name contains TEXT. Can be combined with --category.
  --output-root PATH      Output root. Default: target/eval/live/examples/manual
  --list                  Print matching suite files and exit.
  -h, --help              Show this help.

Examples:
  sh examples/eval/live/run_live_example_evals.sh --dry-config-check --category tools
  sh examples/eval/live/run_live_example_evals.sh --yes-live --category state-machine
  sh examples/eval/live/run_live_example_evals.sh --yes-live --category tools --filter code_search
  sh examples/eval/live/run_live_example_evals.sh --yes-live --output-root target/eval/live/examples/full
EOF
}

mode=""
category_filter=""
filter=""
output_root="target/eval/live/examples/manual"
list_only="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-config-check)
      mode="dry"
      shift
      ;;
    --yes-live)
      mode="live"
      shift
      ;;
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

if [ "$list_only" = "false" ] && [ -z "$mode" ]; then
  echo "error: choose --dry-config-check or --yes-live" >&2
  usage >&2
  exit 2
fi

if [ "$mode" = "live" ]; then
  cat <<'EOF'
WARNING: This will run live provider evals.

Requirements:
  - Provider credentials such as OPENAI_API_KEY must be configured.
  - Network access is required.
  - Provider calls may incur cost.
  - Suites run serially on purpose.
EOF
fi

suite_dir="examples/eval/live/examples"
if [ ! -d "$suite_dir" ]; then
  echo "error: missing $suite_dir" >&2
  exit 2
fi

found=0
passed=0
failed=0
failed_names=""

for suite in "$suite_dir"/*/*_live.yaml; do
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

  if [ "$mode" = "dry" ]; then
    if cargo run -q -p ai-agents-cli -- eval \
      --scenarios "$suite" \
      --dry-config-check; then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      failed_names="$failed_names $display_name"
    fi
  else
    if cargo run -q -p ai-agents-cli -- eval \
      --scenarios "$suite" \
      --output "$output" \
      --real-llm; then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      failed_names="$failed_names $display_name"
    fi
  fi

done

if [ "$found" -eq 0 ]; then
  echo "error: no live suites matched" >&2
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
