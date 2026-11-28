#!/usr/bin/env python3
"""Run a serial three-trial campaign. Every invocation requires a new output directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
SOURCE = Path("src/lib.rs")


def inventory(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def checker_matches(checker, root, *, expect_failure=False):
    flags = ["--expect-failure"] if expect_failure else []
    return subprocess.run([checker, *flags, root, root / SOURCE], env={}).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real-llm", action="store_true")
    mode.add_argument("--dry-config-check", action="store_true")
    mode.add_argument("--verify-fixture", action="store_true")
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    toolchain = Path(subprocess.check_output(["rustup", "which", "rustc"], text=True).strip()).parent.parent
    (output / "toolchain").symlink_to(toolchain)
    checker = output / "checker.py"
    shutil.copy2(BASE / "checker.py", checker)
    original = inventory(BASE / "fixture")
    count = 3 if args.real_llm else 1
    results = [{"trial": n, "passed": False, "stage": "not_run"} for n in range(1, count + 1)]

    def save_results():
        pending = output / "results.pending"
        pending.write_text(json.dumps(results, indent=2) + "\n")
        pending.replace(output / "results.json")

    if args.real_llm:
        save_results()
    for index in range(1, count + 1):
        root = output / f"trial-{index}"
        shutil.copytree(BASE / "fixture", root)
        command = [binary, "repair-eval", "--dry-config-check", root, checker]
        record = results[index - 1]
        if args.real_llm:
            record["stage"] = "preflight"
            save_results()
        subprocess.run(command, check=True)
        if args.dry_config_check:
            return 0
        if not checker_matches(checker, root, expect_failure=True):
            sys.exit("original fixture did not reproduce a behavioral failure")
        if args.verify_fixture:
            src = root / SOURCE
            for replacement, expect_failure in [
                ("last.1 = last.1.max(end);", False),
                ("if end > last.1 { last.1 = end; }", False),
                ("last.1 = start;", True),
            ]:
                src.write_text(original[SOURCE].decode().replace("last.1 = end;", replacement))
                if not checker_matches(checker, root, expect_failure=expect_failure):
                    sys.exit("independent checker failed fixture control")
            return 0
        command[2:3] = ["--real-llm", "--trust-checker", "--share-source"]
        print(f"Starting trial {index}/{count}. Review each edit and checker request.", flush=True)
        record["stage"] = "live"
        save_results()
        # Inherit stdin for individual pending decisions. Persist only the redacted final result.
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1) as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                if line.startswith("TRIAL "):
                    record.update(json.loads(line.removeprefix("TRIAL ")))
            status = child.wait()
        after = inventory(root)
        protected = after.keys() == original.keys() and all(
            after[path] == data for path, data in original.items() if path != SOURCE)
        record.update(exit_code=status, protected_inputs_intact=protected)
        record["passed"] = bool(record["passed"] and status == 0 and protected
                                and after[SOURCE] != original[SOURCE])
        record["stage"] = "finished"
        save_results()
        if not protected:
            sys.exit("unexpected workspace effects; remaining trials not run")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
