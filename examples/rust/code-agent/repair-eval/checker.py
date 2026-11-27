#!/usr/bin/python3 -I
"""Run the protected behavioral tests with read-only inputs, no network or host home."""
from pathlib import Path
import subprocess
import sys


def check(args):
    # In baseline mode, only positive evidence of a failing suite may admit a paid trial.
    expect_failure = args[:1] == ["--expect-failure"]
    root, source = map(Path, args[1:] if expect_failure else args)
    if root / "src/lib.rs" != source:
        raise ValueError("unexpected checker arguments")
    # The campaign pins a real installed toolchain outside the writable project.
    toolchain = Path(__file__).with_name("toolchain").resolve(strict=True)
    command = [
        "/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/etc/alternatives", "/etc/alternatives", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(toolchain), "/toolchain", "--ro-bind", str(root), "/project",
        "--chdir", "/project", "--setenv", "PATH", "/toolchain/bin:/usr/bin",
        "--setenv", "HOME", "/tmp", "--setenv", "CARGO_TARGET_DIR", "/tmp/target",
        "/toolchain/bin/cargo", "test", "--locked", "--offline", "--test", "acceptance",
        "--", "--test-threads=1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="")
    expected = {"empty_and_single", "contained_then_overlapping", "adjacent_and_separate",
                "permutation_preserves_union"}
    lines = result.stdout.splitlines()
    passed = {name for name in expected if f"test {name} ... ok" in lines}
    failed = {name for name in expected if f"test {name} ... FAILED" in lines}
    if result.returncode == 0 and passed == expected:
        if expect_failure:
            print("expected a behavioral failure but all named tests passed", file=sys.stderr)
            return 2
        return 0
    complete = passed.isdisjoint(failed) and passed | failed == expected
    if result.returncode == 101 and failed and complete:
        # Cargo reports a failing test target with 101. Require the outcome of every named test.
        return 0 if expect_failure else 1
    print("checker did not obtain a complete test result", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(check(sys.argv[1:]))
    except (OSError, ValueError, RuntimeError) as error:
        # In normal validation, exit 1 is reserved for a verified behavioral failure.
        print(f"checker setup failed: {error}", file=sys.stderr)
        sys.exit(2)
