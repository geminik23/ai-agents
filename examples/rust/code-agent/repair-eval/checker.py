#!/usr/bin/python3 -I
"""Run the protected behavioral tests with read-only inputs, no network or host home."""
from pathlib import Path
import subprocess
import sys

root, source = map(Path, sys.argv[1:])
if root / "src/lib.rs" != source:
    sys.exit("unexpected checker arguments")
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
    sys.exit(0)
if result.returncode != 0 and failed and passed | failed == expected:
    # Distinct from toolchain errors, missing tests and checker setup failures.
    sys.exit(1)
sys.exit(2)
