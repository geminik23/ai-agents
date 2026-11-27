"""Minimal controllable stdio peer for protocol and lifecycle acceptance."""
import json
import os
from pathlib import Path
import sys
import time

mode, control = sys.argv[1:]
control = Path(control)
(control / "pid").write_text(str(os.getpid()))

def send(value):
    body = json.dumps({"jsonrpc": "2.0", **value}).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()

while line := sys.stdin.buffer.readline():
    length = int(line.split(b":")[1])
    assert sys.stdin.buffer.readline() == b"\r\n"
    message = json.loads(sys.stdin.buffer.read(length))
    method = message["method"]
    if method == "exit":
        break
    if "id" not in message:
        continue
    result = None
    if method == "initialize":
        result = {"capabilities": {"diagnosticProvider": {"interFileDependencies": False}}}
    if method == "textDocument/diagnostic":
        if mode == "crash":
            sys.exit(73)
        (control / "collecting").touch()
        if mode == "wait":
            while not (control / "release").exists():
                time.sleep(0.01)
        send({"id": message["id"] - 1, "result": {"kind": "full", "items": []}})
        result = {"kind": "unchanged"} if mode == "partial" else {"kind": "full", "items": []}
    send({"id": message["id"], "result": result})
