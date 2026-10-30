"""Bounded Unix PTY acceptance with a small cursor-aware screen, using only isolated test fixtures."""
import codecs
import errno
import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
import unicodedata


class Screen:
    def __init__(self):
        self.rows = [[" "] * 120 for _ in range(32)]
        self.row = self.col = 0
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, data):
        self.pending += self.decoder.decode(data)
        while self.pending:
            if self.pending.startswith("\x1b"):
                match = re.match(r"\x1b\[([0-?]*)([ -/]*)([@-~])", self.pending)
                if not match:
                    if len(self.pending) < 2 or self.pending.startswith("\x1b["):
                        break
                    self.pending = self.pending[2:]
                    continue
                args, _, action = match.groups()
                self.pending = self.pending[match.end():]
                values = [int(v) if v.isdigit() else 0 for v in args.split(";")]
                n = values[0] or 1
                if action in "Hf":
                    self.row = max(0, values[0] - 1)
                    self.col = max(0, (values[1] if len(values) > 1 else 1) - 1)
                elif action == "A": self.row = max(0, self.row - n)
                elif action == "B": self.row = min(31, self.row + n)
                elif action == "C": self.col = min(119, self.col + n)
                elif action == "D": self.col = max(0, self.col - n)
                elif action == "G": self.col = max(0, n - 1)
                elif action == "J" and values[0] == 2:
                    self.rows = [[" "] * 120 for _ in range(32)]
                elif action == "K" and self.row < 32:
                    for col in range(min(120, self.col), 120): self.rows[self.row][col] = " "
                continue
            char, self.pending = self.pending[0], self.pending[1:]
            if char == "\r": self.col = 0
            elif char == "\n": self.row = min(31, self.row + 1)
            elif ord(char) >= 32:
                width = 2 if unicodedata.east_asian_width(char) in "WF" else 1
                if self.row < 32 and self.col < 120:
                    self.rows[self.row][self.col] = char
                    if width == 2 and self.col + 1 < 120: self.rows[self.row][self.col + 1] = " "
                self.col += width

    def text(self):
        return "\n".join("".join(row) for row in self.rows)


binary, agent, initial, expected, answer, mode = sys.argv[1:]
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 120, 0, 0))
before = termios.tcgetattr(slave)
env = dict(os.environ, RUST_LOG="off", TASK_FIXTURE_TOKEN="isolated-test-placeholder")


def child_terminal():
    os.setsid()
    fcntl.ioctl(slave, termios.TIOCSCTTY, 0)


process = subprocess.Popen([binary, "run", agent], stdin=slave, stdout=slave, stderr=slave, env=env, preexec_fn=child_terminal)
screen = Screen()
started = stopping = False
answered = set()
received = 0

end = time.monotonic() + 25
try:
    while process.poll() is None and time.monotonic() < end:
        if select.select([master], [], [], 0.02)[0]:
            try:
                data = os.read(master, 65536)
            except OSError as error:
                if error.errno == errno.EIO: break
                raise

            received += len(data)
            if received > 2_000_000: raise AssertionError("unbounded terminal output")
            if b"\x1b[6n" in data: os.write(master, b"\x1b[1;1R")
            screen.feed(data)
        text = screen.text()
        if not started and "Type a message" in text:
            os.write(master, initial.encode("utf-8") + b"\r")
            started = True
        if started and not stopping:
            decision = re.search(r"DECISION (\d+)", text)
            if mode == "cancel" and decision:
                os.write(master, b"\x03")
                stopping = True
            elif mode != "cancel" and all(value in text for value in expected.split("|")):
                os.write(master, b"\x03")
                stopping = True
            elif mode != "cancel" and decision and decision[1] not in answered:
                answered.add(decision[1])
                os.write(master, answer.encode("utf-8") + b"\r")
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"foreground did not return; started={started}, stopping={stopping}, answered={answered}:\n{text}")
    assert started and stopping, "TUI did not reach its actual decision/result:\n" + text
    assert process.returncode == 0, "TUI foreground failed:\n" + text
    after = termios.tcgetattr(slave)
    mask = termios.ICANON | termios.ECHO | termios.ISIG
    assert before[3] & mask == after[3] & mask, "raw terminal mode was not restored"
    print("PTY_OK: foreground outcome and terminal restoration")
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
    os.close(master)
    os.close(slave)
