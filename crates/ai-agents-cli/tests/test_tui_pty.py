"""Check PTY ownership and restoration detection independently of the TUI implementation."""
import os
import subprocess
import sys
import termios
import unittest

from tui_pty import assert_terminal_restored, terminal_session


class TerminalSessionTests(unittest.TestCase):
    def test_foreground_child_exit_preserves_terminal_and_restored_modes(self):
        with terminal_session() as (_, slave, child_terminal):
            before = termios.tcgetattr(slave)
            subprocess.run(
                [sys.executable, "-c", """
import os, signal, sys, termios, tty
assert os.getsid(0) == int(sys.argv[1])
assert os.getpgrp() == os.getpid()
assert os.tcgetpgrp(0) == os.getpgrp()
assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
assert signal.getsignal(signal.SIGTTOU) == signal.SIG_DFL
with open('/dev/tty', 'rb') as terminal:
    assert os.isatty(terminal.fileno())
before = termios.tcgetattr(0)
tty.setraw(0)
termios.tcsetattr(0, termios.TCSANOW, before)
""", str(os.getpid())],
                stdin=slave, stdout=slave, stderr=slave,
                preexec_fn=child_terminal, check=True, timeout=5,
            )
            assert_terminal_restored(before, slave)

    def test_each_missing_local_mode_restoration_is_rejected(self):
        for flag in (termios.ICANON, termios.ECHO, termios.ISIG):
            with self.subTest(flag=flag):
                with terminal_session() as (_, slave, child_terminal):
                    before = termios.tcgetattr(slave)
                    self.assertTrue(before[3] & flag)
                    subprocess.run(
                        [sys.executable, "-c", """
import sys, termios
attributes = termios.tcgetattr(0)
attributes[3] &= ~int(sys.argv[1])
termios.tcsetattr(0, termios.TCSANOW, attributes)
""", str(flag)],
                        stdin=slave, stdout=slave, stderr=slave,
                        preexec_fn=child_terminal, check=True, timeout=5,
                    )
                    with self.assertRaisesRegex(AssertionError, "raw terminal mode was not restored"):
                        assert_terminal_restored(before, slave)


if __name__ == "__main__":
    unittest.main()
