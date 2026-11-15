"""Local transport readiness must not depend on the machine's hostname resolver."""
import ipaddress
import json
import os
from pathlib import Path
import runpy
import socket
import socketserver
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlsplit


class TransportStartupTests(unittest.TestCase):
    def test_http_readiness_uses_the_bound_loopback_without_dns(self):
        for mode in ('http', 'sse', 'oversize-success', 'oversize-error'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                listening = []

                def observe_listener(server):
                    # Bind, listen and ready publication are real; serving is covered by the SDK tests.
                    listening.append(server.socket.getsockname())

                with mock.patch.dict(os.environ, {'MCP_FIXTURE_DIR': directory, 'MCP_FIXTURE_MODE': mode}), \
                     mock.patch.object(socket, 'getfqdn', side_effect=AssertionError('unexpected DNS lookup')), \
                     mock.patch.object(socket, 'gethostbyaddr', side_effect=AssertionError('unexpected DNS lookup')), \
                     mock.patch.object(socket, 'getaddrinfo', side_effect=AssertionError('unexpected DNS lookup')), \
                     mock.patch.object(socketserver.BaseServer, 'serve_forever', observe_listener):
                    namespace = runpy.run_path(str(Path(__file__).with_name('transport.py')))
                try:
                    ready = json.loads((Path(directory) / 'ready.json').read_text())
                    url = urlsplit(ready['url'])
                    self.assertEqual(len(listening), 1)
                    host, port = listening[0]
                    self.assertTrue(ipaddress.ip_address(host).is_loopback)
                    self.assertEqual(url.scheme, 'http')
                    self.assertEqual((url.hostname, url.port), (host, port))
                    self.assertEqual(ready['pid'], os.getpid())
                finally:
                    namespace['server'].server_close()
                    namespace['watchdog'].cancel()


if __name__ == '__main__':
    unittest.main()
