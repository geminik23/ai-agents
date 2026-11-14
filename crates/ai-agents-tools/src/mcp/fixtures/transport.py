"""Local MCP transport fixture; no provider, package download, or external service."""
import http.server
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import threading
import time

root = Path(os.environ['MCP_FIXTURE_DIR'])
mode = os.environ['MCP_FIXTURE_MODE']
lock = threading.Lock()
# A failed test must not leave an indefinite subprocess behind.
watchdog = threading.Timer(45, lambda: os._exit(2))
watchdog.daemon = True
watchdog.start()


def record(message):
    with lock, (root / 'requests.jsonl').open('a') as output:
        output.write(json.dumps(message) + '\n')


def respond(request):
    record(request)
    if 'id' not in request:
        return None
    method = request['method']
    if method == 'initialize':
        result = {'protocolVersion': request['params']['protocolVersion'],
                  'capabilities': {'tools': {}},
                  'serverInfo': {'name': 'local-fixture', 'version': '1'}}
    elif method == 'tools/list':
        result = {'tools': [{'name': name, 'description': name,
                            'inputSchema': {'type': 'object', 'properties': {}}}
                           for name in ['echo', 'fail', 'blocked', 'approval']]}
    elif method == 'tools/call':
        failed = request['params']['name'] == 'fail'
        result = {'content': [
            {'type': 'text', 'text': 'prefix'},
            {'type': 'resource_link', 'uri': 'file:///fixture', 'name': 'ignored'},
            {'type': 'text', 'text': 'remote-error' if failed else request['params']['arguments']['text']}],
            'isError': failed}
    else:
        return {'jsonrpc': '2.0', 'id': request['id'],
                'error': {'code': -32601, 'message': 'unknown fixture method'}}
    return {'jsonrpc': '2.0', 'id': request['id'], 'result': result}


def ready(**extra):
    temporary = root / 'ready.tmp'
    temporary.write_text(json.dumps({'pid': os.getpid(), **extra}))
    temporary.replace(root / 'ready.json')


if mode.startswith('stdio'):
    ready()
    for line in sys.stdin:
        if mode == 'stdio-stall':
            continue
        response = respond(json.loads(line))
        if response is not None:
            print(json.dumps(response), flush=True)
    (root / 'closed').write_text('stdin-eof')
else:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_error(405)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            record({'headers': dict(self.headers)})
            if mode.startswith('oversize'):
                success = mode == 'oversize-success'
                self.send_response(200 if success else 500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                remaining = (16 * 1024 * 1024 if success else 64 * 1024) + 1
                try:
                    while remaining:
                        size = min(remaining, 65536)
                        self.wfile.write(f'{size:x}\r\n'.encode() + b' ' * size + b'\r\n')
                        remaining -= size
                    self.wfile.flush()
                    # Never terminate the chunked body: rejection must come from the SDK cap.
                    time.sleep(30)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            response = respond(request)
            body = json.dumps(response).encode() if response is not None else b''
            if mode == 'sse' and response is not None:
                body = b'event: message\ndata: ' + body + b'\n\n'
            self.send_response(200 if response is not None else 202)
            self.send_header('Content-Type', 'text/event-stream' if mode == 'sse' else 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    scheme = 'http'
    if mode == 'https':
        # Ephemeral fixture-only identity; no private key is checked into the repository.
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', str(root / 'key.pem'), '-out', str(root / 'cert.pem'),
                        '-days', '1', '-subj', '/CN=localhost',
                        '-addext', 'subjectAltName=DNS:localhost'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(root / 'cert.pem', root / 'key.pem')
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = 'https'
    ready(url=f'{scheme}://localhost:{server.server_port}/mcp')
    server.serve_forever()
