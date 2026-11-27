//! Bounded stdio framing for one language server session.

use serde_json::{Value, json};
use std::{io, process::Stdio};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};

const HEADER_LIMIT: usize = 4096;
const MESSAGE_LIMIT: usize = 8 * 1024 * 1024;

pub(super) struct Session {
    pub(super) child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next_id: u64,
}

pub(super) fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl Session {
    pub(super) fn start(command: &mut Command) -> io::Result<Self> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| invalid("missing LSP stdin"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| invalid("missing LSP stdout"))?;
        Ok(Self {
            child,
            input,
            output: BufReader::new(output),
            next_id: 0,
        })
    }

    async fn send(&mut self, value: Value) -> io::Result<()> {
        let body = serde_json::to_vec(&value)?;
        if body.len() > MESSAGE_LIMIT {
            return Err(invalid("LSP request exceeds message limit"));
        }
        self.input
            .write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
            .await?;
        self.input.write_all(&body).await?;
        self.input.flush().await
    }

    async fn receive(&mut self) -> io::Result<Value> {
        // Cap the total header before growing a buffer, including headers without a line ending.
        let mut header = Vec::new();
        let mut length = None;
        let mut remaining = HEADER_LIMIT as u64;
        loop {
            header.clear();
            let count = (&mut self.output)
                .take(remaining)
                .read_until(b'\n', &mut header)
                .await?;
            remaining -= count as u64;
            if count == 0 || !header.ends_with(b"\r\n") {
                return Err(invalid("invalid or incomplete LSP header"));
            }
            if header == b"\r\n" {
                break;
            }
            let line = std::str::from_utf8(&header).map_err(|_| invalid("invalid LSP header"))?;
            let (key, value) = line
                .split_once(':')
                .ok_or_else(|| invalid("invalid LSP header"))?;
            if key.eq_ignore_ascii_case("Content-Length") {
                let value = value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| invalid("invalid LSP length"))?;
                if length.replace(value).is_some() {
                    return Err(invalid("duplicate LSP content length"));
                }
            }
        }
        let length = length
            .filter(|length| *length <= MESSAGE_LIMIT)
            .ok_or_else(|| invalid("missing or oversized LSP content length"))?;
        let mut buffer = vec![0; length];
        self.output.read_exact(&mut buffer).await?;
        let message: Value = serde_json::from_slice(&buffer)?;
        if message["jsonrpc"] != "2.0" {
            return Err(invalid("invalid LSP response"));
        }
        Ok(message)
    }

    pub(super) async fn notify(&mut self, method: &str, params: Value) -> io::Result<()> {
        self.send(json!({"jsonrpc":"2.0", "method":method, "params":params}))
            .await
    }

    // Server-initiated requests have their own ID space and cannot grant authority to modify files.
    async fn next(&mut self) -> io::Result<Value> {
        loop {
            let value = self.receive().await?;
            if let (Some(method), Some(id)) = (value["method"].as_str(), value.get("id")) {
                if method == "workspace/configuration" {
                    let count = value["params"]["items"].as_array().map_or(0, Vec::len);
                    self.send(json!({"jsonrpc":"2.0", "id":id, "result":vec![Value::Null; count]}))
                        .await?;
                } else {
                    self.send(json!({"jsonrpc":"2.0", "id":id, "error":{"code":-32601,"message":"Unsupported request"}})).await?;
                }
            } else {
                return Ok(value);
            }
        }
    }

    pub(super) async fn request(&mut self, method: &str, params: Value) -> io::Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}))
            .await?;
        loop {
            let value = self.next().await?;
            if value["id"].as_u64() == Some(id) {
                if value.get("error").is_some() {
                    return Err(invalid("language server rejected the request"));
                }
                return value
                    .get("result")
                    .cloned()
                    .ok_or_else(|| invalid("missing LSP result"));
            }
        }
    }

    pub(super) async fn notification(&mut self, method: &str) -> io::Result<Value> {
        loop {
            let value = self.next().await?;
            if value["method"] == method {
                return Ok(value["params"].clone());
            }
        }
    }
}
