use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, BufReader},
    net::{TcpListener, TcpStream},
};

pub(super) async fn receive_path(listener: &TcpListener, path: &str) -> (TcpStream, Value, String) {
    let (mut socket, _) = listener.accept().await.unwrap();
    let mut reader = BufReader::new(&mut socket);
    let mut head = String::new();
    let mut len = None;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).await.unwrap() > 0);
        if line == "\r\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && k.eq_ignore_ascii_case("content-length")
        {
            len = Some(v.trim().parse().unwrap());
        }
        head.push_str(&line);
    }
    assert!(head.starts_with(&format!("POST {path} HTTP/1.1\r\n")));
    let mut body = vec![0; len.unwrap()];
    reader.read_exact(&mut body).await.unwrap();
    (
        socket,
        serde_json::from_slice(&body).unwrap(),
        head.to_ascii_lowercase(),
    )
}
