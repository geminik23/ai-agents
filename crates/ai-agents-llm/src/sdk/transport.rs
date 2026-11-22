use super::{Error, Value};
use reqwest::{
    Client, Url,
    header::{AUTHORIZATION, HeaderMap, HeaderValue},
};
use std::time::Duration;
#[derive(Clone)]
pub(super) struct Transport {
    client: Client,
    pub(super) endpoint: Url,
    headers: HeaderMap,
}
impl Transport {
    /// The base URL includes the API prefix (for example `https://api.openai.com/v1`).
    pub(super) fn new(base_url: &str, path: &str, headers: HeaderMap) -> Result<Self, Error> {
        let url = Url::parse(&format!("{}/", base_url.trim_end_matches('/')))
            .map_err(|_| Error::Config("invalid base_url".into()))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(Error::Config(
                "expected an HTTP base URL without credentials, query or fragment".into(),
            ));
        }
        let endpoint = url
            .join(path)
            .map_err(|_| Error::Config("invalid base path".into()))?;
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| Error::Transport)?;
        Ok(Self {
            client,
            endpoint,
            headers,
        })
    }

    pub(super) async fn send(
        &self,
        body: &Value,
        timeout: Option<Duration>,
    ) -> Result<reqwest::Response, Error> {
        let mut http = self
            .client
            .post(self.endpoint.clone())
            .headers(self.headers.clone())
            .json(body);
        if let Some(timeout) = timeout {
            http = http.timeout(timeout);
        }
        let response = http.send().await.map_err(|_| Error::Transport)?;
        if !response.status().is_success() {
            return Err(Error::Http(response.status().as_u16()));
        }
        Ok(response)
    }
}

pub(super) fn secret(value: &str) -> Result<HeaderValue, Error> {
    let mut value = HeaderValue::from_str(value)
        .map_err(|_| Error::Config("invalid credential header".into()))?;
    value.set_sensitive(true);
    Ok(value)
}

pub(super) fn bearer_headers(key: Option<String>) -> Result<HeaderMap, Error> {
    let mut headers = HeaderMap::new();
    if let Some(key) = key.filter(|k| !k.is_empty()) {
        headers.insert(AUTHORIZATION, secret(&format!("Bearer {key}"))?);
    }
    Ok(headers)
}

pub(super) fn frames<B, E>(
    mut bytes: impl futures::Stream<Item = Result<B, E>> + Unpin + Send + 'static,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<String, Error>> + Send>>
where
    B: AsRef<[u8]> + Send + 'static,
    E: Send + 'static,
{
    use futures::StreamExt;
    Box::pin(async_stream::try_stream! {
        let mut line = Vec::new();
        let mut data = String::new();
        let mut skip_lf = false;
        while let Some(chunk) = bytes.next().await {
            for &byte in chunk.map_err(|_| Error::Transport)?.as_ref() {
                if std::mem::take(&mut skip_lf) && byte == b'\n' {
                    continue;
                }
                if byte != b'\r' && byte != b'\n' {
                    if data.len() + line.len() >= super::MAX_SSE_FRAME_BYTES {
                        Err(Error::Protocol("SSE frame exceeds the size limit"))?;
                    }
                    line.push(byte);
                    continue;
                }
                skip_lf = byte == b'\r';
                if line.is_empty() {
                    if !data.is_empty() {
                        data.pop(); // Final newline inserted by the SSE data-field algorithm.
                        yield std::mem::take(&mut data);
                    }
                } else {
                    let field = std::str::from_utf8(&line).map_err(|_| Error::Protocol("invalid UTF-8 in stream"))?;
                    if let Some(value) = field.strip_prefix("data:") {
                        data.push_str(value.strip_prefix(' ').unwrap_or(value));
                        data.push('\n');
                    }
                    line.clear();
                }
            }
        }
    })
}
