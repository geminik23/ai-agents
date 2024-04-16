use std::string::FromUtf8Error;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error(transparent)]
    Utf8Error(#[from] FromUtf8Error),

    #[error("Request error {0}")]
    RequestError(String),
    #[error(transparent)]
    TeraError(#[from] tera::Error),
    #[error(transparent)]
    JSONParsingError(#[from] serde_json::Error), // #[error(transparent)]
                                                 // RequestError(#[from] ureq::Error),
}
