#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Wrong output type")]
    WrongOutputType,
    #[error("Output is empty")]
    OutputIsEmpty,
    #[error("Input is required")]
    InputRequiredError,
    #[error("{0} not found.")]
    NotFound(String),
    #[error(transparent)]
    SLLMError(#[from] sllm::Error),
    #[error(transparent)]
    JsonParsingError(#[from] serde_json::Error),
}
