use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to load ONNX model: {0}")]
    OnnxModelLoad(#[source] ort::Error),

    #[error("Failed to read dictionary from {path}: {source}")]
    DictionaryRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse config.json: {0}")]
    ConfigParse(#[source] serde_json::Error),

    #[error("Failed to read config from {path}: {source}")]
    ConfigRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Model name {0} does not exist")]
    ModelNotFound(String),

    #[error("Language {0} does not exist")]
    LanguageNotFound(String),

    #[error("Failed to build WordPiece tokenizer: {0}")]
    TokenizerBuild(String),

    #[error("Failed to extract audio tensor: {0}")]
    AudioTensorExtract(String),

    #[error("HTTP request failed: {0}")]
    HttpRequest(#[source] reqwest::Error),

    #[error("Failed to extract zip archive: {0}")]
    ZipExtract(#[source] zip::result::ZipError),

    #[error("Failed to connect to server: {0}")]
    ConnectionFailed(String),

    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ONNX runtime error: {0}")]
    OnnxRuntime(#[from] ort::Error),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("Audio error: {0}")]
    Audio(#[from] hound::Error),

    #[error("Zip error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Parse error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("Environment variable error: {0}")]
    EnvVar(#[from] std::env::VarError),

    #[error("gRPC error: {0}")]
    Grpc(Box<tonic::Status>),

    #[error("Server error: {0}")]
    Server(#[from] tonic::transport::Error),

    #[error("Invalid address: {0}")]
    AddrParse(#[from] std::net::AddrParseError),
}

impl From<tonic::Status> for Error {
    fn from(v: tonic::Status) -> Self {
        Self::Grpc(Box::new(v))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
