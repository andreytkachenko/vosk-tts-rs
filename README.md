# Vosk TTS Rust

[![CI](https://github.com/andreytkachenko/vosk-tts-rs/workflows/CI/badge.svg)](https://github.com/andreytkachenko/vosk-tts-rs/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org/)

Rust version of Vosk TTS - Offline Text-to-Speech System based on VITS architecture with ONNX Runtime inference.

## Features

- **Fully Offline**: No cloud dependencies, runs entirely locally
- **ONNX Runtime**: Uses `ort` crate for efficient inference
- **Multiple Speakers**: Supports multi-speaker models
- **gRPC Server**: Production-ready streaming server
- **CLI Tool**: Simple command-line interface
- **Russian Language**: Pre-configured for Russian TTS

## Project Structure

```
vosk-tts-rs/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── main.rs             # CLI binary
│   ├── model.rs            # Model loading and management
│   ├── synth.rs            # Audio synthesis engine
│   ├── g2p.rs              # Grapheme-to-phoneme conversion
│   └── bin/
│       ├── server.rs       # gRPC server binary
│       └── client.rs       # gRPC client binary
├── proto/
│   └── tts_service.proto   # gRPC service definition
├── build.rs                # Proto compilation
└── Cargo.toml
```

## Installation

### Prerequisites

- Rust 1.70+ (edition 2021)
- ONNX Runtime libraries (downloaded automatically)

### Build from Source

```bash
cd vosk-tts-rs
cargo build --release
```

This produces three binaries:
- `vosk-tts-rs` - CLI tool
- `vosk-tts-server-rs` - gRPC server
- `vosk-tts-client-rs` - gRPC client

## Usage

### CLI Tool

```bash
# Synthesize text
./target/release/vosk-tts-rs -i "Привет мир" -o output.wav

# Specify model
./target/release/vosk-tts-rs -m /path/to/model -i "Текст" -o out.wav

# Set speech rate
./target/release/vosk-tts-rs -i "Текст" -r 1.2 -o out.wav

# List available models
./target/release/vosk-tts-rs --list-models

# List languages
./target/release/vosk-tts-rs --list-languages
```

### gRPC Server

```bash
# Start server with default settings (port 5001)
export VOSK_MODEL_PATH=/path/to/vosk-model-tts-ru-0.9-multi
./target/release/vosk-tts-server-rs

# Custom port and interface
export VOSK_SERVER_INTERFACE=127.0.0.1
export VOSK_SERVER_PORT=8080
./target/release/vosk-tts-server-rs
```

### gRPC Client

```bash
# Connect to server and synthesize
./target/release/vosk-tts-client-rs \
  --server http://localhost:5001 \
  --text "Привет мир" \
  --output output.wav

# With speaker selection
./target/release/vosk-tts-client-rs \
  --text "Текст" \
  --speaker 2 \
  --speech-rate 1.1 \
  --output out.wav
```

## Model Download

Download TTS models from: https://alphacephei.com/vosk/models/

Example:
```bash
wget https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.9-multi.zip
unzip vosk-model-tts-ru-0.9-multi.zip
```

## Architecture

### Core Components

1. **Model** (`model.rs`): 
   - Loads ONNX models and dictionaries
   - Handles model downloads
   - Manages BERT tokenizer for prosody

2. **Synth** (`synth.rs`):
   - Main TTS pipeline
   - G2P conversion with multiple strategies
   - ONNX inference
   - WAV file generation

3. **G2P** (`g2p.rs`):
   - Russian grapheme-to-phoneme conversion
   - Stress mark handling
   - Palatalization support

4. **Server** (`server.rs`):
   - gRPC streaming service
   - Chunked audio delivery
   - Multi-threaded processing

### Supported Model Types

- `multistream_v1/v2/v3` - Multistream VITS models with punctuation context
- Models with/without BERT embeddings
- Models with/without blank token interspersion

## Development

### Run Tests

```bash
cargo test
```

### Lint Code

```bash
cargo clippy
```

### Format Code

```bash
cargo fmt
```

## Differences from Python Version

### Implemented
- ✅ Model loading and downloading
- ✅ Dictionary parsing
- ✅ G2P conversion (Russian)
- ✅ WAV file generation
- ✅ gRPC server/client
- ✅ CLI interface
- ✅ BERT tokenizer integration
- ✅ Full ONNX inference with proper tensor handling
- ✅ All multistream G2P variants with embeddings

### Simplified (TODO)
- ⏳ Complete prosody modeling

### Rust-Specific Improvements
- **Type Safety**: Strong typing throughout
- **Memory Safety**: No undefined behavior
- **Concurrency**: Thread-safe design
- **Performance**: Optimized release builds (~33MB binary)
- **Error Handling**: `anyhow` for flexible errors

## Dependencies

- `ort` - ONNX Runtime bindings
- `hound` - WAV file I/O
- `regex` - Text processing
- `serde` - JSON parsing
- `reqwest` - HTTP downloads
- `tonic` - gRPC framework
- `clap` - CLI parsing
- `log`/`env_logger` - Logging

## License

Same as Vosk TTS (Apache 2.0)

## Contributing

Contributions welcome! Please submit issues and pull requests.
