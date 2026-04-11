use clap::Parser;
use log::info;
use std::fs::File;
use std::io::Write;

// Include the generated proto module
mod tts_service {
    tonic::include_proto!("tts_service");
}

use tts_service::{
    audio_format_options::Audio, synthesizer_client::SynthesizerClient, UtteranceSynthesisRequest,
};

#[derive(Parser, Debug)]
#[command(name = "vosk-tts-client-rs")]
#[command(about = "Vosk TTS Client", long_about = None)]
struct Args {
    /// Server address
    #[arg(short = 'u', long, default_value = "http://localhost:5001")]
    server: String,

    /// Input text
    #[arg(short = 'i', long)]
    text: String,

    /// Output WAV file
    #[arg(short, long, default_value = "out.wav")]
    output: String,

    /// Speaker ID
    #[arg(short, long)]
    speaker: Option<i64>,

    /// Speech rate
    #[arg(short = 'r', long, default_value = "1.0")]
    speech_rate: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    info!("Connecting to server: {}", args.server);
    info!("Synthesizing text: {}", args.text);

    // Create gRPC client
    let mut client = SynthesizerClient::connect(args.server.clone())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to server: {}", e))?;

    let request = tonic::Request::new(UtteranceSynthesisRequest {
        text: args.text.clone(),
        speaker_id: args.speaker.map(|id| id as i32),
        speech_rate: Some(args.speech_rate),
        duration_noise_level: None,
        noise_level: None,
        scale: None,
        hints: None,
    });

    let mut response = client.utterance_synthesis(request).await?.into_inner();

    // Create output WAV file
    let mut output_file = File::create(&args.output)?;

    info!("Receiving audio chunks...");

    while let Some(chunk_result) = response.message().await? {
        if let Some(audio_chunk) = chunk_result.audio_chunk {
            if let Some(audio_options) = audio_chunk.audio {
                if let Some(Audio::RawAudio(raw_audio)) = audio_options.audio {
                    output_file.write_all(&raw_audio.data)?;
                    info!("Received {} bytes", raw_audio.data.len());
                }
            }
        }
    }

    info!("Audio saved to {}", args.output);

    Ok(())
}
