use log::info;
use std::env;
use std::sync::{Arc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use vosk_tts_rs::{Model, Synth};

// gRPC service module - this would be generated from .proto file
pub mod tts_service {
    tonic::include_proto!("tts_service");
}

use tts_service::{
    synthesizer_server::{Synthesizer, SynthesizerServer},
    AudioChunk, AudioFormatOptions, UtteranceSynthesisRequest, UtteranceSynthesisResponse,
};

pub struct SynthesizerServicer {
    model: Arc<Mutex<Model>>,
    synth: Synth,
}

impl SynthesizerServicer {
    pub fn new(model_path: Option<&str>) -> anyhow::Result<Self> {
        info!(
            "Loading TTS model from {:?}",
            model_path.unwrap_or("default")
        );
        let model = Model::new(model_path, None, Some("ru"))?;
        let synth = Synth::new();
        Ok(SynthesizerServicer {
            model: Arc::new(Mutex::new(model)),
            synth,
        })
    }
}

#[tonic::async_trait]
impl Synthesizer for SynthesizerServicer {
    type UtteranceSynthesisStream = ReceiverStream<Result<UtteranceSynthesisResponse, Status>>;

    async fn utterance_synthesis(
        &self,
        request: Request<UtteranceSynthesisRequest>,
    ) -> Result<Response<Self::UtteranceSynthesisStream>, Status> {
        let req = request.into_inner();
        let text = req.text;
        let speaker_id = req.speaker_id;
        let speech_rate = req.speech_rate.unwrap_or(1.0);

        info!("Synthesizing text: {}", text);

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let synth = self.synth.clone();
        let model = Arc::clone(&self.model);

        // Clone data for closure
        let text_clone = text.clone();

        // Spawn synthesis in background
        tokio::task::spawn_blocking(move || {
            // Create temporary file for chunked audio
            let temp_output = format!("/tmp/tts_chunk_{}.wav", std::process::id());

            let mut model_guard = model.lock().unwrap();

            match synth.synth(
                &mut model_guard,
                &text_clone,
                &temp_output,
                speaker_id.map(|id| id as i64),
                None,
                Some(speech_rate),
                None,
                None,
            ) {
                Ok(_) => {
                    // Read the WAV file and send as chunks
                    if let Ok(audio_data) = std::fs::read(&temp_output) {
                        let chunk_size = 4096;
                        for chunk in audio_data.chunks(chunk_size) {
                            // Use the oneof enum variant correctly
                            let audio_options = AudioFormatOptions {
                                audio: Some(tts_service::audio_format_options::Audio::RawAudio(
                                    tts_service::RawAudio {
                                        data: chunk.to_vec(),
                                        sample_rate_hertz: 22050,
                                    },
                                )),
                            };

                            let response = UtteranceSynthesisResponse {
                                audio_chunk: Some(AudioChunk {
                                    audio: Some(audio_options),
                                }),
                            };

                            if tx.blocking_send(Ok(response)).is_err() {
                                break;
                            }
                        }

                        // Clean up temp file
                        let _ = std::fs::remove_file(&temp_output);
                    }
                }
                Err(e) => {
                    let _ =
                        tx.blocking_send(Err(Status::internal(format!("Synthesis error: {}", e))));
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

pub async fn serve() -> anyhow::Result<()> {
    let interface = env::var("VOSK_SERVER_INTERFACE").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = env::var("VOSK_SERVER_PORT")
        .unwrap_or_else(|_| "5001".to_string())
        .parse()?;
    let model_path = env::var("VOSK_MODEL_PATH").ok();
    let threads: usize = env::var("VOSK_SERVER_THREADS")
        .unwrap_or_else(|_| num_cpus::get().to_string())
        .parse()?;

    let addr = format!("{}:{}", interface, port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))?;

    info!(
        "Starting TTS server on {}:{} with {} threads",
        interface, port, threads
    );

    let service = SynthesizerServicer::new(model_path.as_deref())?;

    Server::builder()
        .add_service(SynthesizerServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { serve().await })
}
