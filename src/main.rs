use anyhow::Result;
use clap::Parser;
use log::info;
use std::fs;
use vosk_tts_rs::{list_languages, list_models, Model, Synth};

#[derive(Parser, Debug)]
#[command(name = "vosk-tts-rs")]
#[command(about = "Vosk TTS - Synthesize input", long_about = None)]
struct Args {
    /// Model path
    #[arg(short, long)]
    model: Option<String>,

    /// List available models
    #[arg(long)]
    list_models: bool,

    /// List available languages
    #[arg(long)]
    list_languages: bool,

    /// Select model by name
    #[arg(short = 'n', long)]
    model_name: Option<String>,

    /// Select model by language
    #[arg(short, long, default_value = "en-us")]
    lang: String,

    /// Input text string or file path
    #[arg(short, long)]
    input: Option<String>,

    /// Speaker ID for multispeaker model
    #[arg(short, long)]
    speaker: Option<i64>,

    /// Speech rate of the synthesis
    #[arg(short = 'r', long, default_value = "1.0")]
    speech_rate: f32,

    /// Output WAV filename
    #[arg(short, long, default_value = "out.wav")]
    output: String,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    if args.list_models {
        list_models()?;
        return Ok(());
    }

    if args.list_languages {
        list_languages()?;
        return Ok(());
    }

    let input_text = match args.input {
        Some(input) => {
            // Check if it's a file path
            if fs::metadata(&input).is_ok() {
                fs::read_to_string(&input)?
            } else {
                input
            }
        }
        None => {
            info!("Please specify input text or file");
            std::process::exit(1);
        }
    };

    info!("Loading model...");
    let model = Model::new(
        args.model.as_deref(),
        args.model_name.as_deref(),
        Some(&args.lang),
    )?;

    info!("Creating synthesizer...");
    let synth = Synth::new();

    info!("Synthesizing...");
    let mut model = model;
    synth.synth(
        &mut model,
        &input_text,
        &args.output,
        args.speaker,
        None,
        Some(args.speech_rate),
        None,
        None,
    )?;

    info!("Output written to {}", args.output);

    Ok(())
}
