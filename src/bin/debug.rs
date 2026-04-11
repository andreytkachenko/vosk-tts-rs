use std::collections::HashMap;
use std::fs;

use vosk_tts_rs::{Model, Synth};

fn dump_array(name: &str, data: &[f32], shape: &[usize]) {
    let total: usize = shape.iter().product();
    let n_show = 50.min(data.len());
    let vals_str: Vec<String> = data[..n_show].iter().map(|x| format!("{:.6}", x)).collect();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    println!("  {}: shape={:?}, dtype=float32", name, shape);
    println!("    values[{}]: [{}]", n_show, vals_str.join(", "));
    if data.len() > n_show {
        println!("    ... ({} more values)", data.len() - n_show);
    }
    println!("    min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
}

fn dump_array_i64(name: &str, data: &[i64], shape: &[usize]) {
    let n_show = 50.min(data.len());
    let vals_str: Vec<String> = data[..n_show].iter().map(|x| format!("{}", x)).collect();
    let min = data.iter().cloned().fold(i64::MAX, i64::min);
    let max = data.iter().cloned().fold(i64::MIN, i64::max);
    println!("  {}: shape={:?}, dtype=int64", name, shape);
    println!("    values[{}]: [{}]", n_show, vals_str.join(", "));
    if data.len() > n_show {
        println!("    ... ({} more values)", data.len() - n_show);
    }
    println!("    min={}, max={}", min, max);
}

fn main() {
    env_logger::init();

    // Test regex first
    let re = regex::Regex::new(r#"(\.\.\.|- |[ ,.?!;:"()])"#).unwrap();
    let text = "привет мир";
    eprintln!("=== REGEX TEST ===");
    eprintln!("Text: {:?}", text);
    eprintln!("Regex: {}", re);
    for (i, part) in re.split(text).enumerate() {
        eprintln!("  split[{}] = {:?}", i, part);
    }
    eprintln!("=== END REGEX TEST ===");

    let model_path = std::env::var("VOSK_MODEL_PATH").expect("VOSK_MODEL_PATH not set");
    let test_text = "привет мир";

    println!("================================================================================");
    println!("RUST TTS TENSOR DEBUG");
    println!("================================================================================");
    println!("Test text: {}", test_text);
    println!("Model path: {}", model_path);

    let mut model = Model::new(Some(&model_path), None, Some("ru")).unwrap();

    println!("Model type: {:?}", model.config.model_type);
    println!("Has tokenizer: {}", model.tokenizer.is_some());
    println!("No blank: {:?}", model.config.no_blank);
    println!("Sample rate: {}", model.config.audio.sample_rate);
    println!("Num speakers: {}", model.config.num_speakers);

    // Show phoneme_id_map
    println!("\nPhoneme ID map (first 30 entries):");
    for (i, (k, v)) in model.config.phoneme_id_map.iter().take(30).enumerate() {
        println!("  '{}' -> {:?}", k, v);
    }

    let synth = Synth::new();

    let noise_level = model.config.inference.noise_level;
    let speech_rate = model.config.inference.speech_rate;
    let duration_noise_level = model.config.inference.duration_noise_level;
    let scale = model.config.inference.scale;

    println!(
        "\nParameters: noise_level={}, speech_rate={}",
        noise_level, speech_rate
    );
    println!(
        "  duration_noise_level={}, scale={}, speaker_id=0",
        duration_noise_level, scale
    );

    // Manually run G2P and dump tensors
    let text = test_text.trim().replace('—', "-");
    let model_type = model.config.model_type.as_deref().unwrap_or("");
    let has_tokenizer = model.tokenizer.is_some();
    let no_blank = model.config.no_blank.unwrap_or(0);

    println!("\n--- G2P Processing ---");
    println!(
        "Model type: {}, has_tokenizer: {}, no_blank: {}",
        model_type, has_tokenizer, no_blank
    );

    // Run G2P
    let g2p_result = match (model_type, has_tokenizer, no_blank) {
        ("multistream_v3", true, _) => synth.g2p_multistream_scales(&model, &text).unwrap(),
        ("multistream_v2", true, _) => synth.g2p_multistream(&model, &text, true).unwrap(),
        ("multistream_v2", false, _) => synth.g2p_multistream(&model, &text, true).unwrap(),
        ("multistream_v1", true, _) => synth.g2p_multistream(&model, &text, false).unwrap(),
        ("multistream_v1", false, _) => synth.g2p_multistream(&model, &text, false).unwrap(),
        (_, true, nb) if nb != 0 => synth.g2p_noblank(&model, &text).unwrap(),
        (_, true, 0) => synth.g2p_with_embeddings(&model, &text).unwrap(),
        _ => synth.g2p_no_embeddings(&model, &text).unwrap(),
    };

    println!("\n--- Final Input Tensors ---");
    dump_array_i64(
        "input (text)",
        &g2p_result.text_data,
        &g2p_result.text_shape,
    );
    dump_array_i64("input_lengths", &g2p_result.text_lengths, &[1]);

    let scales = vec![noise_level, 1.0 / speech_rate, duration_noise_level];
    dump_array("scales", &scales, &[3]);
    dump_array_i64("sid", &[0], &[1]);
    dump_array("bert", &g2p_result.bert_data, &g2p_result.bert_shape);
    if let Some(ref dur) = g2p_result.duration_extra_data {
        if let Some(ref dur_shape) = g2p_result.duration_extra_shape {
            dump_array("phone_duration_extra", dur, dur_shape);
        }
    }

    // Save to JSON for comparison
    println!("\n--- Saving to JSON ---");
    let mut dump = serde_json::Map::new();
    dump.insert("text".to_string(), serde_json::Value::String(text.clone()));
    dump.insert(
        "model_type".to_string(),
        serde_json::Value::String(model_type.to_string()),
    );
    dump.insert(
        "has_tokenizer".to_string(),
        serde_json::Value::Bool(has_tokenizer),
    );
    dump.insert(
        "no_blank".to_string(),
        serde_json::Value::Number(no_blank.into()),
    );
    dump.insert("input".to_string(), serde_json::json!(g2p_result.text_data));
    dump.insert(
        "input_shape".to_string(),
        serde_json::json!(g2p_result.text_shape),
    );
    dump.insert(
        "input_lengths".to_string(),
        serde_json::json!(g2p_result.text_lengths),
    );
    dump.insert("scales".to_string(), serde_json::json!(scales));
    dump.insert("sid".to_string(), serde_json::json!(vec![0i64]));
    dump.insert(
        "bert_shape".to_string(),
        serde_json::json!(g2p_result.bert_shape),
    );
    dump.insert(
        "bert_flat".to_string(),
        serde_json::json!(g2p_result.bert_data),
    );
    if let Some(ref dur) = g2p_result.duration_extra_data {
        dump.insert("duration_extra".to_string(), serde_json::json!(dur));
        dump.insert(
            "duration_extra_shape".to_string(),
            serde_json::json!(g2p_result.duration_extra_shape),
        );
    }

    // Run inference
    println!("\n--- Running Inference ---");
    let audio = synth
        .synth_audio(
            &mut model,
            &text,
            Some(0),
            Some(noise_level),
            Some(speech_rate),
            Some(duration_noise_level),
            Some(scale),
        )
        .unwrap();

    let audio_f32: Vec<f32> = audio.iter().map(|&x| x as f32 / 32767.0).collect();
    println!("Output audio shape: [{}]", audio.len());
    let n_show = 50.min(audio_f32.len());
    let vals_str: Vec<String> = audio_f32[..n_show]
        .iter()
        .map(|x| format!("{}", x))
        .collect();
    println!("Output first {}: [{}]", n_show, vals_str.join(", "));
    let min = audio
        .iter()
        .cloned()
        .map(|x| x as f32)
        .fold(f32::INFINITY, f32::min);
    let max = audio
        .iter()
        .cloned()
        .map(|x| x as f32)
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "Output min={}, max={}, total samples={}",
        min,
        max,
        audio.len()
    );

    dump.insert(
        "output_shape".to_string(),
        serde_json::json!(vec![audio.len()]),
    );
    dump.insert(
        "output_flat_first50".to_string(),
        serde_json::json!(&audio_f32[..n_show.min(audio_f32.len())]),
    );

    let out_path = "/tmp/rust_tensors.json";
    fs::write(out_path, serde_json::to_string_pretty(&dump).unwrap()).unwrap();
    println!("Saved to {}", out_path);
}
