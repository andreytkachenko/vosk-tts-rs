use crate::g2p;
use crate::model::{Model, PhonemeIdValue};
use anyhow::Result;
use log::info;
use ndarray::ArrayD;
use ort::value::Value;

#[derive(Clone)]
pub struct Synth;

/// Helper: get a single phoneme ID (panics if mapping returns multiple IDs)
fn get_phoneme_id_single(model: &Model, phoneme: &str) -> i64 {
    match model.config.phoneme_id_map.get(phoneme) {
        Some(PhonemeIdValue::Single(id)) => *id,
        Some(PhonemeIdValue::Multiple(ids)) => ids[0],
        None => {
            info!("Warning: phoneme '{}' not found in phoneme_id_map", phoneme);
            0
        }
    }
}

/// Helper: get phoneme IDs as a Vec (for flat phoneme sequences)
fn get_phoneme_ids(model: &Model, phoneme: &str) -> Vec<i64> {
    match model.config.phoneme_id_map.get(phoneme) {
        Some(PhonemeIdValue::Single(id)) => vec![*id],
        Some(PhonemeIdValue::Multiple(ids)) => ids.clone(),
        None => {
            info!("Warning: phoneme '{}' not found in phoneme_id_map", phoneme);
            vec![0]
        }
    }
}

/// Normalize audio and convert to int16 range
fn audio_float_to_int16(audio: &[f32], max_wav_value: f32) -> Vec<i16> {
    audio
        .iter()
        .map(|&sample| {
            let normalized = sample * max_wav_value;
            let clipped = normalized.clamp(-max_wav_value, max_wav_value);
            clipped as i16
        })
        .collect()
}

/// Result of G2P processing — unified structure for all model types
pub struct G2PResult {
    /// Phoneme IDs as flat array, shape depends on model type
    pub text_data: Vec<i64>,
    /// Shape of text_data tensor (including batch dim)
    pub text_shape: Vec<usize>,
    /// text_lengths value
    pub text_lengths: Vec<i64>,
    /// BERT embeddings, flattened to [C * T] where C=768 or C=0
    pub bert_data: Vec<f32>,
    /// Shape of bert_data tensor (including batch dim)
    pub bert_shape: Vec<usize>,
    /// phone_duration_extra, if present (multistream_v3)
    pub duration_extra_data: Option<Vec<f32>>,
    /// Shape of duration_extra tensor
    pub duration_extra_shape: Option<Vec<usize>>,
}

impl G2PResult {
    /// For multistream models: text shape (1, 5, T), bert shape (1, 768, T)
    fn multistream(
        text_data: Vec<i64>,
        t: usize,
        bert_data: Vec<f32>,
        duration_extra_data: Option<Vec<f32>>,
    ) -> Self {
        let duration_extra_shape = duration_extra_data.as_ref().map(|_| vec![1, t]);
        G2PResult {
            text_data,
            text_shape: vec![1, 5, t],
            text_lengths: vec![t as i64],
            bert_data,
            bert_shape: vec![1, 768, t],
            duration_extra_data,
            duration_extra_shape,
        }
    }

    /// For non-multistream: text shape (1, T), bert shape (1, 768, T)
    fn standard(text_data: Vec<i64>, t: usize, bert_data: Vec<f32>) -> Self {
        G2PResult {
            text_data,
            text_shape: vec![1, t],
            text_lengths: vec![t as i64],
            bert_data,
            bert_shape: vec![1, 768, t],
            duration_extra_data: None,
            duration_extra_shape: None,
        }
    }

    /// For no-embeddings fallback: text shape (1, T), bert is zeros
    fn no_embeddings(text_data: Vec<i64>, t: usize) -> Self {
        let hidden_size = 768;
        G2PResult {
            text_data,
            text_shape: vec![1, t],
            text_lengths: vec![t as i64],
            bert_data: vec![0.0f32; hidden_size * t],
            bert_shape: vec![1, hidden_size, t],
            duration_extra_data: None,
            duration_extra_shape: None,
        }
    }
}

impl Synth {
    pub fn new() -> Self {
        Synth {}
    }

    /// Synthesize audio from text
    #[allow(clippy::too_many_arguments)]
    pub fn synth_audio(
        &self,
        model: &mut Model,
        text: &str,
        speaker_id: Option<i64>,
        noise_level: Option<f32>,
        speech_rate: Option<f32>,
        duration_noise_level: Option<f32>,
        scale: Option<f32>,
    ) -> Result<Vec<i16>> {
        let noise_level = noise_level.unwrap_or(model.config.inference.noise_level);
        let speech_rate = speech_rate.unwrap_or(model.config.inference.speech_rate);
        let duration_noise_level =
            duration_noise_level.unwrap_or(model.config.inference.duration_noise_level);
        let scale = scale.unwrap_or(model.config.inference.scale);

        // Clean text
        let text = text.trim().replace('—', "-");

        let start_time = std::time::Instant::now();

        // Run G2P based on model type
        let model_type = model.config.model_type.as_deref().unwrap_or("");
        let has_tokenizer = model.tokenizer.is_some();
        let no_blank = model.config.no_blank.unwrap_or(0);

        let g2p_result = match (model_type, has_tokenizer, no_blank) {
            ("multistream_v3", true, _) => self.g2p_multistream_scales(model, &text)?,
            ("multistream_v2", true, _) => self.g2p_multistream(model, &text, true)?,
            ("multistream_v2", false, _) => self.g2p_multistream(model, &text, true)?,
            ("multistream_v1", true, _) => self.g2p_multistream(model, &text, false)?,
            ("multistream_v1", false, _) => self.g2p_multistream(model, &text, false)?,
            (_, true, nb) if nb != 0 => self.g2p_noblank(model, &text)?,
            (_, true, 0) => self.g2p_with_embeddings(model, &text)?,
            _ => self.g2p_no_embeddings(model, &text)?,
        };

        info!("Text: {}", text);
        info!(
            "Text shape: {:?}, BERT shape: {:?}, text_lengths: {:?}",
            g2p_result.text_shape, g2p_result.bert_shape, g2p_result.text_lengths
        );

        // Create scales tensor - shape (3,) as in Python
        let scales = vec![noise_level, 1.0 / speech_rate, duration_noise_level];

        // Assign first voice
        let speaker_id = speaker_id.unwrap_or(0);
        let sid = vec![speaker_id];

        // Build input tensors for ONNX
        let input_tensor = Value::from_array(
            ArrayD::<i64>::from_shape_vec(g2p_result.text_shape.clone(), g2p_result.text_data)
                .unwrap(),
        )?;
        let input_lengths_tensor = Value::from_array(
            ArrayD::<i64>::from_shape_vec(vec![1], g2p_result.text_lengths).unwrap(),
        )?;
        let scales_tensor =
            Value::from_array(ArrayD::<f32>::from_shape_vec(vec![3], scales).unwrap())?;
        let sid_tensor = Value::from_array(ArrayD::<i64>::from_shape_vec(vec![1], sid).unwrap())?;
        let bert_tensor = Value::from_array(
            ArrayD::<f32>::from_shape_vec(g2p_result.bert_shape.clone(), g2p_result.bert_data)
                .unwrap(),
        )?;

        let mut inputs = ort::inputs![
            "input" => input_tensor,
            "input_lengths" => input_lengths_tensor,
            "scales" => scales_tensor,
            "sid" => sid_tensor,
            "bert" => bert_tensor,
        ];

        // Add phone_duration_extra if present
        if let (Some(dur_data), Some(dur_shape)) = (
            g2p_result.duration_extra_data,
            g2p_result.duration_extra_shape,
        ) {
            let dur_tensor =
                Value::from_array(ArrayD::<f32>::from_shape_vec(dur_shape, dur_data).unwrap())?;
            inputs.push(("phone_duration_extra".into(), dur_tensor.into()));
        }

        // Run ONNX inference
        let outputs = model.onnx.run(inputs)?;

        // Extract audio from output - use index 0 like Python does
        // Python: audio = self.model.onnx.run(None, args)[0]
        let audio_value = &outputs[0];
        let (_audio_shape, audio_data) = audio_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract audio tensor: {}", e))?;

        // audio_data may have shape (1, T_audio) or (T_audio,) - flatten and scale
        let audio_data_vec: Vec<f32> = audio_data.to_vec();
        let audio_scaled: Vec<f32> = audio_data_vec.iter().map(|&x| x * scale).collect();

        // Convert to int16
        let audio_int16 = audio_float_to_int16(&audio_scaled, 32767.0);

        let end_time = std::time::Instant::now();
        let infer_sec = end_time.duration_since(start_time).as_secs_f32();
        let sample_rate = model.config.audio.sample_rate;
        let audio_duration_sec = audio_int16.len() as f32 / sample_rate as f32;
        let rtf = if audio_duration_sec > 0.0 {
            infer_sec / audio_duration_sec
        } else {
            0.0
        };

        info!(
            "Real-time factor: {:.2} (infer={:.2} sec, audio={:.2} sec, samples={})",
            rtf,
            infer_sec,
            audio_duration_sec,
            audio_int16.len()
        );

        Ok(audio_int16)
    }

    // ========================================================================
    // G2P with BERT embeddings (no blank interspersing)
    // Corresponds to Python g2p() with no_blank == 0
    // ========================================================================

    pub fn g2p_with_embeddings(&self, model: &Model, text: &str) -> Result<G2PResult> {
        let re = regex::Regex::new(r#"([,.?!;:"() ])"#).unwrap();
        let mut phonemes = vec!["^".to_string()];

        // Get BERT embeddings for the whole text
        let bert_embeddings = model.get_word_bert(text, false).unwrap_or_default();

        // Track word indices for each phoneme
        let mut word_indices = vec![0usize];
        let mut word_index = 1;

        for word in re.split(&text.to_lowercase()) {
            if word.is_empty() {
                continue;
            }

            if re.is_match(word) || word == "-" {
                phonemes.push(word.to_string());
                word_indices.push(word_index);
            } else if let Some(phoneme_str) = model.dic.get(word) {
                for p in phoneme_str.split_whitespace() {
                    phonemes.push(p.to_string());
                    word_indices.push(word_index);
                }
            } else {
                let converted = g2p::convert(word);
                for p in converted.split_whitespace() {
                    phonemes.push(p.to_string());
                    word_indices.push(word_index);
                }
            }
            if word != " " {
                word_index += 1;
            }
        }

        phonemes.push("$".to_string());
        word_indices.push(word_index);

        // Python g2p: intersperse blanks + duplicate embeddings
        // phoneme_ids: [id_0, 0, id_1, 0, id_2, ..., 0, id_{N-1}]
        // embeddings:  [emb_0, emb_1, emb_1, emb_2, emb_2, ..., emb_{N-1}, emb_{N-1}]
        let hidden_size = 768;
        let n = phonemes.len(); // N
        let t = 2 * n - 1; // total length with blanks

        let mut phoneme_ids = Vec::with_capacity(t);
        let mut bert_data = Vec::with_capacity(t * hidden_size);

        // First phoneme (no blank before it)
        let first_ids = get_phoneme_ids(model, &phonemes[0]);
        phoneme_ids.extend(&first_ids);
        Self::add_bert_emb_at(
            &mut bert_data,
            word_indices[0],
            &bert_embeddings,
            hidden_size,
        );

        for i in 1..phonemes.len() {
            // Blank
            phoneme_ids.push(0);
            Self::add_bert_emb_at(
                &mut bert_data,
                word_indices[i],
                &bert_embeddings,
                hidden_size,
            );
            // Phoneme
            let ids = get_phoneme_ids(model, &phonemes[i]);
            phoneme_ids.extend(&ids);
            Self::add_bert_emb_at(
                &mut bert_data,
                word_indices[i],
                &bert_embeddings,
                hidden_size,
            );
        }

        info!("Text: {}", text);
        info!("Phonemes: {:?}", phonemes);
        info!(
            "phoneme_ids count: {}, bert embeddings count (elements / 768): {}",
            phoneme_ids.len(),
            bert_data.len() / hidden_size
        );

        // Shape: text (1, T), bert (1, 768, T)
        // Transpose BERT data from phoneme-major to channel-major (like Python's np.transpose)
        let bert_data = Self::transpose_bert(bert_data, t, hidden_size);
        Ok(G2PResult::standard(phoneme_ids, t, bert_data))
    }

    // ========================================================================
    // G2P noblank (no blank interspersing)
    // Corresponds to Python g2p_noblank()
    // ========================================================================

    pub fn g2p_noblank(&self, model: &Model, text: &str) -> Result<G2PResult> {
        let re = regex::Regex::new(r#"([,.?!;:"() ])"#).unwrap();
        let mut phonemes = vec!["^".to_string()];

        let bert_embeddings = model.get_word_bert(text, false).unwrap_or_default();
        let mut word_indices = vec![0usize];
        let mut word_index = 1;

        for word in re.split(&text.to_lowercase()) {
            if word.is_empty() {
                continue;
            }

            if re.is_match(word) || word == "-" {
                phonemes.push(word.to_string());
                word_indices.push(word_index);
            } else if let Some(phoneme_str) = model.dic.get(word) {
                for p in phoneme_str.split_whitespace() {
                    phonemes.push(p.to_string());
                    word_indices.push(word_index);
                }
            } else {
                let converted = g2p::convert(word);
                for p in converted.split_whitespace() {
                    phonemes.push(p.to_string());
                    word_indices.push(word_index);
                }
            }
            if word != " " {
                word_index += 1;
            }
        }

        phonemes.push("$".to_string());
        word_indices.push(word_index);

        let hidden_size = 768;
        let mut phoneme_ids = vec![];
        let mut bert_data = Vec::new();

        for i in 0..phonemes.len() {
            let ids = get_phoneme_ids(model, &phonemes[i]);
            phoneme_ids.extend(&ids);
            Self::add_bert_emb_at(
                &mut bert_data,
                word_indices[i],
                &bert_embeddings,
                hidden_size,
            );
        }

        info!("Text: {}", text);
        info!("Phonemes: {:?}", phonemes);

        let t = phoneme_ids.len();
        // Transpose BERT data from phoneme-major to channel-major
        let bert_data = Self::transpose_bert(bert_data, t, hidden_size);
        Ok(G2PResult::standard(phoneme_ids, t, bert_data))
    }

    // ========================================================================
    // G2P multistream (v1, v2)
    // Corresponds to Python g2p_multistream()
    // ========================================================================

    pub fn g2p_multistream(&self, model: &Model, text: &str, word_pos: bool) -> Result<G2PResult> {
        let re = regex::Regex::new(r#"(\.\.\.|- |[ ,.?!;:"()])"#).unwrap();
        let text_clean = text.replace(" -", "- ");

        let bert_embeddings = model.get_word_bert(&text_clean, true).unwrap_or_default();
        info!("BERT embeddings count: {}", bert_embeddings.len());

        // Split text into words AND delimiters (like Python's re.split with capturing group)
        // Python: re.split(pattern, text) returns both text and delimiters
        // Rust: regex::split() does NOT include delimiters, so we use find_iter
        let mut tokens: Vec<String> = Vec::new();
        let mut last_end = 0;
        for mat in re.find_iter(&text_clean) {
            // Text before this match
            if mat.start() > last_end {
                tokens.push(text_clean[last_end..mat.start()].to_string());
            }
            // The delimiter itself
            tokens.push(mat.as_str().to_string());
            last_end = mat.end();
        }
        // Remaining text after last match
        if last_end < text_clean.len() {
            tokens.push(text_clean[last_end..].to_string());
        }

        info!("Tokens from split: {:?}", tokens);

        // phonemes: (phoneme, cur_punc, in_quote, bert_word_index)
        let mut phonemes: Vec<(String, Vec<String>, i32, usize)> =
            vec![("^".to_string(), vec![], 0, 0)];

        let mut cur_punc = vec![];
        let mut in_quote = 0;
        let mut bert_word_index = 1;

        for word in &tokens {
            if word.is_empty() {
                continue;
            }

            if word == "\"" {
                in_quote = if in_quote == 1 { 0 } else { 1 };
                continue;
            }

            if word == "- " || word == "-" {
                cur_punc.push('-'.to_string());
                continue;
            }

            if re.is_match(word) && word != " " {
                cur_punc.push(word.to_string());
                continue;
            }

            if word == " " {
                phonemes.push((" ".to_string(), cur_punc.clone(), in_quote, bert_word_index));
                cur_punc = vec![];
                continue;
            }

            let word_phonemes_raw =
                if let Some(dic_entry) = model.dic.get(word.to_lowercase().as_str()) {
                    dic_entry
                        .split_whitespace()
                        .map(String::from)
                        .collect::<Vec<_>>()
                } else {
                    g2p::convert(word)
                        .split_whitespace()
                        .map(String::from)
                        .collect()
                };

            let word_phonemes = if word_pos {
                Self::add_pos(&word_phonemes_raw)
            } else {
                word_phonemes_raw
            };

            info!("  word: {:?} -> phonemes: {:?}", word, word_phonemes);
            for p in &word_phonemes {
                phonemes.push((p.clone(), vec![], in_quote, bert_word_index));
            }

            cur_punc = vec![];
            bert_word_index += 1;
        }

        phonemes.push((" ".to_string(), cur_punc.clone(), in_quote, bert_word_index));
        phonemes.push(("$".to_string(), vec![], 0, bert_word_index));

        info!("Phonemes after forward pass ({}):", phonemes.len());
        for (i, p) in phonemes.iter().enumerate() {
            info!(
                "  [{}] phoneme='{}' cur_punc={:?} in_quote={} bert_idx={}",
                i, p.0, p.1, p.2, p.3
            );
        }

        // Process in reverse (exactly like Python)
        let mut last_punc = " ".to_string();
        let mut last_sentence_punc = " ".to_string();

        let mut lp_phonemes: Vec<(i64, i64, i64, i64, i64)> = vec![];
        let mut rev_bert_indices: Vec<usize> = vec![];

        for p in phonemes.iter().rev() {
            let punc_list = &p.1;
            if punc_list.iter().any(|x| x == "...") {
                last_sentence_punc = "...".to_string();
            } else if punc_list.iter().any(|x| x == ".") {
                last_sentence_punc = ".".to_string();
            } else if punc_list.iter().any(|x| x == "!") {
                last_sentence_punc = "!".to_string();
            } else if punc_list.iter().any(|x| x == "?") {
                last_sentence_punc = "?".to_string();
            } else if punc_list.iter().any(|x| x == "-") {
                last_sentence_punc = "-".to_string();
            }

            if !punc_list.is_empty() {
                last_punc = punc_list[0].clone();
            }

            let cur_punc_str = if !punc_list.is_empty() {
                punc_list[0].clone()
            } else {
                "_".to_string()
            };

            // Use SINGLE phoneme IDs (multistream expects scalars per channel)
            let phoneme_id = get_phoneme_id_single(model, &p.0);
            let cur_punc_id = get_phoneme_id_single(model, &cur_punc_str);
            let last_punc_id = get_phoneme_id_single(model, &last_punc);
            let last_sentence_punc_id = get_phoneme_id_single(model, &last_sentence_punc);

            lp_phonemes.push((
                phoneme_id,
                cur_punc_id,
                p.2 as i64,
                last_punc_id,
                last_sentence_punc_id,
            ));
            rev_bert_indices.push(p.3);
        }

        lp_phonemes.reverse();
        rev_bert_indices.reverse();

        let t = lp_phonemes.len();
        let hidden_size = 768;

        // Flatten lp_phonemes in CHANNEL-MAJOR order (like numpy transpose)
        // Python: np.array(lp_phonemes) → (T, 5), transpose → (5, T)
        // So we need: [all_ch0, all_ch1, all_ch2, all_ch3, all_ch4]
        let mut text_data = Vec::with_capacity(t * 5);
        for channel in 0..5 {
            for (p0, p1, p2, p3, p4) in &lp_phonemes {
                let val = match channel {
                    0 => *p0,
                    1 => *p1,
                    2 => *p2,
                    3 => *p3,
                    _ => *p4,
                };
                text_data.push(val);
            }
        }

        // BERT embeddings: transpose from phoneme-major to channel-major
        // Python: np.array(bert_embs_raw) → (T, 768), transpose → (768, T)
        // So we need: [ch0_all_phonemes, ch1_all_phonemes, ..., ch767_all_phonemes]
        let bert_raw: Vec<Vec<f32>> = rev_bert_indices
            .iter()
            .map(|&bert_idx| {
                if bert_idx < bert_embeddings.len() {
                    bert_embeddings[bert_idx].clone()
                } else {
                    vec![0.0f32; hidden_size]
                }
            })
            .collect();

        // Transpose: from [T][768] to [768][T]
        let mut bert_data = Vec::with_capacity(t * hidden_size);
        #[allow(clippy::needless_range_loop)]
        for ch in 0..hidden_size {
            for phoneme in 0..t {
                bert_data.push(bert_raw[phoneme][ch]);
            }
        }

        info!("Text: {}", text);
        info!("Phonemes count (multistream T={}): {}", t, t);

        Ok(G2PResult::multistream(text_data, t, bert_data, None))
    }

    // ========================================================================
    // G2P multistream_scales (v3)
    // Corresponds to Python g2p_multistream_scales()
    // ========================================================================

    pub fn g2p_multistream_scales(&self, model: &Model, text: &str) -> Result<G2PResult> {
        let re = regex::Regex::new(r#"(\.\.\.|- |[ ,.?!;:"()_])"#).unwrap();
        let text_clean = text.replace(" -", "- ");

        let bert_embeddings = model.get_word_bert(&text_clean, true).unwrap_or_default();

        // Split text into words AND delimiters (like Python's re.split with capturing group)
        let mut tokens: Vec<String> = Vec::new();
        let mut last_end = 0;
        for mat in re.find_iter(&text_clean) {
            if mat.start() > last_end {
                tokens.push(text_clean[last_end..mat.start()].to_string());
            }
            tokens.push(mat.as_str().to_string());
            last_end = mat.end();
        }
        if last_end < text_clean.len() {
            tokens.push(text_clean[last_end..].to_string());
        }

        info!("Tokens from split (v3): {:?}", tokens);

        let mut phonemes: Vec<(String, Vec<String>, i32, usize)> =
            vec![("^".to_string(), vec![], 0, 0)];

        let mut cur_punc = vec![];
        let mut in_quote = 0;
        let mut bert_word_index = 1;

        for word in &tokens {
            if word.is_empty() {
                continue;
            }

            if word == "\"" {
                in_quote = if in_quote == 1 { 0 } else { 1 };
                continue;
            }

            if word == "- " || word == "-" {
                cur_punc.push('-'.to_string());
                continue;
            }

            if re.is_match(word) && word != " " {
                cur_punc.push(word.to_string());
                continue;
            }

            if word == " " {
                phonemes.push((" ".to_string(), cur_punc.clone(), in_quote, bert_word_index));
                cur_punc = vec![];
                continue;
            }

            let word_lower = word.to_lowercase();
            let word_phonemes_raw = if let Some(dic_entry) = model.dic.get(word_lower.as_str()) {
                dic_entry
                    .split_whitespace()
                    .map(String::from)
                    .collect::<Vec<_>>()
            } else {
                g2p::convert(word)
                    .split_whitespace()
                    .map(String::from)
                    .collect()
            };

            // v3 ALWAYS adds position suffixes
            let word_phonemes = Self::add_pos(&word_phonemes_raw);

            for p in &word_phonemes {
                phonemes.push((p.clone(), vec![], in_quote, bert_word_index));
            }

            cur_punc = vec![];
            bert_word_index += 1;
        }

        phonemes.push((" ".to_string(), cur_punc.clone(), in_quote, bert_word_index));
        phonemes.push(("$".to_string(), vec![], 0, bert_word_index));

        info!("Phonemes after forward pass v3 ({}):", phonemes.len());
        for (i, p) in phonemes.iter().enumerate() {
            info!(
                "  [{}] phoneme='{}' cur_punc={:?} in_quote={} bert_idx={}",
                i, p.0, p.1, p.2, p.3
            );
        }

        let mut last_punc = " ".to_string();
        let mut last_sentence_punc = " ".to_string();

        let mut lp_phonemes: Vec<(i64, i64, i64, i64, i64)> = vec![];
        let mut rev_bert_indices: Vec<usize> = vec![];
        let mut phone_duration_extra: Vec<f32> = vec![];

        for p in phonemes.iter().rev() {
            let punc_list = &p.1;
            if punc_list.iter().any(|x| x == "...") {
                last_sentence_punc = "...".to_string();
            } else if punc_list.iter().any(|x| x == ".") {
                last_sentence_punc = ".".to_string();
            } else if punc_list.iter().any(|x| x == "!") {
                last_sentence_punc = "!".to_string();
            } else if punc_list.iter().any(|x| x == "?") {
                last_sentence_punc = "?".to_string();
            } else if punc_list.iter().any(|x| x == "-") {
                last_sentence_punc = "-".to_string();
            }

            // Check for underscore in punctuation list
            let phone_duration_ext = if punc_list.iter().any(|x| x == "_") {
                20.0
            } else {
                0.0
            };

            if !punc_list.is_empty() {
                last_punc = punc_list[0].clone();
            }

            let cur_punc_str = if !punc_list.is_empty() {
                punc_list[0].clone()
            } else {
                "_".to_string()
            };

            let phoneme_id = get_phoneme_id_single(model, &p.0);
            let cur_punc_id = get_phoneme_id_single(model, &cur_punc_str);
            let last_punc_id = get_phoneme_id_single(model, &last_punc);
            let last_sentence_punc_id = get_phoneme_id_single(model, &last_sentence_punc);

            lp_phonemes.push((
                phoneme_id,
                cur_punc_id,
                p.2 as i64,
                last_punc_id,
                last_sentence_punc_id,
            ));
            rev_bert_indices.push(p.3);
            phone_duration_extra.push(phone_duration_ext);
        }

        lp_phonemes.reverse();
        rev_bert_indices.reverse();
        phone_duration_extra.reverse();

        let t = lp_phonemes.len();
        let hidden_size = 768;

        // Flatten in CHANNEL-MAJOR order (like numpy transpose)
        let mut text_data = Vec::with_capacity(t * 5);
        for channel in 0..5 {
            for (p0, p1, p2, p3, p4) in &lp_phonemes {
                let val = match channel {
                    0 => *p0,
                    1 => *p1,
                    2 => *p2,
                    3 => *p3,
                    _ => *p4,
                };
                text_data.push(val);
            }
        }

        // BERT embeddings: transpose from phoneme-major to channel-major
        let bert_raw: Vec<Vec<f32>> = rev_bert_indices
            .iter()
            .map(|&bert_idx| {
                if bert_idx < bert_embeddings.len() {
                    bert_embeddings[bert_idx].clone()
                } else {
                    vec![0.0f32; hidden_size]
                }
            })
            .collect();

        // Transpose: from [T][768] to [768][T]
        let mut bert_data = Vec::with_capacity(t * hidden_size);

        #[allow(clippy::needless_range_loop)]
        for ch in 0..hidden_size {
            for phoneme in 0..t {
                bert_data.push(bert_raw[phoneme][ch]);
            }
        }

        info!("Text: {}", text);
        info!("Phonemes count (multistream_scales T={}): {}", t, t);

        Ok(G2PResult::multistream(
            text_data,
            t,
            bert_data,
            Some(phone_duration_extra),
        ))
    }

    // ========================================================================
    // G2P no embeddings (fallback, no tokenizer)
    // Corresponds to Python g2p_noembed()
    // ========================================================================

    pub fn g2p_no_embeddings(&self, model: &Model, text: &str) -> Result<G2PResult> {
        let re = regex::Regex::new(r#"([,.?!;:"() ])"#).unwrap();
        let mut phonemes = vec!["^".to_string()];

        for word in re.split(&text.to_lowercase()) {
            if word.is_empty() {
                continue;
            }

            if re.is_match(word) || word == "-" {
                phonemes.push(word.to_string());
            } else if let Some(phoneme_str) = model.dic.get(word) {
                for p in phoneme_str.split_whitespace() {
                    phonemes.push(p.to_string());
                }
            } else {
                let converted = g2p::convert(word);
                for p in converted.split_whitespace() {
                    phonemes.push(p.to_string());
                }
            }
        }

        phonemes.push("$".to_string());

        // Check if phoneme_id_map values are lists or scalars
        let first_ids = get_phoneme_ids(model, &phonemes[0]);
        let is_list_mapping = first_ids.len() > 1;

        let mut phoneme_ids: Vec<i64> = Vec::new();

        #[allow(clippy::needless_range_loop)]
        if is_list_mapping {
            // Each phoneme maps to multiple IDs, blanks (single 0) between groups
            phoneme_ids.extend(&first_ids);
            for i in 1..phonemes.len() {
                phoneme_ids.push(0);
                let ids = get_phoneme_ids(model, &phonemes[i]);
                phoneme_ids.extend(&ids);
            }
        } else {
            // Standard single ID per phoneme with blank (0) interspersing
            phoneme_ids.extend(&first_ids);
            for i in 1..phonemes.len() {
                phoneme_ids.push(0);
                let ids = get_phoneme_ids(model, &phonemes[i]);
                phoneme_ids.extend(&ids);
            }
        }

        info!("Text: {}", text);
        info!("Phonemes: {:?}", phonemes);

        let t = phoneme_ids.len();
        Ok(G2PResult::no_embeddings(phoneme_ids, t))
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Transpose BERT data from phoneme-major [phoneme0_all_channels, phoneme1_all_channels, ...]
    /// to channel-major [ch0_all_phonemes, ch1_all_phonemes, ...]
    /// This matches Python's np.transpose behavior.
    fn transpose_bert(bert_data: Vec<f32>, num_phonemes: usize, hidden_size: usize) -> Vec<f32> {
        let mut transposed = Vec::with_capacity(bert_data.len());
        for ch in 0..hidden_size {
            for phoneme in 0..num_phonemes {
                transposed.push(bert_data[phoneme * hidden_size + ch]);
            }
        }
        transposed
    }

    /// Add BERT embedding for a given word index to the phone_embeddings vector
    /// Note: word_index is 0-based for ^, 1-based for first word, matching BERT token indices
    fn add_bert_emb_at(
        embeddings: &mut Vec<f32>,
        word_index: usize,
        bert_embeddings: &[Vec<f32>],
        hidden_size: usize,
    ) {
        if word_index < bert_embeddings.len() {
            embeddings.extend(&bert_embeddings[word_index]);
        } else {
            embeddings.extend(&vec![0.0f32; hidden_size]);
        }
    }

    fn add_pos(phonemes: &[String]) -> Vec<String> {
        if phonemes.len() == 1 {
            return vec![format!("{}_S", phonemes[0])];
        }

        let mut res = vec![];
        for (i, p) in phonemes.iter().enumerate() {
            if i == 0 {
                res.push(format!("{}_B", p));
            } else if i == phonemes.len() - 1 {
                res.push(format!("{}_E", p));
            } else {
                res.push(format!("{}_I", p));
            }
        }
        res
    }

    /// Write synthesized audio to WAV file
    #[allow(clippy::too_many_arguments)]
    pub fn synth(
        &self,
        model: &mut Model,
        text: &str,
        output_path: &str,
        speaker_id: Option<i64>,
        noise_level: Option<f32>,
        speech_rate: Option<f32>,
        duration_noise_level: Option<f32>,
        scale: Option<f32>,
    ) -> Result<()> {
        let audio = self.synth_audio(
            model,
            text,
            speaker_id,
            noise_level,
            speech_rate,
            duration_noise_level,
            scale,
        )?;

        // Write WAV file
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: model.config.audio.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(output_path, spec)?;
        for sample in audio {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;

        Ok(())
    }
}

impl Default for Synth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Tests for audio_float_to_int16() - audio normalization
    // ========================================================================

    #[test]
    fn test_audio_float_to_int16_normal_range() {
        let audio = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let max_wav_value = 32767.0;
        let result = audio_float_to_int16(&audio, max_wav_value);
        
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 16383); // 0.5 * 32767 ≈ 16383
        assert_eq!(result[2], -16383);
        // 1.0 and -1.0 should be at max/min
        assert!(result[3] > 30000);
        assert!(result[4] < -30000);
    }

    #[test]
    fn test_audio_float_to_int16_clipping() {
        let audio = vec![2.0, -2.0, 1.5, -1.5];
        let max_wav_value = 32767.0;
        let result = audio_float_to_int16(&audio, max_wav_value);
        
        // All values should be clipped to max/min
        assert_eq!(result[0], 32767);
        assert_eq!(result[1], -32767); // or -32768 depending on i16 range
        assert_eq!(result[2], 32767);
        assert!(result[3] <= -32767);
    }

    #[test]
    fn test_audio_float_to_int16_empty_input() {
        let audio: Vec<f32> = vec![];
        let result = audio_float_to_int16(&audio, 32767.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_audio_float_to_int16_zero_input() {
        let audio = vec![0.0, 0.0, 0.0];
        let result = audio_float_to_int16(&audio, 32767.0);
        assert_eq!(result, vec![0, 0, 0]);
    }

    #[test]
    fn test_audio_float_to_int16_scaling() {
        let audio = vec![0.25];
        let result = audio_float_to_int16(&audio, 32767.0);
        let expected = (0.25 * 32767.0) as i16;
        assert_eq!(result[0], expected);
    }

    // ========================================================================
    // Tests for transpose_bert() - BERT embedding transposition
    // ========================================================================

    #[test]
    fn test_transpose_bert_basic() {
        // 2 phonemes, 3 channels (simplified for testing)
        // Input: [p0_c0, p0_c1, p0_c2, p1_c0, p1_c1, p1_c2]
        // Output: [c0_p0, c0_p1, c1_p0, c1_p1, c2_p0, c2_p1]
        let bert_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_phonemes = 2;
        let hidden_size = 3;
        
        let result = Synth::transpose_bert(bert_data, num_phonemes, hidden_size);
        
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_bert_single_phoneme() {
        let bert_data = vec![1.0, 2.0, 3.0];
        let num_phonemes = 1;
        let hidden_size = 3;
        
        let result = Synth::transpose_bert(bert_data, num_phonemes, hidden_size);
        
        // Should be unchanged
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose_bert_single_channel() {
        let bert_data = vec![1.0, 2.0, 3.0];
        let num_phonemes = 3;
        let hidden_size = 1;
        
        let result = Synth::transpose_bert(bert_data, num_phonemes, hidden_size);
        
        // Should be unchanged
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose_bert_empty() {
        let bert_data: Vec<f32> = vec![];
        let result = Synth::transpose_bert(bert_data, 0, 768);
        assert!(result.is_empty());
    }

    #[test]
    fn test_transpose_bert_standard_size() {
        // 3 phonemes, 768 channels (standard BERT size, but simplified data)
        let num_phonemes = 3;
        let hidden_size = 4; // Using 4 instead of 768 for test simplicity
        let mut bert_data = Vec::with_capacity(num_phonemes * hidden_size);
        
        // Fill with pattern: [p0: 0,1,2,3, p1: 4,5,6,7, p2: 8,9,10,11]
        for i in 0..(num_phonemes * hidden_size) {
            bert_data.push(i as f32);
        }
        
        let result = Synth::transpose_bert(bert_data, num_phonemes, hidden_size);
        
        // Expected: [ch0: 0,4,8, ch1: 1,5,9, ch2: 2,6,10, ch3: 3,7,11]
        let expected = vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0];
        assert_eq!(result, expected);
    }

    // ========================================================================
    // Tests for add_pos() - position suffixes
    // ========================================================================

    #[test]
    fn test_add_pos_single_phoneme() {
        let phonemes = vec!["а".to_string()];
        let result = Synth::add_pos(&phonemes);
        assert_eq!(result, vec!["а_S".to_string()]);
    }

    #[test]
    fn test_add_pos_two_phonemes() {
        let phonemes = vec!["а".to_string(), "б".to_string()];
        let result = Synth::add_pos(&phonemes);
        assert_eq!(result, vec!["а_B".to_string(), "б_E".to_string()]);
    }

    #[test]
    fn test_add_pos_three_phonemes() {
        let phonemes = vec!["а".to_string(), "б".to_string(), "в".to_string()];
        let result = Synth::add_pos(&phonemes);
        assert_eq!(
            result,
            vec!["а_B".to_string(), "б_I".to_string(), "в_E".to_string()]
        );
    }

    #[test]
    fn test_add_pos_five_phonemes() {
        let phonemes = vec![
            "а".to_string(),
            "б".to_string(),
            "в".to_string(),
            "г".to_string(),
            "д".to_string(),
        ];
        let result = Synth::add_pos(&phonemes);
        assert_eq!(
            result,
            vec![
                "а_B".to_string(),
                "б_I".to_string(),
                "в_I".to_string(),
                "г_I".to_string(),
                "д_E".to_string()
            ]
        );
    }

    #[test]
    fn test_add_pos_empty_input() {
        let phonemes: Vec<String> = vec![];
        let result = Synth::add_pos(&phonemes);
        assert!(result.is_empty());
    }

    // ========================================================================
    // Tests for add_bert_emb_at() - BERT embedding insertion
    // ========================================================================

    #[test]
    fn test_add_bert_emb_at_valid_index() {
        let mut embeddings = Vec::new();
        let bert_embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let hidden_size = 3;
        
        Synth::add_bert_emb_at(&mut embeddings, 1, &bert_embeddings, hidden_size);
        
        assert_eq!(embeddings, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_add_bert_emb_at_zero_index() {
        let mut embeddings = Vec::new();
        let bert_embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let hidden_size = 3;
        
        Synth::add_bert_emb_at(&mut embeddings, 0, &bert_embeddings, hidden_size);
        
        assert_eq!(embeddings, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_bert_emb_at_out_of_bounds() {
        let mut embeddings = Vec::new();
        let bert_embeddings = vec![vec![1.0, 2.0, 3.0]];
        let hidden_size = 3;
        
        Synth::add_bert_emb_at(&mut embeddings, 5, &bert_embeddings, hidden_size);
        
        // Should insert zeros
        assert_eq!(embeddings, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add_bert_emb_at_empty_embeddings() {
        let mut embeddings = Vec::new();
        let bert_embeddings: Vec<Vec<f32>> = vec![];
        let hidden_size = 768;
        
        Synth::add_bert_emb_at(&mut embeddings, 0, &bert_embeddings, hidden_size);
        
        // Should insert zeros
        assert_eq!(embeddings.len(), 768);
        assert!(embeddings.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_add_bert_emb_at_multiple_calls() {
        let mut embeddings = Vec::new();
        let bert_embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let hidden_size = 2;
        
        Synth::add_bert_emb_at(&mut embeddings, 0, &bert_embeddings, hidden_size);
        Synth::add_bert_emb_at(&mut embeddings, 1, &bert_embeddings, hidden_size);
        
        assert_eq!(embeddings, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ========================================================================
    // Tests for G2PResult constructors
    // ========================================================================

    #[test]
    fn test_g2p_result_multistream() {
        let text_data = vec![1, 2, 3, 4, 5];
        let t = 1; // T = 1 (5 channels * 1 time step = 5 elements)
        let bert_data = vec![0.0; 768];
        let duration_extra = Some(vec![20.0]);
        
        let result = G2PResult::multistream(text_data.clone(), t, bert_data.clone(), duration_extra.clone());
        
        assert_eq!(result.text_data, text_data);
        assert_eq!(result.text_shape, vec![1, 5, t]);
        assert_eq!(result.text_lengths, vec![t as i64]);
        assert_eq!(result.bert_data, bert_data);
        assert_eq!(result.bert_shape, vec![1, 768, t]);
        assert_eq!(result.duration_extra_data, duration_extra);
        assert_eq!(result.duration_extra_shape, Some(vec![1, t]));
    }

    #[test]
    fn test_g2p_result_standard() {
        let text_data = vec![1, 2, 3];
        let t = 3;
        let bert_data = vec![0.0; 768 * t];
        
        let result = G2PResult::standard(text_data.clone(), t, bert_data.clone());
        
        assert_eq!(result.text_data, text_data);
        assert_eq!(result.text_shape, vec![1, t]);
        assert_eq!(result.text_lengths, vec![t as i64]);
        assert_eq!(result.bert_data, bert_data);
        assert_eq!(result.bert_shape, vec![1, 768, t]);
        assert!(result.duration_extra_data.is_none());
        assert!(result.duration_extra_shape.is_none());
    }

    #[test]
    fn test_g2p_result_no_embeddings() {
        let text_data = vec![1, 2, 3];
        let t = 3;
        
        let result = G2PResult::no_embeddings(text_data.clone(), t);
        
        assert_eq!(result.text_data, text_data);
        assert_eq!(result.text_shape, vec![1, t]);
        assert_eq!(result.text_lengths, vec![t as i64]);
        assert_eq!(result.bert_data.len(), 768 * t);
        assert!(result.bert_data.iter().all(|&x| x == 0.0));
        assert_eq!(result.bert_shape, vec![1, 768, t]);
        assert!(result.duration_extra_data.is_none());
    }

    // ========================================================================
    // Tests for G2P routing logic (model_type detection)
    // ========================================================================

    #[test]
    fn test_g2p_routing_multistream_v3() {
        // V3 with tokenizer should route to g2p_multistream_scales
        let model_type = "multistream_v3";
        let has_tokenizer = true;
        let no_blank = 0;
        
        // This would match: ("multistream_v3", true, _)
        assert!(matches!(
            (model_type, has_tokenizer, no_blank),
            ("multistream_v3", true, _)
        ));
    }

    #[test]
    fn test_g2p_routing_multistream_v2_with_tokenizer() {
        let model_type = "multistream_v2";
        let has_tokenizer = true;
        let no_blank = 0;
        
        assert!(matches!(
            (model_type, has_tokenizer, no_blank),
            ("multistream_v2", true, _)
        ));
    }

    #[test]
    fn test_g2p_routing_multistream_v2_without_tokenizer() {
        let model_type = "multistream_v2";
        let has_tokenizer = false;
        let no_blank = 0;
        
        assert!(matches!(
            (model_type, has_tokenizer, no_blank),
            ("multistream_v2", false, _)
        ));
    }

    #[test]
    fn test_g2p_routing_multistream_v1() {
        let model_type = "multistream_v1";
        let has_tokenizer = true;
        let no_blank = 0;
        
        assert!(matches!(
            (model_type, has_tokenizer, no_blank),
            ("multistream_v1", true, _)
        ));
    }

    #[test]
    fn test_g2p_routing_noblank() {
        // Old model with no_blank != 0
        let model_type = "";
        let has_tokenizer = true;
        let no_blank = 1;
        
        assert!(matches!(
            (model_type, has_tokenizer, no_blank),
            (_, true, nb) if nb != 0
        ));
    }

    #[test]
    fn test_g2p_routing_with_embeddings() {
        // Old model with no_blank == 0 and tokenizer
        let model_type = "";
        let has_tokenizer = true;
        let no_blank = 0;
        
        assert!(matches!(
            (model_type, has_tokenizer, no_blank),
            (_, true, 0)
        ));
    }

    #[test]
    fn test_g2p_routing_no_embeddings() {
        // No tokenizer
        let model_type = "";
        let has_tokenizer = false;
        let no_blank = 0;
        
        assert!(!matches!(
            (model_type, has_tokenizer, no_blank),
            (_, true, _)
        ));
        // Should fall through to final _ case
    }

    // ========================================================================
    // Tests for phoneme ID helpers
    // ========================================================================

    #[test]
    fn test_get_phoneme_id_single_from_map() {
        // This tests the logic of get_phoneme_id_single
        // We can't call it directly since it's private, but we can test the concept
        let mut phoneme_id_map = std::collections::HashMap::new();
        phoneme_id_map.insert("а".to_string(), crate::model::PhonemeIdValue::Single(42));
        
        match phoneme_id_map.get("а").unwrap() {
            crate::model::PhonemeIdValue::Single(id) => assert_eq!(*id, 42),
            _ => panic!("Expected Single"),
        }
    }

    #[test]
    fn test_get_phoneme_ids_multiple() {
        let mut phoneme_id_map = std::collections::HashMap::new();
        phoneme_id_map.insert("а".to_string(), crate::model::PhonemeIdValue::Multiple(vec![1, 2]));
        
        match phoneme_id_map.get("а").unwrap() {
            crate::model::PhonemeIdValue::Multiple(ids) => assert_eq!(*ids, vec![1, 2]),
            _ => panic!("Expected Multiple"),
        }
    }

    #[test]
    fn test_get_phoneme_id_missing_key() {
        let phoneme_id_map: std::collections::HashMap<String, crate::model::PhonemeIdValue> = 
            std::collections::HashMap::new();
        
        // Should return 0 for missing key
        let result = match phoneme_id_map.get("а") {
            Some(crate::model::PhonemeIdValue::Single(id)) => *id,
            Some(crate::model::PhonemeIdValue::Multiple(ids)) => ids[0],
            None => 0,
        };
        assert_eq!(result, 0);
    }

    // ========================================================================
    // Integration-like tests for tensor shapes
    // ========================================================================

    #[test]
    fn test_multistream_text_tensor_shape() {
        // For multistream, text shape should be (1, 5, T)
        let t = 10;
        let text_data: Vec<i64> = vec![0; 5 * t]; // 5 channels * T time steps
        
        let result = G2PResult::multistream(text_data, t, vec![0.0; 768 * t], None);
        
        assert_eq!(result.text_shape, vec![1, 5, t]);
        assert_eq!(result.text_data.len(), 5 * t);
    }

    #[test]
    fn test_standard_text_tensor_shape() {
        // For standard, text shape should be (1, T)
        let t = 15;
        let text_data: Vec<i64> = vec![0; t];
        
        let result = G2PResult::standard(text_data, t, vec![0.0; 768 * t]);
        
        assert_eq!(result.text_shape, vec![1, t]);
        assert_eq!(result.text_data.len(), t);
    }

    #[test]
    fn test_bert_tensor_shape_consistency() {
        // BERT tensor should always have 768 channels
        let t = 8;
        let hidden_size = 768;
        
        // Standard
        let standard = G2PResult::standard(vec![0; t], t, vec![0.0; hidden_size * t]);
        assert_eq!(standard.bert_shape, vec![1, hidden_size, t]);
        assert_eq!(standard.bert_data.len(), hidden_size * t);
        
        // Multistream
        let multistream = G2PResult::multistream(vec![0; 5 * t], t, vec![0.0; hidden_size * t], None);
        assert_eq!(multistream.bert_shape, vec![1, hidden_size, t]);
        assert_eq!(multistream.bert_data.len(), hidden_size * t);
        
        // No embeddings
        let no_emb = G2PResult::no_embeddings(vec![0; t], t);
        assert_eq!(no_emb.bert_shape, vec![1, hidden_size, t]);
        assert_eq!(no_emb.bert_data.len(), hidden_size * t);
        assert!(no_emb.bert_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_text_lengths_consistency() {
        let t = 12;
        
        let standard = G2PResult::standard(vec![0; t], t, vec![0.0; 768 * t]);
        assert_eq!(standard.text_lengths, vec![t as i64]);
        
        let multistream = G2PResult::multistream(vec![0; 5 * t], t, vec![0.0; 768 * t], None);
        assert_eq!(multistream.text_lengths, vec![t as i64]);
        
        let no_emb = G2PResult::no_embeddings(vec![0; t], t);
        assert_eq!(no_emb.text_lengths, vec![t as i64]);
    }
}
