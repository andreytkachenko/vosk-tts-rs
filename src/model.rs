use crate::error::{Error, Result};
use log::info;
use ndarray::ArrayD;
use ort::session::Session;
use ort::value::Value;
use reqwest::blocking::Client;
use serde::de::{self, Deserializer, Visitor};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::Tokenizer;
use zip::ZipArchive;

const MODEL_PRE_URL: &str = "https://alphacephei.com/vosk/models/";
const MODEL_LIST_URL: &str = "https://alphacephei.com/vosk/models/model-list.json";
const DEFAULT_LANGUAGE: &str = "ru";

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub lang: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub obsolete: String,
}

#[derive(Debug, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
}

#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "default_noise_level")]
    pub noise_level: f32,
    #[serde(default = "default_speech_rate")]
    pub speech_rate: f32,
    #[serde(default = "default_duration_noise_level")]
    pub duration_noise_level: f32,
    #[serde(default = "default_scale")]
    pub scale: f32,
}

fn default_noise_level() -> f32 {
    0.8
}
fn default_speech_rate() -> f32 {
    1.0
}
fn default_duration_noise_level() -> f32 {
    0.8
}
fn default_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone)]
pub enum PhonemeIdValue {
    Single(i64),
    Multiple(Vec<i64>),
}

impl<'de> Deserialize<'de> for PhonemeIdValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PhonemeIdValueVisitor;

        impl<'de> Visitor<'de> for PhonemeIdValueVisitor {
            type Value = PhonemeIdValue;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an integer or a list of integers")
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<PhonemeIdValue, E>
            where
                E: de::Error,
            {
                Ok(PhonemeIdValue::Single(value))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<PhonemeIdValue, E>
            where
                E: de::Error,
            {
                Ok(PhonemeIdValue::Single(value as i64))
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<PhonemeIdValue, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut vec = Vec::new();
                while let Some(elem) = seq.next_element()? {
                    vec.push(elem);
                }
                Ok(PhonemeIdValue::Multiple(vec))
            }
        }

        deserializer.deserialize_any(PhonemeIdValueVisitor)
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub audio: AudioConfig,
    pub inference: InferenceConfig,
    pub phoneme_id_map: HashMap<String, PhonemeIdValue>,
    pub num_symbols: u32,
    pub num_speakers: u32,
    #[serde(default)]
    pub speaker_id_map: HashMap<String, u32>,
    #[serde(default)]
    pub model_type: Option<String>,
    // Legacy fields for backward compatibility
    #[serde(default)]
    pub no_blank: Option<i64>,
}

pub struct Model {
    pub onnx: Session,
    pub dic: HashMap<String, String>,
    pub config: ModelConfig,
    pub tokenizer: Option<Tokenizer>,
    pub bert_onnx: Option<RefCell<Session>>,
}

impl Model {
    pub fn new(
        model_path: Option<&str>,
        model_name: Option<&str>,
        lang: Option<&str>,
    ) -> Result<Self> {
        let model_path = match model_path {
            Some(path) => PathBuf::from(path),
            None => Self::get_model_path(model_name, lang)?,
        };

        info!("Loading model from {}", model_path.display());

        // Load ONNX model
        let onnx = Session::builder()?
            .commit_from_file(model_path.join("model.onnx"))
            .map_err(Error::OnnxModelLoad)?;

        // Load dictionary
        let dic = Self::load_dictionary(&model_path)?;

        // Load config
        let config = Self::load_config(&model_path)?;
        info!("Config: {:#?}", config);

        // Load BERT tokenizer and model if available
        let bert_path = model_path.join("bert");
        info!("Loading BERT model from {}", bert_path.display());
        let (tokenizer, bert_onnx) = if bert_path.join("vocab.txt").exists() {
            // Load tokenizer from vocab.txt using WordPiece (BERT-style)
            let vocab_path = bert_path.join("vocab.txt");
            let vocab_path_str = vocab_path.to_string_lossy().to_string();
            let wp = WordPiece::from_file(&vocab_path_str)
                .unk_token("[UNK]".to_string())
                .continuing_subword_prefix("##".to_string())
                .build()
                .map_err(|e| Error::TokenizerBuild(e.to_string()))?;
            let mut tokenizer = Tokenizer::new(wp);

            // Add whitespace pre-tokenizer (like BERT)
            use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
            tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));

            // Add BERT post-processor (adds [CLS] and [SEP])
            tokenizer.with_post_processor(Some(BertProcessing::new(
                ("[SEP]".to_string(), 102), // [SEP] token ID
                ("[CLS]".to_string(), 101), // [CLS] token ID
            )));

            info!("tokenizer loaded");

            let bert_session = Session::builder()?
                .commit_from_file(bert_path.join("model.onnx"))
                .ok();

            (Some(tokenizer), bert_session.map(RefCell::new))
        } else {
            (None, None)
        };

        Ok(Model {
            onnx,
            dic,
            config,
            tokenizer,
            bert_onnx,
        })
    }

    fn load_dictionary(model_path: &Path) -> Result<HashMap<String, String>> {
        let mut dic = HashMap::new();
        let mut probs: HashMap<String, f32> = HashMap::new();

        let dict_path = model_path.join("dictionary");
        let content = fs::read_to_string(&dict_path).map_err(|e| Error::DictionaryRead {
            path: dict_path.to_string_lossy().to_string(),
            source: e,
        })?;

        for line in content.lines() {
            let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();

            if parts.len() >= 3 {
                let word = parts[0];
                let prob: f32 = parts[1].parse().unwrap_or(0.0);
                let phonemes = parts[2];

                let current_prob = probs.get(word).copied().unwrap_or(0.0);
                if prob > current_prob {
                    dic.insert(word.to_string(), phonemes.to_string());
                    probs.insert(word.to_string(), prob);
                }
            }
        }

        Ok(dic)
    }

    fn load_config(model_path: &Path) -> Result<ModelConfig> {
        let config_path = model_path.join("config.json");
        let content = fs::read_to_string(&config_path).map_err(|e| Error::ConfigRead {
            path: config_path.to_string_lossy().to_string(),
            source: e,
        })?;

        let config: ModelConfig = serde_json::from_str(&content).map_err(Error::ConfigParse)?;

        Ok(config)
    }

    /// Get BERT embeddings for text, similar to Python's get_word_bert
    /// Returns a list of [768] embeddings, one per token (excluding tokens starting with '#')
    pub fn get_word_bert(&self, text: &str, nopunc: bool) -> Option<Vec<Vec<f32>>> {
        let tokenizer = self.tokenizer.as_ref()?;
        let bert_session_ref = self.bert_onnx.as_ref()?;
        let mut bert_session = bert_session_ref.borrow_mut();

        // Encode text
        let text_clean = text.replace(['+', '_'], "");
        let encoding = tokenizer.encode(text_clean.as_str(), true).ok()?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let tokens = encoding.get_tokens();

        info!(
            "BERT encode: text='{}', tokens={:?}, input_ids={:?}",
            text_clean, tokens, ids
        );

        // Run BERT inference
        let input_ids_array = ArrayD::<i64>::from_shape_vec(vec![1, ids.len()], ids).ok()?;
        let attention_mask_array =
            ArrayD::<i64>::from_shape_vec(vec![1, attention_mask.len()], attention_mask).ok()?;
        let type_ids_array =
            ArrayD::<i64>::from_shape_vec(vec![1, type_ids.len()], type_ids).ok()?;

        let inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids_array).ok()?,
            "attention_mask" => Value::from_array(attention_mask_array).ok()?,
            "token_type_ids" => Value::from_array(type_ids_array).ok()?,
        ];

        let outputs = bert_session.run(inputs).ok()?;

        // Extract embeddings - shape [1, seq_len, 768]
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>().ok()?;
        info!(
            "BERT output: seq_len={}, first 10 values: {:?}",
            data.len() / 768,
            &data[..10.min(data.len())]
        );

        // data is [1, seq_len, 768] flattened
        let hidden_size = 768;
        let punc_pattern = regex::Regex::new(r#"[-,.?!;:"]"#).ok();

        // Select tokens that don't start with '#' (subword tokens)
        // For multistream models (nopunc=True), also filter punctuation
        // We include [CLS] and [SEP] since Python's word_index references them
        let mut selected_embeddings: Vec<Vec<f32>> = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            if !token.starts_with('#') {
                let skip_punc = nopunc
                    && punc_pattern
                        .as_ref()
                        .map(|p| p.is_match(token))
                        .unwrap_or(false);
                if !skip_punc {
                    // Extract embedding for this token
                    let start = i * hidden_size;
                    let end = start + hidden_size;
                    selected_embeddings.push(data[start..end].to_vec());
                }
            }
        }

        Some(selected_embeddings)
    }

    fn get_model_path(model_name: Option<&str>, lang: Option<&str>) -> Result<PathBuf> {
        let model_dirs = Self::get_model_dirs();

        match model_name {
            Some(name) => Self::get_model_by_name(name, &model_dirs),
            None => {
                let lang = lang.unwrap_or(DEFAULT_LANGUAGE);
                Self::get_model_by_lang(lang, &model_dirs)
            }
        }
    }

    fn get_model_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(path) = env::var("VOSK_MODEL_PATH") {
            dirs.push(PathBuf::from(path));
        }

        dirs.push(PathBuf::from("/usr/share/vosk"));

        if let Some(home) = dirs::home_dir() {
            if cfg!(target_os = "windows") {
                dirs.push(home.join("AppData").join("Local").join("vosk"));
            } else {
                dirs.push(home.join(".cache").join("vosk"));
            }
        }

        dirs
    }

    fn get_model_by_name(model_name: &str, model_dirs: &[PathBuf]) -> Result<PathBuf> {
        // Search in local directories
        for dir in model_dirs {
            if !dir.exists() {
                continue;
            }

            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    if let Ok(name) = entry.file_name().into_string() {
                        if name == model_name {
                            return Ok(entry.path());
                        }
                    }
                }
            }
        }

        // Download from remote
        let models = Self::fetch_model_list()?;
        if models.iter().any(|m| m.name == model_name) {
            let first_dir = model_dirs
                .iter()
                .find(|d| d.parent().is_some())
                .map(|d| d.parent().unwrap().to_path_buf())
                .unwrap_or_else(|| PathBuf::from("/tmp/vosk"));

            Self::download_model(&first_dir, model_name)?;
            Ok(first_dir.join(model_name))
        } else {
            Err(Error::ModelNotFound(model_name.to_string()))
        }
    }

    fn get_model_by_lang(lang: &str, model_dirs: &[PathBuf]) -> Result<PathBuf> {
        let pattern =
            regex::Regex::new(&format!(r"vosk-model-tts(-small)?-{}", regex::escape(lang)))?;

        // Search in local directories
        for dir in model_dirs {
            if !dir.exists() {
                continue;
            }

            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    println!("{:?} {}", entry, pattern);
                    if let Ok(name) = entry.file_name().into_string() {
                        if pattern.is_match(&name) {
                            return Ok(entry.path());
                        }
                    }
                }
            }
        }

        // Download from remote
        let models = Self::fetch_model_list()?;
        let matching_models: Vec<_> = models
            .iter()
            .filter(|m| m.lang == lang && m.model_type == "small" && m.obsolete == "false")
            .collect();

        if let Some(model) = matching_models.first() {
            let first_dir = model_dirs
                .iter()
                .find(|d| d.parent().is_some())
                .map(|d| d.parent().unwrap().to_path_buf())
                .unwrap_or_else(|| PathBuf::from("/tmp/vosk"));

            Self::download_model(&first_dir, &model.name)?;
            Ok(first_dir.join(&model.name))
        } else {
            Err(Error::LanguageNotFound(lang.to_string()))
        }
    }

    fn fetch_model_list() -> Result<Vec<ModelInfo>> {
        let client = Client::new();
        let response = client
            .get(MODEL_LIST_URL)
            .send()?
            .json::<Vec<ModelInfo>>()?;
        Ok(response)
    }

    fn download_model(base_dir: &Path, model_name: &str) -> Result<()> {
        let zip_url = format!("{}{}.zip", MODEL_PRE_URL, model_name);
        let zip_path = base_dir.join(format!("{}.zip", model_name));

        info!("Downloading model from {}", zip_url);

        // Create directory if it doesn't exist
        fs::create_dir_all(base_dir)?;

        // Download zip
        let client = Client::new();
        let response = client.get(&zip_url).send()?;
        let bytes = response.bytes()?;

        info!("Downloaded {} bytes", bytes.len());

        // Extract zip
        let mut archive = ZipArchive::new(std::io::Cursor::new(bytes.as_ref()))?;
        archive.extract(base_dir)?;

        // Clean up zip file
        if zip_path.exists() {
            fs::remove_file(zip_path)?;
        }

        Ok(())
    }
}

pub fn list_models() -> Result<()> {
    let client = Client::new();
    let response = client
        .get(MODEL_LIST_URL)
        .send()?
        .json::<Vec<ModelInfo>>()?;

    for model in response {
        println!("{}", model.name);
    }

    Ok(())
}

pub fn list_languages() -> Result<()> {
    let client = Client::new();
    let response = client
        .get(MODEL_LIST_URL)
        .send()?
        .json::<Vec<ModelInfo>>()?;

    let languages: std::collections::HashSet<String> =
        response.into_iter().map(|m| m.lang).collect();

    for lang in languages {
        println!("{}", lang);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Tests for PhonemeIdValue deserialization
    // ========================================================================

    #[test]
    fn test_phoneme_id_value_single() {
        let json = "42";
        let result: PhonemeIdValue = serde_json::from_str(json).unwrap();
        match result {
            PhonemeIdValue::Single(val) => assert_eq!(val, 42),
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_phoneme_id_value_multiple() {
        let json = "[1, 2, 3]";
        let result: PhonemeIdValue = serde_json::from_str(json).unwrap();
        match result {
            PhonemeIdValue::Multiple(vals) => assert_eq!(vals, vec![1, 2, 3]),
            _ => panic!("Expected Multiple variant"),
        }
    }

    #[test]
    fn test_phoneme_id_value_in_hashmap() {
        let json = r#"{"a": 1, "b": [2, 3]}"#;
        let result: HashMap<String, PhonemeIdValue> = serde_json::from_str(json).unwrap();

        match result.get("a").unwrap() {
            PhonemeIdValue::Single(val) => assert_eq!(*val, 1),
            _ => panic!("Expected Single for 'a'"),
        }

        match result.get("b").unwrap() {
            PhonemeIdValue::Multiple(vals) => assert_eq!(*vals, vec![2, 3]),
            _ => panic!("Expected Multiple for 'b'"),
        }
    }

    // ========================================================================
    // Tests for InferenceConfig defaults
    // ========================================================================

    #[test]
    fn test_inference_config_defaults() {
        let json = r#"{}"#;
        let config: InferenceConfig = serde_json::from_str(json).unwrap();

        assert!((config.noise_level - 0.8).abs() < f32::EPSILON);
        assert!((config.speech_rate - 1.0).abs() < f32::EPSILON);
        assert!((config.duration_noise_level - 0.8).abs() < f32::EPSILON);
        assert!((config.scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_explicit_values() {
        let json = r#"{
            "noise_level": 0.5,
            "speech_rate": 1.2,
            "duration_noise_level": 0.6,
            "scale": 0.9
        }"#;
        let config: InferenceConfig = serde_json::from_str(json).unwrap();

        assert!((config.noise_level - 0.5).abs() < f32::EPSILON);
        assert!((config.speech_rate - 1.2).abs() < f32::EPSILON);
        assert!((config.duration_noise_level - 0.6).abs() < f32::EPSILON);
        assert!((config.scale - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_partial_defaults() {
        let json = r#"{"noise_level": 0.3}"#;
        let config: InferenceConfig = serde_json::from_str(json).unwrap();

        assert!((config.noise_level - 0.3).abs() < f32::EPSILON);
        assert!((config.speech_rate - 1.0).abs() < f32::EPSILON); // default
        assert!((config.duration_noise_level - 0.8).abs() < f32::EPSILON); // default
        assert!((config.scale - 1.0).abs() < f32::EPSILON); // default
    }

    // ========================================================================
    // Tests for ModelConfig deserialization
    // ========================================================================

    #[test]
    fn test_model_config_minimal() {
        let json = r#"{
            "audio": {"sample_rate": 24000},
            "inference": {},
            "phoneme_id_map": {"a": 1},
            "num_symbols": 100,
            "num_speakers": 1
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.audio.sample_rate, 24000);
        assert_eq!(config.num_symbols, 100);
        assert_eq!(config.num_speakers, 1);
        assert!(config.model_type.is_none());
        assert!(config.no_blank.is_none());
        assert!(config.speaker_id_map.is_empty());
    }

    #[test]
    fn test_model_config_full() {
        let json = r#"{
            "audio": {"sample_rate": 48000},
            "inference": {"noise_level": 0.5, "speech_rate": 1.1},
            "phoneme_id_map": {"a": 1, "b": [2, 3]},
            "num_symbols": 200,
            "num_speakers": 5,
            "speaker_id_map": {"alice": 0},
            "model_type": "multistream_v2",
            "no_blank": 1
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.audio.sample_rate, 48000);
        assert!((config.inference.noise_level - 0.5).abs() < f32::EPSILON);
        assert!((config.inference.speech_rate - 1.1).abs() < f32::EPSILON);
        assert_eq!(config.num_symbols, 200);
        assert_eq!(config.num_speakers, 5);
        assert_eq!(config.speaker_id_map.get("alice"), Some(&0));
        assert_eq!(config.model_type, Some("multistream_v2".to_string()));
        assert_eq!(config.no_blank, Some(1));
    }

    #[test]
    fn test_model_config_multistream_v1() {
        let json = r#"{
            "audio": {"sample_rate": 24000},
            "inference": {},
            "phoneme_id_map": {"a": 1},
            "num_symbols": 100,
            "num_speakers": 1,
            "model_type": "multistream_v1"
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, Some("multistream_v1".to_string()));
    }

    #[test]
    fn test_model_config_multistream_v3() {
        let json = r#"{
            "audio": {"sample_rate": 24000},
            "inference": {},
            "phoneme_id_map": {"a": 1},
            "num_symbols": 100,
            "num_speakers": 1,
            "model_type": "multistream_v3"
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, Some("multistream_v3".to_string()));
    }

    #[test]
    fn test_model_config_legacy_no_blank() {
        let json = r#"{
            "audio": {"sample_rate": 24000},
            "inference": {},
            "phoneme_id_map": {"a": 1},
            "num_symbols": 100,
            "num_speakers": 1,
            "no_blank": 1
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.no_blank, Some(1));
        assert!(config.model_type.is_none());
    }

    #[test]
    fn test_model_config_no_blank_zero() {
        let json = r#"{
            "audio": {"sample_rate": 24000},
            "inference": {},
            "phoneme_id_map": {"a": 1},
            "num_symbols": 100,
            "num_speakers": 1,
            "no_blank": 0
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.no_blank, Some(0));
    }

    // ========================================================================
    // Tests for AudioConfig
    // ========================================================================

    #[test]
    fn test_audio_config() {
        let json = r#"{"sample_rate": 44100}"#;
        let config: AudioConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sample_rate, 44100);
    }

    #[test]
    fn test_audio_config_standard_rate() {
        let json = r#"{"sample_rate": 24000}"#;
        let config: AudioConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sample_rate, 24000);
    }

    // ========================================================================
    // Tests for ModelInfo
    // ========================================================================

    #[test]
    fn test_model_info_deserialization() {
        let json = r#"{
            "name": "vosk-model-tts-ru",
            "lang": "ru",
            "type": "small",
            "obsolete": "false"
        }"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();

        assert_eq!(info.name, "vosk-model-tts-ru");
        assert_eq!(info.lang, "ru");
        assert_eq!(info.model_type, "small");
        assert_eq!(info.obsolete, "false");
    }

    #[test]
    fn test_model_info_list() {
        let json = r#"[
            {"name": "model1", "lang": "ru", "type": "small", "obsolete": "false"},
            {"name": "model2", "lang": "en", "type": "large", "obsolete": "true"}
        ]"#;
        let infos: Vec<ModelInfo> = serde_json::from_str(json).unwrap();

        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].name, "model1");
        assert_eq!(infos[1].lang, "en");
    }

    // ========================================================================
    // Tests for dictionary loading logic (simulated)
    // ========================================================================

    #[test]
    fn test_dictionary_priority_selection() {
        // Simulate dictionary loading logic:
        // When same word has multiple entries, highest probability wins
        let mut dic: HashMap<String, String> = HashMap::new();
        let mut probs: HashMap<String, f32> = HashMap::new();

        let lines = vec![
            "привет 0.5 ph r' iy v' e t",
            "привет 0.8 ph r' i v' e t",
            "привет 0.3 ph r' iy v' e t0",
        ];

        for line in lines {
            let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();
            if parts.len() >= 3 {
                let word = parts[0];
                let prob: f32 = parts[1].parse().unwrap_or(0.0);
                let phonemes = parts[2];

                let current_prob = probs.get(word).copied().unwrap_or(0.0);
                if prob > current_prob {
                    dic.insert(word.to_string(), phonemes.to_string());
                    probs.insert(word.to_string(), prob);
                }
            }
        }

        // Should have selected the entry with 0.8 probability
        assert_eq!(dic.get("привет"), Some(&"ph r' i v' e t".to_string()));
        assert_eq!(*probs.get("привет").unwrap(), 0.8);
    }

    #[test]
    fn test_dictionary_single_entry() {
        let mut dic: HashMap<String, String> = HashMap::new();
        let mut probs: HashMap<String, f32> = HashMap::new();

        let line = "мир 0.9 m' i r";
        let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();

        if parts.len() >= 3 {
            let word = parts[0];
            let prob: f32 = parts[1].parse().unwrap_or(0.0);
            let phonemes = parts[2];
            dic.insert(word.to_string(), phonemes.to_string());
            probs.insert(word.to_string(), prob);
        }

        assert_eq!(dic.get("мир"), Some(&"m' i r".to_string()));
    }

    #[test]
    fn test_dictionary_malformed_lines() {
        let mut dic: HashMap<String, String> = HashMap::new();
        let mut probs: HashMap<String, f32> = HashMap::new();

        let lines = vec![
            "valid 0.5 v a l' i d",
            "invalid_no_prob",
            "also_invalid 0.5",
            "",
        ];

        for line in lines {
            let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();
            if parts.len() >= 3 {
                let word = parts[0];
                let prob: f32 = parts[1].parse().unwrap_or(0.0);
                let phonemes = parts[2];

                let current_prob = probs.get(word).copied().unwrap_or(0.0);
                if prob > current_prob {
                    dic.insert(word.to_string(), phonemes.to_string());
                    probs.insert(word.to_string(), prob);
                }
            }
        }

        // Only the valid line should be added
        assert_eq!(dic.len(), 1);
        assert!(dic.contains_key("valid"));
    }

    #[test]
    fn test_dictionary_invalid_probability() {
        let mut dic: HashMap<String, String> = HashMap::new();
        let mut probs: HashMap<String, f32> = HashMap::new();

        let line = "word not_a_number phonemes";
        let parts: Vec<&str> = line.splitn(3, char::is_whitespace).collect();

        if parts.len() >= 3 {
            let word = parts[0];
            let prob: f32 = parts[1].parse().unwrap_or(0.0); // Should default to 0.0
            let phonemes = parts[2];

            let current_prob = probs.get(word).copied().unwrap_or(0.0);
            if prob > current_prob {
                dic.insert(word.to_string(), phonemes.to_string());
                probs.insert(word.to_string(), prob);
            }
        }

        // Should not be added since prob is 0.0
        assert!(!dic.contains_key("word"));
    }
}
