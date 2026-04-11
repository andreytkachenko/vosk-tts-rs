use anyhow::{Context, Result};
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
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PhonemeIdValueVisitor;

        impl<'de> Visitor<'de> for PhonemeIdValueVisitor {
            type Value = PhonemeIdValue;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an integer or a list of integers")
            }

            fn visit_i64<E>(self, value: i64) -> Result<PhonemeIdValue, E>
            where
                E: de::Error,
            {
                Ok(PhonemeIdValue::Single(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<PhonemeIdValue, E>
            where
                E: de::Error,
            {
                Ok(PhonemeIdValue::Single(value as i64))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<PhonemeIdValue, A::Error>
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
            .context("Failed to load ONNX model")?;

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
                .map_err(|e| anyhow::anyhow!("Failed to build WordPiece tokenizer: {}", e))?;
            let mut tokenizer = Tokenizer::new(wp);

            // Add whitespace pre-tokenizer (like BERT)
            use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
            tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));

            // Add BERT post-processor (adds [CLS] and [SEP])
            tokenizer.with_post_processor(Some(BertProcessing::new(
                ("[SEP]".to_string(), 102),  // [SEP] token ID
                ("[CLS]".to_string(), 101),  // [CLS] token ID
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
        let content = fs::read_to_string(&dict_path)
            .context(format!("Failed to read dictionary from {:?}", dict_path))?;

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
        let content = fs::read_to_string(&config_path)
            .context(format!("Failed to read config from {:?}", config_path))?;

        let config: ModelConfig =
            serde_json::from_str(&content).context("Failed to parse config.json")?;

        Ok(config)
    }

    /// Get BERT embeddings for text, similar to Python's get_word_bert
    /// Returns a list of [768] embeddings, one per token (excluding tokens starting with '#')
    pub fn get_word_bert(&self, text: &str, nopunc: bool) -> Option<Vec<Vec<f32>>> {
        let tokenizer = self.tokenizer.as_ref()?;
        let bert_session_ref = self.bert_onnx.as_ref()?;
        let mut bert_session = bert_session_ref.borrow_mut();

        // Encode text
        let text_clean = text.replace('+', "").replace('_', "");
        let encoding = tokenizer
            .encode(text_clean.as_str(), true)
            .ok()?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let tokens = encoding.get_tokens();

        info!("BERT encode: text='{}', tokens={:?}, input_ids={:?}", text_clean, tokens, ids);

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
        info!("BERT output: seq_len={}, first 10 values: {:?}", data.len() / 768, &data[..10.min(data.len())]);

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
            anyhow::bail!("Model name {} does not exist", model_name)
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
            anyhow::bail!("Language {} does not exist", lang)
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
