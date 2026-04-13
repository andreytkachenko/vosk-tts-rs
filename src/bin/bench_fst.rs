use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use vosk_tts_rs::fst_dict::FstDictionary;

/// Benchmark: Compare loading time of traditional HashMap dictionary vs FST dictionary.
fn main() {
    let dict_dir = "vosk-model-tts-ru-0.9-multi";

    // ============================================================
    // Benchmark 1: Traditional HashMap dictionary loading
    // ============================================================
    println!("=== Benchmark 1: Traditional HashMap loading ===");

    let start = Instant::now();
    let file = File::open(format!("{}/dictionary", dict_dir))
        .expect("Cannot open dictionary");
    let reader = BufReader::new(file);

    let mut hash_dict: HashMap<String, Vec<String>> = HashMap::new();
    let mut line_count = 0usize;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() < 2 {
            continue;
        }

        line_count += 1;
        let word = parts[0].to_string();
        let rest = parts[1];
        let phonemes: Vec<&str> = rest.split_whitespace().skip(1).collect();
        let phoneme_str = phonemes.join(" ");

        hash_dict.entry(word).or_default().push(phoneme_str);
    }

    let load_time_hash = start.elapsed();
    let mem_size_hash = {
        // Approximate memory: number of entries * average key/value size
        let total_chars: usize = hash_dict
            .iter()
            .map(|(k, v)| k.len() + v.iter().map(|s| s.len()).sum::<usize>())
            .sum();
        total_chars
    };

    println!(
        "Loaded {} lines, {} unique words in {:.3}s",
        line_count,
        hash_dict.len(),
        load_time_hash.as_secs_f64()
    );
    println!("Approximate data size: {:.2} MB", mem_size_hash as f64 / 1024.0 / 1024.0);

    // ============================================================
    // Benchmark 2: FST dictionary loading
    // ============================================================
    println!("\n=== Benchmark 2: FST dictionary loading ===");

    let start = Instant::now();
    let fst_dict = FstDictionary::from_dir(dict_dir).expect("Failed to load FST dictionary");
    let load_time_fst = start.elapsed();

    let fst_file_size = std::fs::metadata(format!("{}/dictionary.fst", dict_dir))
        .unwrap()
        .len();
    let phonemes_file_size = std::fs::metadata(format!("{}/dictionary.phonemes", dict_dir))
        .unwrap()
        .len();

    println!(
        "Loaded {} unique words in {:.3}s",
        fst_dict.len(),
        load_time_fst.as_secs_f64()
    );
    println!(
        "FST file: {:.2} MB, Phonemes file: {:.2} MB (total: {:.2} MB)",
        fst_file_size as f64 / 1024.0 / 1024.0,
        phonemes_file_size as f64 / 1024.0 / 1024.0,
        (fst_file_size + phonemes_file_size) as f64 / 1024.0 / 1024.0
    );

    // ============================================================
    // Benchmark 3: Lookup performance
    // ============================================================
    println!("\n=== Benchmark 3: Lookup performance ===");

    // Test words (mix of common, rare, and non-existent)
    let test_words = vec![
        "привет",
        "молоко",
        "здравствуйте",
        "электричество",
        "несуществующее",
        "кот",
        "работа",
        "человек",
        "хорошо",
        "спасибо",
    ];

    // HashMap lookups
    let start = Instant::now();
    let mut hash_hits = 0;
    for _ in 0..100 {
        for word in &test_words {
            if hash_dict.contains_key(*word) {
                hash_hits += 1;
            }
        }
    }
    let lookup_time_hash = start.elapsed();

    // FST lookups
    let start = Instant::now();
    let mut fst_hits = 0;
    for _ in 0..100 {
        for word in &test_words {
            if fst_dict.contains(word) {
                fst_hits += 1;
            }
        }
    }
    let lookup_time_fst = start.elapsed();

    println!(
        "HashMap: {} hits in {:.3}s ({:.0} lookups/s)",
        hash_hits,
        lookup_time_hash.as_secs_f64(),
        (test_words.len() * 100) as f64 / lookup_time_hash.as_secs_f64()
    );
    println!(
        "FST:     {} hits in {:.3}s ({:.0} lookups/s)",
        fst_hits,
        lookup_time_fst.as_secs_f64(),
        (test_words.len() * 100) as f64 / lookup_time_fst.as_secs_f64()
    );

    // ============================================================
    // Summary
    // ============================================================
    println!("\n=== Summary ===");
    println!(
        "Load time:   HashMap {:.3}s → FST {:.3}s ({:.1}× faster)",
        load_time_hash.as_secs_f64(),
        load_time_fst.as_secs_f64(),
        load_time_hash.as_secs_f64() / load_time_fst.as_secs_f64()
    );
    println!(
        "Lookup time: HashMap {:.3}s → FST {:.3}s ({:.1}× {})",
        lookup_time_hash.as_secs_f64(),
        lookup_time_fst.as_secs_f64(),
        if lookup_time_hash < lookup_time_fst {
            lookup_time_fst.as_secs_f64() / lookup_time_hash.as_secs_f64()
        } else {
            lookup_time_hash.as_secs_f64() / lookup_time_fst.as_secs_f64()
        },
        if lookup_time_hash < lookup_time_fst {
            "slower"
        } else {
            "faster"
        }
    );
}
