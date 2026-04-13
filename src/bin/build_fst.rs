use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write, BufWriter};
use std::time::Instant;

/// Builds an FST-backed dictionary from the Vosk TTS dictionary file.
///
/// Output files:
/// - dictionary.fst: FST map (word → offset + length in phonemes file)
/// - dictionary.phonemes: concatenated phoneme data
fn main() {
    let dict_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vosk-model-tts-ru-0.9-multi/dictionary".to_string());
    let output_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "vosk-model-tts-ru-0.9-multi".to_string());

    println!("Building FST from: {}", dict_path);
    let start = Instant::now();

    // Phase 1: Read dictionary and group by word
    let file = File::open(&dict_path).expect("Cannot open dictionary file");
    let reader = BufReader::new(file);

    // BTreeMap for sorted insertion (required by fst::Map)
    // word → Vec<phoneme_string>
    let mut word_entries: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total_lines = 0usize;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        total_lines += 1;

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() < 2 {
            continue;
        }

        let word = parts[0].to_string();
        let rest = parts[1];

        // Extract phonemes (skip version number)
        let phonemes: Vec<&str> = rest.split_whitespace().skip(1).collect();
        let phoneme_str = phonemes.join(" ");

        word_entries.entry(word).or_default().push(phoneme_str);

        if total_lines % 200000 == 0 {
            let elapsed = start.elapsed();
            eprintln!(
                "Read {} lines in {:.2}s, {} unique words...",
                total_lines,
                elapsed.as_secs_f64(),
                word_entries.len()
            );
        }
    }

    let elapsed_read = start.elapsed();
    eprintln!(
        "Read {} lines, {} unique words in {:.2}s",
        total_lines,
        word_entries.len(),
        elapsed_read.as_secs_f64()
    );

    // Phase 2: Build phoneme data file and FST map
    // We'll store phonemes as null-separated strings in a binary file
    // FST value: (offset, length) encoded as u64: (offset << 32) | length

    let fst_path = format!("{}/dictionary.fst", output_dir);
    let phonemes_path = format!("{}/dictionary.phonemes", output_dir);

    let phonemes_file = File::create(&phonemes_path).expect("Cannot create phonemes file");
    let mut phonemes_writer = BufWriter::new(phonemes_file);

    let mut fst_map: Vec<(Vec<u8>, u64)> = Vec::with_capacity(word_entries.len());
    let mut current_offset: u64 = 0;
    let mut total_phoneme_bytes = 0u64;

    for (word, phonemes_list) in &word_entries {
        // Join multiple pronunciations with null byte
        let combined: String = phonemes_list.join("\0");
        let bytes = combined.as_bytes();
        let len = bytes.len() as u64;

        // Write to phonemes file
        phonemes_writer.write_all(bytes).expect("Failed to write phonemes");
        // Write null terminator
        phonemes_writer.write_all(&[0]).expect("Failed to write terminator");
        total_phoneme_bytes += len + 1;

        // Encode value: (offset << 32) | (length + 1)  // +1 for terminator
        let encoded_value = (current_offset << 32) | (len + 1);
        fst_map.push((word.as_bytes().to_vec(), encoded_value));

        current_offset += len + 1;

        if fst_map.len() % 200000 == 0 {
            eprintln!(
                "Building FST: {} / {} entries...",
                fst_map.len(),
                word_entries.len()
            );
        }
    }

    phonemes_writer.flush().expect("Failed to flush phonemes file");
    let elapsed_phonemes = start.elapsed();
    eprintln!(
        "Written {} bytes of phoneme data in {:.2}s",
        total_phoneme_bytes,
        elapsed_phonemes.as_secs_f64()
    );

    // Phase 3: Build and serialize FST
    eprintln!("Building FST map...");
    let fst_file = File::create(&fst_path).expect("Failed to create FST file");
    let mut fst_builder = fst::MapBuilder::new(fst_file).expect("Failed to create FST builder");

    for (key, value) in &fst_map {
        fst_builder.insert(key, *value).expect("Failed to insert into FST");
    }

    fst_builder.finish().expect("Failed to finish FST");

    let elapsed_total = start.elapsed();
    eprintln!("FST built in {:.2}s total", elapsed_total.as_secs_f64());

    // Phase 4: Print stats
    let fst_size = std::fs::metadata(&fst_path).unwrap().len();
    let phonemes_size = std::fs::metadata(&phonemes_path).unwrap().len();

    eprintln!("\n=== Output Files ===");
    eprintln!("FST file:      {} ({:.2} MB)", fst_path, fst_size as f64 / 1024.0 / 1024.0);
    eprintln!("Phonemes file: {} ({:.2} MB)", phonemes_path, phonemes_size as f64 / 1024.0 / 1024.0);
    eprintln!("Total entries: {}", word_entries.len());
    eprintln!("\nDone!");
}
