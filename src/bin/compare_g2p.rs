use std::fs::File;
use std::io::{BufRead, BufReader};

use vosk_tts_rs::g2p::{convert, convert_with_stress};

fn main() {
    let file = File::open("vosk-model-tts-ru-0.9-multi/dictionary")
        .expect("Не удалось открыть файл dictionary");
    let reader = BufReader::new(file);

    let mut total_lines = 0usize;
    
    // Метод 1: convert без ударений (baseline)
    let mut m1_match = 0usize;
    
    // Метод 2: фиксированное ударение на 2-ю с конца
    let mut m2_match = 0usize;
    
    // Метод 3: smart heuristic (None = auto by vowel count)
    let mut m3_match = 0usize;

    // Примеры улучшений
    let mut m3_examples = 30usize;

    for line in reader.lines() {
        let line = line.expect("Ошибка чтения строки");
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        total_lines += 1;

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() < 2 {
            continue;
        }

        let word = parts[0];
        let rest = parts[1];

        let phonemes_from_dict: Vec<&str> = rest.split_whitespace().skip(1).collect();
        let dict_phonemes = phonemes_from_dict.join(" ");

        // Метод 1: без ударений
        let r1 = convert(word);
        if r1 == dict_phonemes {
            m1_match += 1;
        }

        // Метод 2: 2-я с конца
        let r2 = convert_with_stress(word, Some(2));
        if r2 == dict_phonemes {
            m2_match += 1;
        }

        // Метод 3: smart heuristic
        let r3 = convert_with_stress(word, None);
        if r3 == dict_phonemes {
            m3_match += 1;
            // Показываем примеры где smart лучше fixed
            if m3_examples > 0 && r2 != dict_phonemes && (m3_match as i64) > (m2_match as i64) {
                m3_examples -= 1;
                println!(
                    "smart: {} | dict: '{}' | m2: '{}' | m3: '{}'",
                    word, dict_phonemes, r2, r3
                );
            }
        }

        if total_lines % 100000 == 0 {
            eprintln!("Обработано: {} строк...", total_lines);
        }
    }

    eprintln!("\n=== СРАВНЕНИЕ МЕТОДОВ ===");
    eprintln!("Всего строк: {}", total_lines);
    
    eprintln!("\nМетод 1 (baseline, без ударений):  {} ({:.2}%)", m1_match,
              (m1_match as f64 / total_lines as f64) * 100.0);
    eprintln!("Метод 2 (fixed 2-я с конца):       {} ({:.2}%)  [+{}]", m2_match,
              (m2_match as f64 / total_lines as f64) * 100.0, m2_match as i64 - m1_match as i64);
    eprintln!("Метод 3 (smart heuristic):         {} ({:.2}%)  [+{}]", m3_match,
              (m3_match as f64 / total_lines as f64) * 100.0, m3_match as i64 - m1_match as i64);
    eprintln!("Метод 3 vs Метод 2:                              [+{}]", m3_match as i64 - m2_match as i64);
}
