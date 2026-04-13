use std::fs::File;
use std::io::{BufRead, BufReader};

use vosk_tts_rs::g2p::convert_with_stress;

fn strip_stress(s: &str) -> String {
    s.split_whitespace()
        .map(|p| {
            let chars: Vec<char> = p.chars().collect();
            if chars.len() > 1 {
                let last = chars.last().unwrap();
                if *last == '0' || *last == '1' {
                    chars[..chars.len()-1].iter().collect()
                } else {
                    p.to_string()
                }
            } else {
                p.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn count_vowels(s: &str) -> usize {
    s.chars().filter(|c| "аоуэыяёюеиАОУЭЫЯЁЮЕИ".contains(*c)).count()
}

fn is_cyrillic_only(word: &str) -> bool {
    word.chars().all(|c| 
        ('\u{0410}'..='\u{044F}').contains(&c) || 
        c == 'ё' || c == 'Ё' || c == '-' || c == '\''
    )
}

fn main() {
    let file = File::open("vosk-model-tts-ru-0.9-multi/dictionary")
        .expect("Не удалось открыть файл");
    let reader = BufReader::new(file);

    let mut total_cyr = 0usize;
    let mut matched_m2 = 0usize;

    // Для анализа: сколько гласных в словах где не совпало
    let mut mismatch_by_vowel_count: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
    
    // Позиция ударения для каждого количества гласных
    let mut stress_pos_by_vowel_count: std::collections::HashMap<usize, std::collections::HashMap<usize, usize>> = std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line.expect("Ошибка чтения");
        let line = line.trim().to_string();
        if line.is_empty() { continue; }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() < 2 { continue; }

        let word = parts[0];
        let rest = parts[1];

        if !is_cyrillic_only(word) { continue; }
        total_cyr += 1;

        let phonemes: Vec<&str> = rest.split_whitespace().skip(1).collect();
        let dict_phonemes = phonemes.join(" ");

        let r2 = convert_with_stress(word, Some(2));
        let r2_no_stress = strip_stress(&r2);
        let dict_no_stress = strip_stress(&dict_phonemes);

        if r2 == dict_phonemes {
            matched_m2 += 1;
            continue;
        }

        // Не совпало — анализируем
        let vowel_count = count_vowels(word);
        
        // Считаем позицию ударения в словаре (какая по счёту гласная с 1)
        let mut stressed_vowel_num: Option<usize> = None;
        let mut vowel_num = 0;
        for ph in &phonemes {
            let first_char = ph.chars().next().unwrap_or('\0');
            if "aoeuyi".contains(first_char) {
                if ph.ends_with('1') {
                    stressed_vowel_num = Some(vowel_num); // 0-based
                    break;
                }
                vowel_num += 1;
            }
        }

        if let Some(sv) = stressed_vowel_num {
            let entry = stress_pos_by_vowel_count.entry(vowel_count).or_default();
            *entry.entry(sv).or_insert(0) += 1;
        }

        let entry = mismatch_by_vowel_count.entry(vowel_count).or_insert((0, 0));
        if r2_no_stress == dict_no_stress {
            entry.0 += 1; // только ударения
        } else {
            entry.1 += 1; // фонемы разные
        }

        if total_cyr % 200000 == 0 {
            eprintln!("Обработано: {} кириллических слов...", total_cyr);
        }
    }

    eprintln!("\n=== АНАЛИЗ РАСХОЖДЕНИЙ (Метод 2: 2-я с конца) ===");
    eprintln!("Всего кириллических слов: {}", total_cyr);
    eprintln!("Совпало: {} ({:.2}%)", matched_m2, (matched_m2 as f64 / total_cyr as f64) * 100.0);
    eprintln!("Не совпало: {}", total_cyr - matched_m2);

    eprintln!("\n--- Расхождения по количеству гласных ---");
    eprintln!("Гласных | Только ударения | Фонемы разные");
    let mut sorted: Vec<_> = mismatch_by_vowel_count.iter().collect();
    sorted.sort_by_key(|(k, _)| **k);
    for (vc, (stress_only, phones_diff)) in &sorted {
        eprintln!("{:>7} | {:>15} | {:?}", vc, stress_only, phones_diff);
    }

    eprintln!("\n--- Позиция ударения для слов где Метод 2 не совпал ---");
    eprintln!("Гласных | Позиция ударения (0-based) → количество");
    let mut stress_sorted: Vec<_> = stress_pos_by_vowel_count.iter().collect();
    stress_sorted.sort_by_key(|(k, _)| **k);
    for (vc, positions) in &stress_sorted {
        if **vc > 6 { continue; } // пропускаем длинные
        let pos_list: Vec<_> = positions.iter().collect();
        let pos_str: Vec<String> = pos_list.iter()
            .map(|(p, c)| format!("{}→{}", p, c))
            .collect();
        eprintln!("{:>7} | {}", vc, pos_str.join(", "));
    }
}
