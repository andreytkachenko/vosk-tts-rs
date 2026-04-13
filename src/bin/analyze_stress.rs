use std::fs::File;
use std::io::{BufRead, BufReader};

use vosk_tts_rs::g2p::convert;

fn strip_stress(s: &str) -> String {
    s.split_whitespace()
        .map(|p| {
            // Убираем 0/1 в конце фонемы: "a1" -> "a", "nj0" -> "nj"
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

fn is_cyrillic_word(word: &str) -> bool {
    word.chars().any(|c| ('\u{0410}'..='\u{044F}').contains(&c) || c == 'ё' || c == 'Ё')
}

fn is_only_cyrillic_and_hyphen(word: &str) -> bool {
    word.chars().all(|c| 
        ('\u{0410}'..='\u{044F}').contains(&c) || 
        ('\u{0401}'..='\u{0401}').contains(&c) ||
        c == 'ё' || c == 'Ё' || c == '-'
    )
}

fn main() {
    let file = File::open("vosk-model-tts-ru-0.9-multi/dictionary")
        .expect("Не удалось открыть файл");
    let reader = BufReader::new(file);

    let mut total_cyrillic = 0usize;
    let mut match_with_stress = 0usize;
    let mut match_without_stress = 0usize;
    let mut stress_only_diff = 0usize; // отличаются только ударениями
    let mut phones_diff = 0usize; // отличаются фонемами

    let mut stress_from_end_counts: [usize; 6] = [0; 6]; // [1-я с конца, 2-я с конца, ...]
    let mut single_stress_words = 0usize;

    for line in reader.lines() {
        let line = line.expect("Ошибка чтения");
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() < 2 {
            continue;
        }

        let word = parts[0];
        let rest = parts[1];

        // Только кириллические слова
        if !is_only_cyrillic_and_hyphen(word) {
            continue;
        }

        total_cyrillic += 1;

        let phonemes_from_dict: Vec<&str> = rest.split_whitespace().skip(1).collect();
        let dict_phonemes = phonemes_from_dict.join(" ");
        let dict_no_stress = strip_stress(&dict_phonemes);

        let g2p_result = convert(word);
        let g2p_no_stress = strip_stress(&g2p_result);

        if g2p_result == dict_phonemes {
            match_with_stress += 1;
            match_without_stress += 1;
            continue;
        }

        if g2p_no_stress == dict_no_stress {
            match_without_stress += 1;
            stress_only_diff += 1;

            // Анализируем позицию ударения в словаре
            // Считаем гласные с 1 в dict
            let stressed_count = phonemes_from_dict.iter().filter(|p| p.ends_with('1')).count();
            if stressed_count == 1 {
                single_stress_words += 1;
                // Находим позицию ударной гласной среди всех гласных
                let vowel_ph_with_stress: Vec<usize> = phonemes_from_dict.iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.chars().next().map(|c| "aoeuyi".contains(c)).unwrap_or(false) 
                        && p.ends_with('1')
                    })
                    .map(|(i, _)| i)
                    .collect();
                
                // Позиция с конца среди гласных фонем
                let total_vowels_in_ph: usize = phonemes_from_dict.iter()
                    .filter(|p| p.chars().next().map(|c| "aoeuyi".contains(c)).unwrap_or(false))
                    .count();
                
                if let Some(vph_idx) = vowel_ph_with_stress.first() {
                    let vowel_num = phonemes_from_dict[..*vph_idx].iter()
                        .filter(|p| p.chars().next().map(|c| "aoeuyi".contains(c)).unwrap_or(false))
                        .count();
                    let from_end = total_vowels_in_ph - vowel_num;
                    if from_end > 0 && from_end <= 5 {
                        stress_from_end_counts[from_end - 1] += 1;
                    } else if from_end >= 6 {
                        stress_from_end_counts[5] += 1;
                    }
                }
            }
        } else {
            phones_diff += 1;
        }

        if total_cyrillic % 100000 == 0 {
            eprintln!("Обработано: {} кириллических слов...", total_cyrillic);
        }
    }

    eprintln!("\n=== Статистика для кириллических слов ===");
    eprintln!("Всего кириллических слов: {}", total_cyrillic);
    eprintln!("Совпадение С ударениями: {} ({:.2}%)", match_with_stress, 
              (match_with_stress as f64 / total_cyrillic as f64) * 100.0);
    eprintln!("Совпадение БЕЗ учёта ударений: {} ({:.2}%)", match_without_stress,
              (match_without_stress as f64 / total_cyrillic as f64) * 100.0);
    eprintln!("Только ударения различаются: {} ({:.2}%)", stress_only_diff,
              (stress_only_diff as f64 / total_cyrillic as f64) * 100.0);
    eprintln!("Фонемы различаются: {} ({:.2}%)", phones_diff,
              (phones_diff as f64 / total_cyrillic as f64) * 100.0);

    eprintln!("\n=== Позиция ударения (для слов с одним ударением) ===");
    eprintln!("Слов с одним ударением: {}", single_stress_words);
    let labels = ["1-я с конца", "2-я с конца", "3-я с конца", "4-я с конца", "5-я с конца", "6+ с конца"];
    for (i, count) in stress_from_end_counts.iter().enumerate() {
        if single_stress_words > 0 {
            let pct = (*count as f64 / single_stress_words as f64) * 100.0;
            eprintln!("{}: {} ({:.2}%)", labels[i], count, pct);
        }
    }
}
