use std::collections::HashSet;

const SOFT_LETTERS: &str = "яёюиье";
const START_SYL: &str = "#ъьаяоёуюэеиы-";
const OTHERS: &str = "#+-ьъ";

const SOFTHARD_CONS: &[(&str, &str)] = &[
    ("б", "b"),
    ("в", "v"),
    ("г", "g"),
    ("Г", "g"),
    ("д", "d"),
    ("з", "z"),
    ("к", "k"),
    ("л", "l"),
    ("м", "m"),
    ("н", "n"),
    ("п", "p"),
    ("р", "r"),
    ("с", "s"),
    ("т", "t"),
    ("ф", "f"),
    ("х", "h"),
];

const OTHER_CONS: &[(&str, &str)] = &[
    ("ж", "zh"),
    ("ц", "c"),
    ("ч", "ch"),
    ("ш", "sh"),
    ("щ", "sch"),
    ("й", "j"),
];

const VOWELS: &[(&str, &str)] = &[
    ("а", "a"),
    ("я", "a"),
    ("у", "u"),
    ("ю", "u"),
    ("о", "o"),
    ("ё", "o"),
    ("э", "e"),
    ("е", "e"),
    ("и", "i"),
    ("ы", "y"),
];

#[derive(Debug, Clone)]
struct Phone {
    symbol: String,
    stress: i32,
}

fn pallatize(phones: &mut Vec<Phone>) {
    let soft_letters: HashSet<char> = SOFT_LETTERS.chars().collect();

    for i in 0..phones.len().saturating_sub(1) {
        let phone_symbol = phones[i].symbol.clone();

        if let Some(&(_, replacement)) = SOFTHARD_CONS
            .iter()
            .find(|&&(letter, _)| letter == phone_symbol)
        {
            let next_char = phones[i + 1].symbol.chars().next();
            if let Some(next_ch) = next_char {
                if soft_letters.contains(&next_ch) {
                    phones[i] = Phone {
                        symbol: format!("{}j", replacement),
                        stress: 0,
                    };
                } else {
                    phones[i] = Phone {
                        symbol: replacement.to_string(),
                        stress: 0,
                    };
                }
            }
        }

        if let Some(&(_, replacement)) = OTHER_CONS
            .iter()
            .find(|&&(letter, _)| letter == phone_symbol)
        {
            phones[i] = Phone {
                symbol: replacement.to_string(),
                stress: 0,
            };
        }
    }
}

fn convert_vowels(phones: &[Phone]) -> Vec<String> {
    let mut new_phones = Vec::new();
    let start_syl: HashSet<char> = START_SYL.chars().collect();
    let soft_vowels: HashSet<char> = "яюеё".chars().collect();
    let vowel_map: std::collections::HashMap<char, &str> = VOWELS
        .iter()
        .map(|&(cyr, lat)| (cyr.chars().next().unwrap(), lat))
        .collect();

    let mut prev = String::from("");

    for phone in phones {
        if start_syl.contains(&prev.chars().next().unwrap_or('\0')) {
            let first_char = phone.symbol.chars().next();
            if let Some(ch) = first_char {
                if soft_vowels.contains(&ch) {
                    new_phones.push("j".to_string());
                }
            }
        }

        let first_char = phone.symbol.chars().next();
        if let Some(ch) = first_char {
            if let Some(&vowel) = vowel_map.get(&ch) {
                new_phones.push(format!("{}{}", vowel, phone.stress));
            } else {
                new_phones.push(phone.symbol.clone());
            }
        }

        prev = phone.symbol.clone();
    }

    new_phones
}

/// Converts a stress-marked Russian word to phoneme sequence
///
/// # Arguments
/// * `stress_word` - Word with stress marked using '+' character (e.g., "абстр+акция")
///
/// # Returns
/// Space-separated phoneme string
pub fn convert(stress_word: &str) -> String {
    let phones_str = format!("#{}#", stress_word);
    let others_set: HashSet<char> = OTHERS.chars().collect();

    // Assign stress marks
    let mut stress_phones = Vec::new();
    let mut stress = 0;

    for ch in phones_str.chars() {
        if ch == '+' {
            stress = 1;
        } else {
            stress_phones.push(Phone {
                symbol: ch.to_string(),
                stress,
            });
            stress = 0;
        }
    }

    // Palatalize
    pallatize(&mut stress_phones);

    // Convert vowels
    let phones = convert_vowels(&stress_phones);

    // Filter out unwanted characters
    let filtered: Vec<String> = phones
        .into_iter()
        .filter(|p| {
            p.chars()
                .next()
                .map(|ch| !others_set.contains(&ch))
                .unwrap_or(false)
        })
        .collect();

    filtered.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_simple() {
        let result = convert("пр+ивет");
        assert!(!result.is_empty());
        println!("Result: {}", result);
    }

    #[test]
    fn test_convert_no_stress() {
        let result = convert("привет");
        assert!(!result.is_empty());
        println!("Result: {}", result);
    }
}
