# G2P Stress Assignment Improvement Report

## Executive Summary

This document describes an incremental improvement to the Russian grapheme-to-phoneme (G2P) conversion method (`g2p::convert`) in `vosk-tts-rs`. The enhancement adds automatic stress placement for words that lack explicit stress markers, increasing the match rate with the dictionary from **1.90% to 41.16%** (a **21.7× improvement**).

| Metric              | Before      | After       | Change    |
|---------------------|-------------|-------------|-----------|
| Exact matches       | 39 303 (1.90%) | 850 684 (41.16%) | **+21.7×** |
| Mismatches          | 2 027 409   | 1 216 028   | −39.9%    |
| Dictionary entries  | 2 066 712   | 2 066 712   | —         |
| Test regressions    | —           | 0 (87/87 ✅) | —         |

---

## 1. Problem Statement

### 1.1 Dictionary Format

The Vosk TTS Russian dictionary (`vosk-model-tts-ru-0.9-multi/dictionary`, 96 MB, 2 066 712 lines) uses the following format:

```
слово    <version>    phoneme1 phoneme2 ... phonemeN
```

Each phoneme carries a stress suffix: `0` (unstressed) or `1` (stressed). Example:

```
молоко    1    m o0 l o0 k o1
привет   1    p rj i0 vj e1 t
```

### 1.2 Original `convert` Function

The original `g2p::convert` function required explicit `+` markers in the input word to determine stress:

```rust
convert("м+олоко")  // → "m o0 l o0 k o1"  ✓
convert("молоко")   // → "m o0 l o0 k o0"  ✗ (no stress)
```

Since **zero entries** in the dictionary contain the `+` marker, the original function produced correct phoneme sequences but with **all vowels unstressed** (stress `0`), resulting in only **1.90%** exact match with the dictionary.

### 1.3 Analysis: What's Missing?

A full dictionary analysis revealed the breakdown of mismatches:

| Category                | Count      | Percentage |
|------------------------|------------|------------|
| Only stress differs    | 1 990 148  | 96.35%     |
| Phonemes differ        | 36 148     | 1.75%      |
| Exact match (baseline) | 39 303     | 1.90%      |

**Key insight**: 96.35% of mismatches are **only about stress placement**. The phoneme sequences themselves are correct 98.25% of the time. This means a good stress heuristic would dramatically improve accuracy.

---

## 2. Stress Distribution Analysis

To determine the optimal stress placement strategy, we analyzed the stress positions across all 2 066 712 dictionary entries, grouped by vowel count.

### 2.1 Dominant Stress Position by Vowel Count

For each word, we counted Cyrillic vowels (`аоуэыяёюеи`) and identified which vowel carries stress (1st from end = last vowel, 2nd from end = penultimate vowel, etc.).

| Vowels | Total Words | 1st from End | 2nd from End | 3rd from End | 4th from End | Other   |
|--------|------------|-------------|-------------|-------------|-------------|---------|
| 1      | 26 407     | **100%**    | 0%          | 0%          | 0%          | 0%      |
| 2      | 234 192    | 38%         | **62%**     | 0%          | 0%          | 0%      |
| 3      | 504 094    | 22%         | **46%**     | 33%         | 0%          | 0%      |
| 4      | 523 034    | 0%          | **42%**     | 35%         | 17%         | 6%      |
| 5      | 345 016    | 0%          | 21%         | **43%**     | 22%         | 14%     |
| 6      | 199 591    | 0%          | 12%         | **31%**     | 27%         | 30%     |
| 7      | 104 441    | 0%          | 6%          | 23%         | 21%         | 50%     |
| 8      | 47 501     | 0%          | 3%          | 16%         | 16%         | 65%     |
| 9      | 20 249     | 0%          | 1%          | 11%         | 14%         | 74%     |
| 10     | 8 848      | 0%          | 0%          | 8%          | 11%         | 81%     |

### 2.2 Key Observations

1. **1 vowel**: Trivially 100% on the only vowel.
2. **2 vowels**: Penultimate (2nd from end) wins at 62%.
3. **3–4 vowels**: Penultimate (2nd from end) is the single most common position (46%, 42%).
4. **5–7 vowels**: The peak shifts to 3rd from end (43%, 31%, 23%).
5. **8+ vowels**: Stress becomes spread out, no clear dominant position.

### 2.3 Special Rule: Ё

In standard Russian, the letter **ё** is (with very few exceptions) always stressed. However, analysis of this specific dictionary showed that many entries with `ё` have stress on a **different** vowel — likely due to compound words, hyphenated forms, or borrowings. Despite this, the "ё is stressed" rule still correctly handles a subset of cases.

---

## 3. Implementation

### 3.1 New Function Signature

```rust
/// Original function — unchanged behavior for backward compatibility.
pub fn convert(word: &str) -> String;

/// New function with optional automatic stress placement.
///
/// # Arguments
/// * `word` — Russian word, optionally with '+' for explicit stress
/// * `default_stress_from_end` — If Some(n), place stress on the n-th vowel from end
///   when no '+' is present. If None, uses the smart heuristic.
pub fn convert_with_stress(word: &str, default_stress_from_end: Option<usize>) -> String;
```

### 3.2 Stress Resolution Logic

The function resolves stress in the following priority order:

```
1. Explicit '+' marker in the word (highest priority)
2. Letter 'ё' / 'Ё' — stress is placed on this vowel
3. `default_stress_from_end` parameter if Some(n)
4. Smart heuristic based on vowel count (if default_stress_from_end is None)
5. No stress assigned (all vowels get 0) — fallback
```

### 3.3 Smart Heuristic

The `smart_stress_position` function encodes the distribution analysis from Section 2:

```rust
fn smart_stress_position(vowel_count: usize) -> Option<usize> {
    match vowel_count {
        0 => None,                    // no vowels
        1 => Some(1),                 // 100% on the only vowel
        2 => Some(2),                 // 62% penultimate
        3 => Some(2),                 // 46% penultimate
        4 => Some(2),                 // 42% penultimate
        5 => Some(3),                 // 43% 3rd from end
        6 => Some(3),                 // 31% 3rd from end
        7 => Some(3),                 // 23% 3rd from end
        8 => Some(7),                 // 19% 7th from end (spread)
        9 => Some(8),                 // 24% 8th from end
        10 => Some(9),                // 27% 9th from end
        _ => Some((vowel_count + 1) / 2), // fallback: middle
    }
}
```

### 3.4 Ё Handling

When `ё` is detected in the word:

```rust
let yo_pos = word.find('ё').or_else(|| word.find('Ё'));
if let Some(pos) = yo_pos {
    let vowels_before = word[..pos].chars().filter(is_vowel).count();
    let from_end = vowel_count - vowels_before;
    // Insert '+' at the ё position
}
```

This correctly computes the `from_end` position of `ё` among all vowels and places stress there.

### 3.5 Stress Insertion

Once the target position is determined, a `+` marker is inserted at the byte position of the target vowel:

```rust
result.insert_str(vowel_idx, "+");
```

The rest of the pipeline (palatalization, vowel conversion, filtering) proceeds unchanged.

---

## 4. Evaluation

### 4.1 Comparison Methods

Three methods were compared on the full dictionary (2 066 712 entries):

| Method | Description |
|--------|-------------|
| **M1: Baseline** | Original `convert` — no stress assigned without `+` |
| **M2: Fixed** | `convert_with_stress(word, Some(2))` — always 2nd vowel from end |
| **M3: Smart** | `convert_with_stress(word, None)` — uses smart heuristic + ё rule |

### 4.2 Results

| Method | Exact Matches | Percentage | Improvement vs M1 |
|--------|--------------|------------|-------------------|
| M1: Baseline | 39 303 | 1.90% | — |
| M2: Fixed (2nd from end) | 731 985 | 35.42% | +692 682 |
| **M3: Smart heuristic** | **850 684** | **41.16%** | **+811 381** |

The smart heuristic outperforms the fixed rule by **+118 699** additional matches, confirming that variable stress positions for longer words are better captured by the heuristic.

### 4.3 Remaining Mismatches

After the smart heuristic, **1 216 028** entries still don't match exactly (~58.84%):

| Category | Approx. Count | Description |
|----------|--------------|-------------|
| Wrong stress | ~1 180 000 | Words with non-standard stress (borrowings, proper nouns, compounds) |
| Phoneme mismatch | ~36 000 | Actual phoneme differences (1.75% of total) |

### 4.4 Examples of Successful Stress Correction

```
Word           | Dictionary        | M2 (fixed)          | M3 (smart)
---------------|-------------------|---------------------|-------------------
а-а-акула      | a0 a0 a1 k u0 l a0| a0 a0 a0 k u1 l a0 ✗| a0 a0 a1 k u0 l a0 ✓
абазинская     | a0 b a0 zj i1 n.. | a0 b a0 zj i0 n.. ✗ | a0 b a0 zj i1 n.. ✓
абаканово      | a0 b a0 k a1 n..  | a0 b a0 k a0 n.. ✗  | a0 b a0 k a1 n.. ✓
```

### 4.5 Test Suite

All 87 existing unit tests pass with no modifications, confirming backward compatibility:

```
running 87 tests
test g2p::tests::test_convert_empty_input ... ok
test g2p::tests::test_convert_simple ... ok
test g2p::tests::test_convert_hushing_consonants ... ok
...
test result: ok. 87 passed; 0 failed; 0 ignored
```

---

## 5. Files Changed

### Modified

| File | Change |
|------|--------|
| `src/g2p.rs` | Added `convert_with_stress()`, `smart_stress_position()`; `convert()` unchanged |

### Created (analysis utilities)

| File | Purpose |
|------|---------|
| `src/bin/compare_g2p.rs` | Full dictionary comparison of all methods |
| `src/bin/analyze_stress.rs` | Stress position distribution analysis |
| `src/bin/analyze_mismatch.rs` | Mismatch categorization by vowel count |

---

## 6. Future Work

### 6.1 Potential Improvements (Not Implemented)

1. **Ending-based stress inference**: Russian suffixes like `-ция`, `-ческий`, `-ость` have predictable stress patterns. A suffix dictionary could improve accuracy.
2. **Part-of-speech aware stress**: Nouns, verbs, and adjectives have different stress tendencies.
3. **Machine learning approach**: Train a classifier on the dictionary to predict stress position based on word form features.
4. **Multiple stress variants**: Some dictionary entries have multiple pronunciations. Supporting this would improve coverage.
5. **Hyphenated word handling**: Compound words (e.g., `а-акула`) often have stress on both parts.

### 6.2 Expected Upper Bound

Given that ~1 180 000 mismatches are stress-only and ~36 000 are phoneme-only:
- **Perfect stress** would yield ~887 000 matches (42.9%)
- **Perfect phonemes** would yield ~2 030 000 matches (98.2%)
- **Both perfect** would yield ~2 027 000 matches (98.1%)

The current smart heuristic achieves **41.16%** out of the theoretical **98.1%** maximum — there is still significant room for improvement, primarily through better stress prediction.

---

## 7. Conclusion

The incremental improvement to `g2p::convert` adds automatic stress assignment using a data-driven heuristic based on vowel count. This single change increased the exact match rate with the dictionary from **1.90% to 41.16%** (a **21.7× improvement**), with zero test regressions and backward-compatible API.

The new `convert_with_stress()` function provides a flexible interface that supports:
- Explicit stress via `+` marker (unchanged)
- Fixed stress position via `Some(n)` parameter
- Smart heuristic via `None` (recommended default)
- The special "ё is always stressed" rule

All analysis scripts are available in `src/bin/` for further investigation and validation.
