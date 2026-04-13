# FST Dictionary for Fast Loading

## Overview

The FST (Finite State Transducer) dictionary provides near-instant loading of the Vosk TTS Russian pronunciation dictionary, replacing the traditional HashMap-based approach that takes several seconds to parse the 96 MB dictionary file.

## Problem

The original dictionary loading process:
1. Reads the 96 MB `dictionary` file line by line
2. Parses each line into word + phonemes
3. Builds a `HashMap<String, Vec<String>>` with 2M+ entries

**Loading time**: ~2.9 seconds (unacceptable for cold starts)

## Solution

The FST approach splits the dictionary into two optimized files:

| File | Size | Purpose |
|------|------|---------|
| `dictionary.fst` | 27 MB | FST map: word → (offset, length) |
| `dictionary.phonemes` | 51 MB | Concatenated phoneme data (null-separated) |

### How It Works

```
┌─────────────────────────────────────────────────┐
│                   dictionary.fst                 │
│  FST (Finite State Transducer) Map               │
│  ┌─────────┬─────────────────────────────┐       │
│  │ "кот"   │ offset=0x1A3F, len=0x0007   │       │
│  │ "котик" │ offset=0x1B20, len=0x000B   │       │
│  │ ...     │ ...                         │       │
│  └─────────┴─────────────────────────────┘       │
└─────────────────────────────────────────────────┘
                      │ lookup("кот")
                      ▼
┌─────────────────────────────────────────────────┐
│                dictionary.phonemes               │
│  ... │ k o1 t │ z a1 m o0 k\0z a0 m o1 k │ ...  │
│        ↑              ↑                          │
│    offset=          offset=                      │
│    0x1A3F            0x1B20                      │
└─────────────────────────────────────────────────┘
```

1. **Build phase**: Dictionary is parsed once, sorted, and compiled into an FST
2. **Load phase**: FST is read into memory (~5ms), phonemes file is memory-mapped (zero-copy)
3. **Lookup phase**: FST search returns (offset, length) → direct slice from mmap

### Multiple Pronunciations

Some words have multiple valid pronunciations (e.g., `замок` can be stressed on different syllables). These are stored as null-separated (`\0`) strings in the phonemes file:

```
замок → "z a1 m o0 k\0z a0 m o1 k"
```

## Usage

### Building the FST

```bash
cargo run --bin build_fst -- <dictionary_path> <output_dir>

# Example:
cargo run --bin build_fst -- vosk-model-tts-ru-0.9-multi/dictionary vosk-model-tts-ru-0.9-multi
```

This produces:
- `vosk-model-tts-ru-0.9-multi/dictionary.fst`
- `vosk-model-tts-ru-0.9-multi/dictionary.phonemes`

### Loading the FST Dictionary

```rust
use vosk_tts_rs::fst_dict::FstDictionary;

// Load from directory containing .fst and .phonemes files
let dict = FstDictionary::from_dir("vosk-model-tts-ru-0.9-multi")
    .expect("Failed to load FST dictionary");

// Look up phonemes (returns all pronunciations)
if let Some(pronunciations) = dict.lookup("привет") {
    for p in &pronunciations {
        println!("Pronunciation: {}", p);
    }
}

// Or get just the first pronunciation
if let Some(phonemes) = dict.lookup_first("кот") {
    println!("Phonemes: {}", phonemes);
}

// Check existence
if dict.contains("молоко") {
    println!("Word exists");
}

// Get word count
println!("Dictionary size: {} entries", dict.len());
```

## Performance Benchmarks

### Loading Time

| Method | Time | Speedup |
|--------|------|---------|
| HashMap (original) | 2.909s | 1× |
| **FST (new)** | **0.005s** | **593×** |

### Lookup Performance

| Method | Lookups/s | Notes |
|--------|-----------|-------|
| HashMap | 31.2M/s | In-memory hash table |
| FST | 1.7M/s | FST traversal + mmap slice |

The FST lookup is ~18× slower than HashMap but still very fast at 1.7 million lookups per second — more than sufficient for TTS workloads.

### File Sizes

| Component | Size | Notes |
|-----------|------|-------|
| Original dictionary | 96 MB | Plain text |
| FST file | 27 MB | Compressed FST structure |
| Phonemes file | 51 MB | Binary phoneme data |
| **Total** | **78 MB** | **18% smaller** than original |

The FST's compression comes from shared prefixes in words (e.g., "пре", "про", "при" share common paths in the trie).

## Architecture

### `FstDictionary` struct

```rust
pub struct FstDictionary {
    fst: Map<Vec<u8>>,       // FST map loaded into memory
    phonemes_mmap: Mmap,     // Memory-mapped phoneme data (zero-copy)
}
```

### Key Design Decisions

1. **FST read into memory**: The `fst` crate requires owned data (`Vec<u8>`). The 27 MB FST file fits comfortably in memory.

2. **Phonemes via mmap**: The 51 MB phonemes file is memory-mapped using `memmap2`, enabling zero-copy access to phoneme data. The OS handles page caching automatically.

3. **Single-value encoding**: Each FST value is a `u64` encoding both offset and length:
   ```
   value = (offset << 32) | length
   ```
   This allows offsets up to 4 billion (sufficient for 51 MB file) and lengths up to 4 GB per entry.

4. **Sorted insertion**: FSTs require keys in sorted order, so a `BTreeMap` is used during construction.

## Files

| File | Description |
|------|-------------|
| `src/fst_dict.rs` | FST dictionary module with `FstDictionary` struct |
| `src/bin/build_fst.rs` | CLI utility to build FST from raw dictionary |
| `src/bin/bench_fst.rs` | Benchmark comparing HashMap vs FST performance |

## Dependencies

```toml
fst = "0.4"      # FST implementation (BurntSushi/fst)
memmap2 = "0.9"  # Memory-mapped file support
```

## Future Improvements

1. **Lazy FST loading**: Currently the entire FST is loaded into memory. For memory-constrained environments, mmap-based FST loading could be explored.

2. **Phoneme compression**: Phoneme strings use space-separated codes. A simple encoding (e.g., single-byte phoneme IDs) could reduce the phonemes file size.

3. **Cache hot words**: Frequently looked-up words could be cached in a small LRU cache for even faster access.

4. **Incremental updates**: Currently the FST must be rebuilt from scratch. An incremental update mechanism would allow adding new words without full rebuild.
