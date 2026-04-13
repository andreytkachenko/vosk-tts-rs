use std::path::Path;

use fst::{Map, Streamer};
use memmap2::Mmap;

/// FST-backed dictionary for fast phoneme lookup.
///
/// The dictionary is split into two files:
/// - `dictionary.fst`: FST map from word to (offset, length) in the phonemes file
/// - `dictionary.phonemes`: concatenated phoneme strings, null-separated
///
/// Both files are memory-mapped for instant loading and zero-copy access.
pub struct FstDictionary {
    fst: Map<Vec<u8>>,
    phonemes_mmap: Mmap,
}

impl FstDictionary {
    /// Load an FST dictionary from a directory containing `dictionary.fst` and `dictionary.phonemes`.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = dir.as_ref();
        let fst_path = dir.join("dictionary.fst");
        let phonemes_path = dir.join("dictionary.phonemes");

        if !fst_path.exists() || !phonemes_path.exists() {
            return Err(format!(
                "FST files not found in {}. Run `build_fst` first.",
                dir.display()
            )
            .into());
        }

        // Read FST into memory (fst requires owned data)
        let fst_data = std::fs::read(&fst_path)?;
        let fst = Map::new(fst_data)?;

        // Memory-map phonemes file
        let phonemes_file = std::fs::File::open(&phonemes_path)?;
        let phonemes_mmap = unsafe { Mmap::map(&phonemes_file)? };

        Ok(Self { fst, phonemes_mmap })
    }

    /// Look up phonemes for a word. Returns all pronunciations (may be multiple).
    pub fn lookup(&self, word: &str) -> Option<Vec<String>> {
        let value = self.fst.get(word)?;

        let offset = (value >> 32) as usize;
        let length = (value & 0xFFFFFFFF) as usize;

        // Slice from memory-mapped file (zero-copy)
        let data = &self.phonemes_mmap[offset..offset + length];

        // Split by null bytes to get multiple pronunciations
        let text = String::from_utf8_lossy(data);
        let pronunciations: Vec<String> = text
            .split('\0')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        if pronunciations.is_empty() {
            None
        } else {
            Some(pronunciations)
        }
    }

    /// Look up phonemes and return the first pronunciation.
    pub fn lookup_first(&self, word: &str) -> Option<String> {
        self.lookup(word).and_then(|mut v| v.pop())
    }

    /// Check if a word exists in the dictionary.
    pub fn contains(&self, word: &str) -> bool {
        self.fst.contains_key(word)
    }

    /// Return the number of unique words in the dictionary.
    pub fn len(&self) -> usize {
        self.fst.len() as usize
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.fst.is_empty()
    }

    /// Iterate over all words in the dictionary (sorted).
    pub fn words(&self) -> Vec<String> {
        let mut stream = self.fst.stream();
        let mut words = Vec::with_capacity(self.fst.len() as usize);
        while let Some((key, _)) = stream.next() {
            if let Ok(word) = std::str::from_utf8(key) {
                words.push(word.to_string());
            }
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write;

    fn create_test_fst() -> (tempfile::TempDir, FstDictionary) {
        let dir = tempfile::tempdir().unwrap();

        // Create test data
        let entries: BTreeMap<String, Vec<String>> = [
            ("привет".to_string(), vec!["p rj i0 vj e1 t".to_string()]),
            (
                "замок".to_string(),
                vec!["z a1 m o0 k".to_string(), "z a0 m o1 k".to_string()],
            ),
            ("кот".to_string(), vec!["k o1 t".to_string()]),
        ]
        .into_iter()
        .collect();

        // Build FST file
        let fst_path = dir.path().join("dictionary.fst");
        let phonemes_path = dir.path().join("dictionary.phonemes");

        let mut phonemes_file = std::fs::File::create(&phonemes_path).unwrap();
        let fst_file = std::fs::File::create(&fst_path).unwrap();
        let mut builder = fst::MapBuilder::new(fst_file).unwrap();
        let mut offset: u64 = 0;

        for (word, pronunciations) in &entries {
            let combined = pronunciations.join("\0");
            let bytes = combined.as_bytes();
            let len = bytes.len() as u64;

            phonemes_file.write_all(bytes).unwrap();
            phonemes_file.write_all(&[0]).unwrap();

            let value = (offset << 32) | (len + 1);
            builder.insert(word.as_bytes(), value).unwrap();

            offset += len + 1;
        }

        builder.finish().unwrap();

        let dict = FstDictionary::from_dir(dir.path()).unwrap();
        (dir, dict)
    }

    #[test]
    fn test_lookup_single() {
        let (_dir, dict) = create_test_fst();
        let result = dict.lookup("кот");
        assert!(result.is_some());
        let pronunciations = result.unwrap();
        assert_eq!(pronunciations.len(), 1);
        assert_eq!(pronunciations[0], "k o1 t");
    }

    #[test]
    fn test_lookup_multiple() {
        let (_dir, dict) = create_test_fst();
        let result = dict.lookup("замок");
        assert!(result.is_some());
        let pronunciations = result.unwrap();
        assert_eq!(pronunciations.len(), 2);
        assert!(pronunciations.contains(&"z a1 m o0 k".to_string()));
        assert!(pronunciations.contains(&"z a0 m o1 k".to_string()));
    }

    #[test]
    fn test_lookup_not_found() {
        let (_dir, dict) = create_test_fst();
        let result = dict.lookup("несуществующее");
        assert!(result.is_none());
    }

    #[test]
    fn test_lookup_first() {
        let (_dir, dict) = create_test_fst();
        let result = dict.lookup_first("замок");
        assert!(result.is_some());
        // Should return one of the two pronunciations
        let phoneme = result.unwrap();
        assert!(phoneme == "z a1 m o0 k" || phoneme == "z a0 m o1 k");
    }

    #[test]
    fn test_contains() {
        let (_dir, dict) = create_test_fst();
        assert!(dict.contains("привет"));
        assert!(dict.contains("замок"));
        assert!(!dict.contains("несуществующее"));
    }

    #[test]
    fn test_len() {
        let (_dir, dict) = create_test_fst();
        assert_eq!(dict.len(), 3);
    }

    #[test]
    fn test_words() {
        let (_dir, dict) = create_test_fst();
        let words = dict.words();
        assert_eq!(words.len(), 3);
        // Should be sorted
        assert_eq!(words[0], "замок");
        assert_eq!(words[1], "кот");
        assert_eq!(words[2], "привет");
    }
}
