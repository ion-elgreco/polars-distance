use distances::strings::{hamming, levenshtein};

pub fn hamming_distance_string(x: &str, y: &str) -> u32 {
    hamming(x, y)
}

pub fn levenshtein_distance_string(x: &str, y: &str) -> u32 {
    levenshtein(x, y)
}
