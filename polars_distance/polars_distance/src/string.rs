use rapidfuzz::distance::*;

// HAMMING
pub fn hamming_dist(x: &str, y: &str) -> u32 {
    hamming::distance_with_args(x.chars(), y.chars(), &hamming::Args::default().pad(true)) as u32
}

pub fn hamming_normalized_dist(x: &str, y: &str) -> f64 {
    hamming::normalized_distance_with_args(
        x.chars(),
        y.chars(),
        &hamming::Args::default().pad(true),
    )
}

// LEVENSHTEIN
pub fn levenshtein_dist(x: &str, y: &str) -> u32 {
    levenshtein::distance(x.chars(), y.chars()) as u32
}

pub fn levenshtein_normalized_dist(x: &str, y: &str) -> f64 {
    levenshtein::normalized_distance(x.chars(), y.chars())
}

// DAM LEVENSHTEIN
pub fn dam_levenshtein_dist(x: &str, y: &str) -> u32 {
    damerau_levenshtein::distance(x.chars(), y.chars()) as u32
}

pub fn dam_levenshtein_normalized_dist(x: &str, y: &str) -> f64 {
    damerau_levenshtein::normalized_distance(x.chars(), y.chars())
}

// INDEL
pub fn indel_dist(x: &str, y: &str) -> u32 {
    indel::distance(x.chars(), y.chars()) as u32
}

pub fn indel_normalized_dist(x: &str, y: &str) -> f64 {
    indel::normalized_distance(x.chars(), y.chars())
}

// JARO
pub fn jaro_dist(x: &str, y: &str) -> u32 {
    jaro::distance(x.chars(), y.chars()) as u32
}

pub fn jaro_normalized_dist(x: &str, y: &str) -> f64 {
    jaro::normalized_distance(x.chars(), y.chars())
}

// JARO WINKLER
pub fn jaro_winkler_dist(x: &str, y: &str) -> u32 {
    jaro_winkler::distance(x.chars(), y.chars()) as u32
}

pub fn jaro_winkler_normalized_dist(x: &str, y: &str) -> f64 {
    jaro_winkler::normalized_distance(x.chars(), y.chars())
}

// LONGEST COMMON SUB SEQUENCE
pub fn lcs_seq_dist(x: &str, y: &str) -> u32 {
    lcs_seq::distance(x.chars(), y.chars()) as u32
}

pub fn lcs_seq_normalized_dist(x: &str, y: &str) -> f64 {
    lcs_seq::normalized_distance(x.chars(), y.chars())
}

// OSA
pub fn osa_dist(x: &str, y: &str) -> u32 {
    osa::distance(x.chars(), y.chars()) as u32
}

pub fn osa_normalized_dist(x: &str, y: &str) -> f64 {
    osa::normalized_distance(x.chars(), y.chars())
}

// POSTFIX
pub fn postfix_dist(x: &str, y: &str) -> u32 {
    postfix::distance(x.chars(), y.chars()) as u32
}

pub fn postfix_normalized_dist(x: &str, y: &str) -> f64 {
    postfix::normalized_distance(x.chars(), y.chars())
}

// PREFIX
pub fn prefix_dist(x: &str, y: &str) -> u32 {
    prefix::distance(x.chars(), y.chars()) as u32
}

pub fn prefix_normalized_dist(x: &str, y: &str) -> f64 {
    prefix::normalized_distance(x.chars(), y.chars())
}
