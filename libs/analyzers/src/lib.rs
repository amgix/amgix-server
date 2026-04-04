use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use murmurhash3::murmurhash3_x86_32;
use rust_stemmers::{Algorithm, Stemmer};
use unicode_segmentation::UnicodeSegmentation;
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use stopwords::{NLTK, Language, Stopwords};

// BM25+ TF saturation parameters with length normalization
// TF_sat = ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d|/log(|d|)))))
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
const BM25_DELTA: f32 = 0.0;

/// Base multipliers on BM25 TF saturation for each WMTR channel (`tokenize_wmtr` only).
const WMTR_WEIGHT_WHITESPACE: f32 = 4.0;
const WMTR_WEIGHT_LANG: f32 = 8.0;

/// Compute BM25-style TF saturation with length normalization
/// Formula: ((tf * (k1 + 1)) / (tf + k1 * length_norm))
/// length_norm = 1 - b + b * (|d|/avgdl) where |d| is doc length and avgdl is average doc length
fn bm25_tf_saturation(tf: f32, length_norm: f32) -> f32 {
    ((tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * length_norm)) + BM25_DELTA
}

// Cache stopword sets per language code to avoid per-call reconstruction
static STOPWORDS_CACHE: Lazy<RwLock<HashMap<String, Arc<HashSet<String>>>>> = Lazy::new(|| {
    RwLock::new(HashMap::new())
});

/// Map language code to NLTK Language enum
/// Only includes languages supported by the stopwords crate
fn lang_code_to_nltk_language(lang_code: &str) -> Option<Language> {
    match lang_code.to_lowercase().as_str() {
        "en" => Some(Language::English),
        "es" => Some(Language::Spanish),
        "fr" => Some(Language::French),
        "de" => Some(Language::German),
        "it" => Some(Language::Italian),
        "pt" => Some(Language::Portuguese),
        "nl" => Some(Language::Dutch),
        "ru" => Some(Language::Russian),
        "ar" => Some(Language::Arabic),
        "tr" => Some(Language::Turkish),
        "sv" => Some(Language::Swedish),
        "da" => Some(Language::Danish),
        "no" => Some(Language::Norwegian),
        "fi" => Some(Language::Finnish),
        "hu" => Some(Language::Hungarian),
        "ro" => Some(Language::Romanian),
        "el" => Some(Language::Greek),
        _ => None,
    }
}

fn get_stopwords_cached(lang_code: &str) -> Arc<HashSet<String>> {
    // Fast path: try read lock
    if let Ok(cache) = STOPWORDS_CACHE.read() {
        if let Some(set_arc) = cache.get(lang_code) {
            return Arc::clone(set_arc);
        }
    }

    // Fetch stopwords from NLTK
    let stopwords_set = if let Some(language) = lang_code_to_nltk_language(lang_code) {
        if let Some(stopwords) = NLTK::stopwords(language) {
            stopwords.iter().map(|s| s.to_lowercase()).collect::<HashSet<String>>()
        } else {
            HashSet::new()
        }
    } else {
        // Unsupported language - return empty set
        HashSet::new()
    };

    let built_arc = Arc::new(stopwords_set);

    // Insert with write lock if absent, return the canonical Arc
    if let Ok(mut cache) = STOPWORDS_CACHE.write() {
        let entry = cache.entry(lang_code.to_string()).or_insert_with(|| Arc::clone(&built_arc));
        return Arc::clone(entry);
    }

    // Fallback: return the built Arc if locks were poisoned
    built_arc
}

/// Hash a token string to a 32-bit token ID using MurmurHash3
/// This matches Python's mmh3.hash(feature, signed=True) behavior exactly
fn hash_token(token: &str) -> u32 {
    // MurmurHash3 32-bit with seed 0 (matching Python's mmh3.hash default)
    let hash_unsigned = murmurhash3_x86_32(token.as_bytes(), 0);
    
    // Convert unsigned to signed (matching Python's signed=True)
    let hash_signed = hash_unsigned as i32;
    
    // Map to positive range [0, TOKEN_HASH_RANGE)
    // TOKEN_HASH_RANGE = 2147483647 (2^31 - 1, Mersenne prime)
    // Python does: hash_value % TOKEN_HASH_RANGE
    // For negative values, Rust's rem_euclid ensures positive result
    const TOKEN_HASH_RANGE: i32 = 2147483647;
    (hash_signed.rem_euclid(TOKEN_HASH_RANGE)) as u32
}

/// Hash a (prefix '#'*kind, token) pair without allocating a formatted String
/// kind: number of '#' characters to prefix (1 -> '#', 2 -> '##', 3 -> '###')
fn hash_with_prefix(kind: u8, token: &str) -> u32 {
    // Use static prefixes to avoid per-token loop overhead
    let prefix: &[u8] = match kind {
        1 => b"#",
        2 => b"##",
        3 => b"###",
        _ => b"",
    };
    let mut buf: Vec<u8> = Vec::with_capacity(prefix.len() + token.len());
    buf.extend_from_slice(prefix);
    buf.extend_from_slice(token.as_bytes());
    let hash_unsigned = murmurhash3_x86_32(&buf, 0);
    let hash_signed = hash_unsigned as i32;
    const TOKEN_HASH_RANGE: i32 = 2147483647;
    (hash_signed.rem_euclid(TOKEN_HASH_RANGE)) as u32
}

/// Count token frequencies and compute BM25-style saturated TF weights
/// Returns Vec<(token_id, weight)>
fn get_count_weights(tokens: &[String], base_weight: f32, avgdl: f32) -> Vec<(u32, f32)> {
    let mut token_counts: FxHashMap<String, usize> = FxHashMap::default();
    
    for token in tokens {
        *token_counts.entry(token.clone()).or_insert(0) += 1;
    }
    
    let doc_length = tokens.len() as f32;
    let length_ratio = doc_length / avgdl;
    let length_norm = 1.0 - BM25_B + BM25_B * length_ratio;
    
    let mut token_weights: Vec<(u32, f32)> = Vec::with_capacity(token_counts.len());
    for (token, count) in token_counts.iter() {
        let token_id = hash_token(token);
        let tf = *count as f32;
        let weight = base_weight * bm25_tf_saturation(tf, length_norm);
        token_weights.push((token_id, weight));
    }
    
    token_weights
}

/// Calculate length-based boost for word-level tokens
/// Formula: boost = (token_length / text_length) * max_boost_scaled
/// where max_boost_scaled = max_boost * min(1, 1 / log10(text_length))
const WMTR_LEN_BOOST: f32 = 0.5;

fn calculate_length_boost(token_length: usize, text_length: usize, max_boost_scaled: f32) -> f32 {
    // Skip very short tokens (less informative)
    if token_length < 3 {
        return 0.0;
    }
    
    let length_ratio = token_length as f32 / text_length as f32;
    let boost = length_ratio * max_boost_scaled;
    
    boost
}

/// Count token frequencies and compute BM25-style saturated TF weights using a prefix-kind without allocating prefixed Strings
/// Applies length-based boost for word-level tokens (kind 1 or 2), not trigrams (kind 3)
/// Uses precomputed length_norm for BM25 normalization
fn get_count_weights_prefixed_with_norm(tokens: &[String], base_weight: f32, kind: u8, text_length: usize, length_norm: f32) -> Vec<(u32, f32)> {
    let mut token_counts: FxHashMap<&str, usize> = FxHashMap::default();
    for token in tokens {
        *token_counts.entry(token.as_str()).or_insert(0) += 1;
    }
    
    // Precompute max_boost_scaled once for all tokens (text_length is constant)
    let max_boost_scaled = if kind == 1 || kind == 2 {
        let log10_text_len = (text_length as f32).log10().max(1.0);
        WMTR_LEN_BOOST * (1.0 / log10_text_len).min(1.0)
    } else {
        0.0  // Not used for trigrams, but avoids unused variable warning
    };
    
    let mut token_weights: Vec<(u32, f32)> = Vec::with_capacity(token_counts.len());
    for (token, count) in token_counts.into_iter() {
        let token_id = hash_with_prefix(kind, token);
        let tf = count as f32;
        let base_score = base_weight * bm25_tf_saturation(tf, length_norm);
        
        // Apply length boost only to word-level tokens (kind 1 or 2), not trigrams (kind 3)
        let final_weight = if kind == 1 || kind == 2 {
            let length_boost = calculate_length_boost(token.len(), text_length, max_boost_scaled);
            base_score + length_boost  // Additive boost
        } else {
            base_score
        };
        
        token_weights.push((token_id, final_weight));
    }
    token_weights
}

/// Apply top-k filtering to token weights
/// Returns (indices, values) as two separate vectors
fn top_k(mut token_weights: Vec<(u32, f32)>, k: usize) -> (Vec<u32>, Vec<f32>) {
    if k == 0 || token_weights.is_empty() {
        return (Vec::new(), Vec::new());
    }

    if k >= token_weights.len() {
        // Sort all when k covers entire set
        token_weights.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let indices: Vec<u32> = token_weights.iter().map(|(id, _)| *id).collect();
        let values: Vec<f32> = token_weights.iter().map(|(_, w)| *w).collect();
        return (indices, values);
    }

    // Partially select the k-th largest (partition by descending weight)
    let kth = k - 1; // zero-based index for the k-th element
    let (top_slice, _, _) = token_weights.select_nth_unstable_by(kth, |a, b| b.1.partial_cmp(&a.1).unwrap());

    // Now sort only the top k elements in descending order
    top_slice.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let indices: Vec<u32> = top_slice.iter().take(k).map(|(id, _)| *id).collect();
    let values: Vec<f32> = top_slice.iter().take(k).map(|(_, w)| *w).collect();
    (indices, values)
}

/// Deduplicate sparse vector by summing weights for duplicate indices
fn dedup_sparse(indices: Vec<u32>, values: Vec<f32>) -> (Vec<u32>, Vec<f32>) {
    let mut token_map: FxHashMap<u32, f32> = FxHashMap::default();
    
    for (idx, val) in indices.iter().zip(values.iter()) {
        *token_map.entry(*idx).or_insert(0.0) += val;
    }
    
    let mut result: Vec<(u32, f32)> = token_map.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let indices: Vec<u32> = result.iter().map(|(id, _)| *id).collect();
    let values: Vec<f32> = result.iter().map(|(_, w)| *w).collect();
    
    (indices, values)
}

/// Whitespace tokenizer
/// Splits on whitespace and returns sparse vector
#[pyfunction]
fn tokenize_whitespace(
    text: String,
    lang_code: String,
    top_k_limit: usize,
    use_stopwords: bool,
    avgdl: f32,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    if text.trim().is_empty() {
        return Ok((vec![], vec![]));
    }
    
    // Get stopwords or use empty set
    let stopwords_set_arc = if use_stopwords {
        get_stopwords_cached(&lang_code)
    } else {
        Arc::new(HashSet::new())
    };
    
    // Split on whitespace
    let tokens: Vec<String> = text
        .split_whitespace()
        .filter(|t| !stopwords_set_arc.contains::<str>(t))
        .map(|t| t.to_string())
        .collect();
    
    let token_weights = get_count_weights(&tokens, 1.0, avgdl);
    let (indices, values) = top_k(token_weights, top_k_limit);
    let (indices, values) = dedup_sparse(indices, values);
    
    Ok((indices, values))
}

/// Trigrams tokenizer
/// Generates character-level trigrams with padding
#[pyfunction]
fn tokenize_trigrams(
    text: String,
    top_k_limit: usize,
    avgdl: f32,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    if text.trim().is_empty() {
        return Ok((vec![], vec![]));
    }
    
    // Add space padding (2 spaces on each side to match Python implementation)
    let padded_text = format!("  {}  ", text);
    
    // Extract and count trigrams on-the-fly (Unicode char-based, exact parity)
    let chars: Vec<char> = padded_text.chars().collect();
    let mut trigram_counts: FxHashMap<String, usize> = FxHashMap::default();
    for i in 0..chars.len().saturating_sub(2) {
        let trigram: String = chars[i..i+3].iter().collect();
        *trigram_counts.entry(trigram).or_insert(0) += 1;
    }

    // Compute weights using BM25-style TF saturation
    // Document length for trigrams is the total number of trigrams generated
    let doc_length = chars.len().saturating_sub(2) as f32;
    let length_ratio = doc_length / avgdl;
    let length_norm = 1.0 - BM25_B + BM25_B * length_ratio;
    let mut token_weights: Vec<(u32, f32)> = Vec::with_capacity(trigram_counts.len());
    for (token, count) in trigram_counts.into_iter() {
        let token_id = hash_with_prefix(3, &token);
        let tf = count as f32;
        let weight = 1.0 * bm25_tf_saturation(tf, length_norm);
        token_weights.push((token_id, weight));
    }
    let (indices, values) = top_k(token_weights, top_k_limit);
    let (indices, values) = dedup_sparse(indices, values);
    
    Ok((indices, values))
}

/// Tokenize text using UAX#29 word boundaries, filtering to alphanumeric tokens
/// Uses Unicode Text Segmentation word boundaries and only returns tokens containing letters and numbers
/// Handles apostrophes between letters (e.g., "don't") and periods between letters/numbers (e.g., "U.S.A.", "3.141")
fn tokenize_with_stemming(text: &str, lang_code: &str, stopwords: &std::collections::HashSet<String>) -> Vec<String> {
    // Map language codes to rust-stemmers algorithms
    // rust-stemmers supports 18 languages with Snowball stemmers
    let algorithm_opt = match lang_code.to_lowercase().as_str() {
        "en" => Some(Algorithm::English),
        "fr" => Some(Algorithm::French),
        "de" => Some(Algorithm::German),
        "es" => Some(Algorithm::Spanish),
        "it" => Some(Algorithm::Italian),
        "pt" => Some(Algorithm::Portuguese),
        "nl" => Some(Algorithm::Dutch),
        "sv" => Some(Algorithm::Swedish),
        "no" => Some(Algorithm::Norwegian),
        "da" => Some(Algorithm::Danish),
        "ru" => Some(Algorithm::Russian),
        "tr" => Some(Algorithm::Turkish),
        "hu" => Some(Algorithm::Hungarian),
        "ar" => Some(Algorithm::Arabic),
        "el" => Some(Algorithm::Greek),
        "ro" => Some(Algorithm::Romanian),
        "ta" => Some(Algorithm::Tamil),
        _ => None, // Unsupported language - use basic tokenization
    };
    
    // Use UAX#29 word boundaries (unicode-segmentation implements this correctly)
    // Filter to tokens containing letters and numbers only (handles apostrophes and periods within tokens)
    // Note: input is assumed preprocessed (lowercased) upstream per caller's contract
    let mut tokens: Vec<String> = Vec::new();
    
    // Get word boundaries using UAX#29 (Unicode Text Segmentation)
    // UAX#29 correctly handles apostrophes, periods, commas in numbers, and other punctuation
    for word_boundary in text.split_word_bounds() {
        // Skip empty tokens and pure-punctuation (no letters or numbers)
        if word_boundary.is_empty() || !word_boundary.chars().any(|c| c.is_alphanumeric()) {
            continue;
        }

        // Remove stopwords
        if !stopwords.contains(word_boundary) {
            tokens.push(word_boundary.to_string());
        }
    }
    
    // Apply stemming if available
    if let Some(algorithm) = algorithm_opt {
        let stemmer = Stemmer::create(algorithm);
        tokens.into_iter()
            .map(|token| stemmer.stem(&token).to_string())
            .collect()
    } else {
        // No stemming for unsupported languages
        tokens
    }
}

/// Full-text tokenizer with language-aware stemming
#[pyfunction]
fn tokenize_fulltext(
    text: String,
    lang_code: String,
    top_k_limit: usize,
    use_stopwords: bool,
    avgdl: f32,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    if text.trim().is_empty() {
        return Ok((vec![], vec![]));
    }
    
    // Get stopwords or use empty set
    let stopwords_set_arc = if use_stopwords {
        get_stopwords_cached(&lang_code)
    } else {
        Arc::new(HashSet::new())
    };
    
    // Tokenize with language-aware stemming
    let tokens = tokenize_with_stemming(&text, &lang_code, &*stopwords_set_arc);
    
    let token_weights = get_count_weights(&tokens, 1.0, avgdl);
    let (indices, values) = top_k(token_weights, top_k_limit);
    let (indices, values) = dedup_sparse(indices, values);
    
    Ok((indices, values))
}

/// WMTR (Weighted Multilevel Token Representation) tokenizer
/// Combines whitespace tokens, language-aware tokens (with stemming), and trigrams with different weights
#[pyfunction]
fn tokenize_wmtr(
    text: String,
    lang_code: String,
    top_k_limit: usize,
    word_weight_percentage: u32,
    use_stopwords: bool,
    avgdl: f32,
    trigram_weight: f32,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    if text.trim().is_empty() {
        return Ok((vec![], vec![]));
    }
    
    // Get stopwords or use empty set
    let stopwords_set_arc = if use_stopwords {
        get_stopwords_cached(&lang_code)
    } else {
        Arc::new(HashSet::new())
    };
    
    // 1. Whitespace tokens (special-character preserving)
    let whitespace_tokens: Vec<String> = text
        .split_whitespace()
        .filter(|t| !stopwords_set_arc.contains::<str>(t))
        .map(|t| t.to_string())
        .collect();
    
    // 2. Language-aware tokens (with stemming support)
    let lang_tokens = tokenize_with_stemming(&text, &lang_code, &*stopwords_set_arc);
    
    // Filter whitespace tokens to remove those already in lang_tokens (borrowed set to avoid clones)
    let lang_tokens_set: FxHashSet<&str> = lang_tokens.iter().map(|s| s.as_str()).collect();
    let whitespace_tokens: Vec<String> = whitespace_tokens
        .into_iter()
        .filter(|t| !lang_tokens_set.contains(t.as_str()))
        .collect();
    
    // 3. Trigrams from text with stopwords removed (Unicode char-based)
    // Remove stopwords from text before generating trigrams to reduce noise
    let text_without_stopwords: String = text
        .split_whitespace()
        .filter(|word| !stopwords_set_arc.contains::<str>(word))
        .collect::<Vec<&str>>()
        .join(" ");
    let padded_text = format!(" {} ", text_without_stopwords);
    let chars: Vec<char> = padded_text.chars().collect();
    let mut trigram_counts: FxHashMap<String, usize> = FxHashMap::default();
    for i in 0..chars.len().saturating_sub(2) {
        let trigram: String = chars[i..i+3].iter().collect();
        *trigram_counts.entry(trigram).or_insert(0) += 1;
    }
    
    // Get text length in characters for length-based boosting
    let text_length = text.chars().count();
    
    // Calculate total document token count for BM25 length normalization
    // Use lang_tokens as the primary tokenization (they include stemming)
    let total_doc_length = lang_tokens.len() as f32;
    let length_ratio = total_doc_length / avgdl;
    let length_norm = 1.0 - BM25_B + BM25_B * length_ratio;
    
    // Apply different base weights and prefixes for each token type
    // Pass text_length for length-based boosting (applied to word-level tokens only)
    // Pass total_doc_length for BM25 normalization (same for all components)
    let ws_weights = get_count_weights_prefixed_with_norm(
        &whitespace_tokens,
        WMTR_WEIGHT_WHITESPACE,
        1,
        text_length,
        length_norm,
    );
    let lang_weights = get_count_weights_prefixed_with_norm(
        &lang_tokens,
        WMTR_WEIGHT_LANG,
        2,
        text_length,
        length_norm,
    );
    // Build trigram weights from counts map using BM25-style TF saturation
    // Use the same length_norm as word-level tokens
    let mut trigram_weights: Vec<(u32, f32)> = Vec::with_capacity(trigram_counts.len());
    for (token, count) in trigram_counts.into_iter() {
        let token_id = hash_with_prefix(3, &token);
        let tf = count as f32;
        let weight = trigram_weight * bm25_tf_saturation(tf, length_norm);
        trigram_weights.push((token_id, weight));
    }
    
    // Combine word weights (whitespace + lang)
    let mut word_weights = ws_weights;
    word_weights.extend(lang_weights);
    
    // Calculate how many slots for words vs trigrams
    let word_k = ((top_k_limit as f32) * (word_weight_percentage as f32 / 100.0)) as usize;
    let trigram_k = top_k_limit.saturating_sub(word_k);
    
    // Get top-k for each type
    let (mut word_indices, mut word_values) = top_k(word_weights, word_k);
    let (trigram_indices, trigram_values) = top_k(trigram_weights, trigram_k);
    
    // Combine and deduplicate
    word_indices.extend(trigram_indices);
    word_values.extend(trigram_values);
    
    let (indices, values) = dedup_sparse(word_indices, word_values);
    
    Ok((indices, values))
}

/// Test function to return raw tokens (no stemming, no stopwords)
#[pyfunction]
fn tokenize_raw(text: String) -> PyResult<Vec<String>> {
    let empty_stopwords: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Use unsupported language code to skip stemming
    let tokens = tokenize_with_stemming(&text, "xx", &empty_stopwords);
    Ok(tokens)
}

/// Debug function to see what split_word_bounds returns
#[pyfunction]
fn debug_word_bounds(text: String) -> PyResult<Vec<String>> {
    let boundaries: Vec<String> = text.split_word_bounds().map(|s| s.to_string()).collect();
    Ok(boundaries)
}

/// Debug function to export stemmed tokens
#[pyfunction]
fn debug_stemmed_tokens(
    text: String,
    lang_code: String,
) -> PyResult<Vec<String>> {
    let stopwords_set_arc = get_stopwords_cached(&lang_code);
    let tokens = tokenize_with_stemming(&text, &lang_code, &*stopwords_set_arc);
    Ok(tokens)
}

/// Debug function to trace through get_count_weights
#[pyfunction]
fn debug_count_weights(
    text: String,
    lang_code: String,
    avgdl: f32,
) -> PyResult<Vec<(String, u32, f32)>> {
    let stopwords_set_arc = get_stopwords_cached(&lang_code);
    let tokens = tokenize_with_stemming(&text, &lang_code, &*stopwords_set_arc);
    let token_weights = get_count_weights(&tokens, 1.0, avgdl);
    
    // Return (token_string, token_id, weight) for debugging
    let mut result = Vec::new();
    for (token_id, weight) in token_weights {
        // Try to find which token this ID corresponds to
        for token in &tokens {
            if hash_token(token) == token_id {
                result.push((token.clone(), token_id, weight));
                break;
            }
        }
    }
    Ok(result)
}

/// Python module definition
#[pymodule]
fn amgix_analyzers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_whitespace, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_trigrams, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_fulltext, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_wmtr, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_raw, m)?)?;
    m.add_function(wrap_pyfunction!(debug_word_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(debug_stemmed_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(debug_count_weights, m)?)?;
    Ok(())
}

