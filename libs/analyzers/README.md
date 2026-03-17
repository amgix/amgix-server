# Amgix Analyzers

High-performance Rust implementations of Amgix tokenizers, exposed to Python via PyO3.
Module name: `amgix_analyzers`

## Tokenizers

- **WMTR** (Weighted Multilevel Token Representation): Combines whitespace, language-aware, and trigram tokens
- **Full-text**: Language-aware tokenization with stopword filtering
- **Trigrams**: Character-level trigram tokenization
- **Whitespace**: Simple whitespace-based tokenization

## Build & Install

Requires Rust toolchain (cargo/rustc) and Python 3.8+.

```bash
# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Build and install in development mode
maturin develop

# Build an optimized wheel (optional)
maturin build --release
```

## Usage

```python
import amgix_analyzers as aa

# Whitespace tokenization (now requires lang_code for cached stopwords)
indices, values = aa.tokenize_whitespace(
    text="example text",
    lang_code="en",
    stopwords=["the", "a"],
    top_k_limit=100,
)

# Trigrams
indices, values = aa.tokenize_trigrams(
    text="example text",
    top_k_limit=100,
)

# Full-text (with language-aware stemming)
indices, values = aa.tokenize_fulltext(
    text="example text",
    lang_code="en",  # ISO 639-1
    stopwords=["the", "a"],
    top_k_limit=100,
)

# WMTR (with language-aware stemming)
indices, values = aa.tokenize_wmtr(
    text="example text",
    lang_code="en",
    stopwords=["the", "a"],
    top_k_limit=100,
    word_weight_percentage=80,
)
```

## Supported Languages for Stemming

The following languages support full stemming via `rust-stemmers` (Snowball stemmers):
- Arabic (`ar`)
- Danish (`da`)
- Dutch (`nl`)
- English (`en`)
- French (`fr`)
- German (`de`)
- Greek (`el`)
- Hungarian (`hu`)
- Italian (`it`)
- Norwegian (`no`)
- Portuguese (`pt`)
- Romanian (`ro`)
- Russian (`ru`)
- Spanish (`es`)
- Swedish (`sv`)
- Tamil (`ta`)
- Turkish (`tr`)

For all other languages, tokenization uses Unicode word segmentation with stopword filtering (no stemming), which matches Whoosh's `StandardAnalyzer` behavior.

## Notes

- MurmurHash3 matches Python `mmh3.hash(feature, signed=True)` exactly.
- Stopwords are cached per `lang_code` across calls.
- WMTR/full-text use Unicode regex tokenization + optional stemming (via `rust-stemmers`).
- Trigrams are Unicode char-based with exact parity vs Python, now counted on-the-fly for performance.
- Internals avoid unnecessary allocations (prefix hashing without `format!`, partial top-k selection).

