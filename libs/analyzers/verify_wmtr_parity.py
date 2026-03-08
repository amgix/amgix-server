#!/usr/bin/env python3
"""
Verify WMTR parity between Python implementation (wmtr.WMTRVector)
and Rust implementation (amgix_analyzers.tokenize_wmtr), using the
actual Python code path (no reimplementation).
"""

import os
import sys
from types import SimpleNamespace
import importlib
from typing import List, Tuple
from stopwordsiso import stopwords as sw_iso
import amgix_analyzers
import mmh3, math
from whoosh.analysis import StandardAnalyzer, LanguageAnalyzer
from whoosh.lang import languages
import time

# Import the actual Python implementation via package import (to resolve relative imports)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_SRC = os.path.join(REPO_ROOT, "src")
# Ensure both project root (so 'src.*' absolute imports work) and src/ (for direct core.* imports) are present
for p in (REPO_ROOT, PY_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

wmtr_module = importlib.import_module("src.core.vector.wmtr")
WMTRVector = wmtr_module.WMTRVector


def python_wmtr_indices_via_impl(text: str, top_k: int, lang_default: str) -> Tuple[List[int], List[float]]:
    """Call the actual WMTRVector._get_sparse_vector with a minimal config
    to mirror production code.
    """
    cfg = SimpleNamespace(
        top_k=top_k,
        language_detect=False,
        language_confidence=0.0,
        language_default_code=lang_default,
        name="wmtr",
    )
    vec = WMTRVector()
    return vec._get_sparse_vector(cfg, text)


def python_wmtr_debug_via_impl(text: str, top_k: int, lang_default: str) -> Tuple[List[int], List[float]]:
    return python_wmtr_indices_via_impl(text, top_k, lang_default)


def build_python_token_labels(
    text: str,
    lang_code: str,
    stoplist: List[str],
    top_k_limit: int,
    word_weight_percentage: int,
) -> dict:
    # Whitespace tokens with external stop filtering
    ws_tokens = [t for t in text.split() if t not in stoplist]
    # Language tokens per wmtr.py
    if lang_code in languages:
        analyzer = LanguageAnalyzer(lang_code)
        lang_tokens = [t.text for t in analyzer(text)]
        if stoplist:
            lang_tokens = [tok for tok in lang_tokens if tok not in stoplist]
    else:
        analyzer = StandardAnalyzer(stoplist=stoplist)
        lang_tokens = [t.text for t in analyzer(text)]
    # Remove overlap
    lang_set = set(lang_tokens)
    ws_tokens = [t for t in ws_tokens if t not in lang_set]
    # Trigrams
    padded = f" {text} "
    trigrams = [padded[i : i + 3] for i in range(len(padded) - 2)]

    def items(tokens, base, prefix):
        counts = {}
        for tok in tokens:
            key = prefix + tok
            counts[key] = counts.get(key, 0) + 1
        out = []
        for key, cnt in counts.items():
            tid = mmh3.hash(key, signed=True) % 2147483647
            w = base * math.log(1 + cnt)
            out.append((tid, w, key))
        return out

    ws_items = items(ws_tokens, 1.0, '#')
    lang_items = items(lang_tokens, 2.0, '##')
    tri_items = items(trigrams, 0.25, '###')
    word_items = ws_items + lang_items

    word_k = int(top_k_limit * (word_weight_percentage / 100.0))
    trigram_k = max(0, top_k_limit - word_k)
    word_items.sort(key=lambda x: x[1], reverse=True)
    tri_items.sort(key=lambda x: x[1], reverse=True)
    w_top = word_items[:word_k]
    t_top = tri_items[:trigram_k]

    label_map = {}
    for tid, _w, key in w_top + t_top:
        label_map.setdefault(tid, key)
    return label_map


def run():
    tests = [
        "test",
        "example",
        "1/4-20",
        "LP100L9.75",
        "#token",
        "##language",
        "###abc",
        "special-char_token",
        "UPPERCASE",
        "with spaces",
        "日本語",
        "emoji😀",
        "The quick brown fox jumps over the lazy dog",
        "Product LP100L9.75 with 1/4-20 NPT fitting",
        "Big-time under_score 3.141 rocks!",
        "denoise, de-noise, and re.noise",
        # Long text case
        (
            "lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            * 1000
        ).strip(),
    ]

    # Params
    lang_code = "en"
    stoplist = list(sw_iso(lang_code)) or []
    top_k_limit = 10000  # very large to avoid truncation effects
    word_weight_percentage = 90

    print("=" * 80)
    print("Verifying WMTR parity (Python vs Rust)")
    print("=" * 80)

    failures = 0
    for text in tests:
        # Normalize input to mirror preprocessing: trim + lowercase
        norm = text.strip().lower()
        # Python indices/values via actual wmtr implementation
        t0 = time.perf_counter()
        py_idx, py_vals = python_wmtr_debug_via_impl(norm, top_k_limit, lang_code)
        t_py = (time.perf_counter() - t0) * 1000.0

        # Rust indices
        t1 = time.perf_counter()
        rs_idx, _ = amgix_analyzers.tokenize_wmtr(
            norm, lang_code, top_k_limit, word_weight_percentage, use_stopwords=True
        )
        t_rs = (time.perf_counter() - t1) * 1000.0

        py_set = set(py_idx)
        rs_set = set(rs_idx)

        ok = py_set == rs_set
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] {text[:50]}  (py: {t_py:.2f} ms, rs: {t_rs:.2f} ms)")
        if not ok:
            failures += 1
            only_py_ids = list(py_set - rs_set)
            only_rs_ids = list(rs_set - py_set)
            py_labels = build_python_token_labels(
                norm, lang_code, stoplist, top_k_limit, word_weight_percentage
            )
            print(f"  Only in Python ({len(only_py_ids)}):")
            for tid in only_py_ids[:20]:
                print(f"    id={tid} token={py_labels.get(tid, '?')}")
            if len(only_py_ids) > 20:
                print(f"    ... and {len(only_py_ids)-20} more")
            print(f"  Only in Rust ({len(only_rs_ids)}):")
            for tid in only_rs_ids[:20]:
                print(f"    id={tid} token={py_labels.get(tid, '?')}")
            if len(only_rs_ids) > 20:
                print(f"    ... and {len(only_rs_ids)-20} more")

    print("-" * 80)
    if failures:
        print(f"Completed with {failures} mismatches.")
    else:
        print("All test cases match.")


if __name__ == "__main__":
    run()


