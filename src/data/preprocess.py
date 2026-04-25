"""Text preprocessing for e-commerce queries.

Handles normalization, tokenization, abbreviation expansion,
cleaning of noisy user input, and DataFrame-level batch processing.

Typical usage
-------------
>>> from src.data.preprocess import run_pipeline
>>> df = run_pipeline()          # reads config, writes data/processed/
>>> df.head()

Or use the individual helpers directly::

    from src.data.preprocess import clean_text, tokenize
    clean_text("Nike  Running   Shoes")   # → "nike running shoes"
    tokenize("nike running shoes")        # → ["nike", "running", "shoes"]
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import spacy
from spacy.language import Language

from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger(__name__)

# Keys may contain slashes (``w/``, ``w/o``) so we use regex-based expansion
# rather than naïve ``str.split()``.

ABBREVIATIONS: Dict[str, str] = {
    r"(?<!\w)w/o(?!\w)": "without",      # w/o → without (before w/ so it matches first)
    r"(?<!\w)w/(?!\w)": "with",           # w/ → with
    r"\bqty\b": "quantity",
    r"\bpcs\b": "pieces",
    r"\bpc\b": "piece",
    r"\bsz\b": "size",
    r"\bsm\b": "small",
    r"\bmd\b": "medium",
    r"\blg\b": "large",
    r"\bxl\b": "extra large",
    r"\bblk\b": "black",
    r"\bwht\b": "white",
    r"\bgld\b": "gold",
    r"\bslv\b": "silver",
    r"\borg\b": "organic",
    r"\bss\b": "stainless steel",
    r"\bbt\b": "bluetooth",
    r"\busb-c\b": "usb type c",
    r"\bpkg\b": "package",
    r"\bct\b": "count",
    r"\boz\b": "ounce",
    r"\bft\b": "feet",
    r"\bin\b": "inch",
    r"\blb\b": "pound",
    r"\blbs\b": "pounds",
}

# Pre-compile the patterns once for speed (longest patterns first so
# ``w/o`` is tried before ``w/``).
_ABBREV_COMPILED: List[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0]))
]


_nlp: Optional[Language] = None


def _load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load a spaCy model, downloading if needed."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model '{model_name}' …")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


def get_nlp(model_name: str = "en_core_web_sm") -> Language:
    """Get or initialize the spaCy tokenizer."""
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model(model_name)
    return _nlp




def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII-friendly equivalents."""
    # Apply targeted replacements BEFORE NFKD (which decomposes ™ → TM, etc.)
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dashes
        "\u2026": "...",                # ellipsis
        "\u00d7": "x",                 # multiplication sign
        "\u00ae": "",                   # ®
        "\u2122": "",                   # ™
        "\u00a0": " ",                  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text)
    return text


def expand_abbreviations(
    text: str,
    compiled_patterns: Optional[List[tuple[re.Pattern, str]]] = None,
) -> str:
    """Expand common e-commerce abbreviations using regex word-boundary matching.

    Examples::

        >>> expand_abbreviations("w/ charger")
        'with charger'
        >>> expand_abbreviations("w/o box")
        'without box'
        >>> expand_abbreviations("sz large")
        'size large'
    """
    if compiled_patterns is None:
        compiled_patterns = _ABBREV_COMPILED
    for pattern, replacement in compiled_patterns:
        text = pattern.sub(replacement, text)
    return text


def clean_text(
    text: str,
    lowercase: bool = True,
    strip_special: bool = True,
    expand_abbrevs: bool = True,
) -> str:
    """Full text cleaning pipeline for a single string.

    Order of operations:
      1. Unicode normalization
      2. Abbreviation expansion  (before lowercasing so patterns stay simple)
      3. Strip special characters (keep alphanumeric, spaces, hyphens, dots, slashes)
      4. Collapse whitespace
      5. Lowercase

    Args:
        text: Raw text string.
        lowercase: Convert to lowercase.
        strip_special: Remove characters outside ``[a-zA-Z0-9 \\-./]``.
        expand_abbrevs: Expand abbreviations.

    Returns:
        Cleaned string (empty string for invalid input).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)

    if expand_abbrevs:
        text = expand_abbreviations(text)

    if strip_special:
        # Keep alphanumeric, spaces, hyphens, periods, slashes (for sizes like 1/2)
        text = re.sub(r"[^a-zA-Z0-9\s\-./]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text




def tokenize(text: str, use_spacy: bool = True) -> List[str]:
    """Tokenize text into a list of tokens.

    Args:
        text: Cleaned text to tokenize.
        use_spacy: If ``True`` use spaCy tokenizer, otherwise whitespace split.

    Returns:
        List of token strings.
    """
    if not text:
        return []
    if use_spacy:
        nlp = get_nlp()
        doc = nlp(text)
        return [token.text for token in doc if not token.is_space]
    return text.split()




def preprocess_queries(
    queries: List[str],
    lowercase: bool = True,
    strip_special: bool = True,
    expand_abbrevs: bool = True,
    min_length: int = 2,
    max_length: int = 20,
) -> List[dict]:
    """Preprocess a batch of queries and filter by token length.

    Args:
        queries: List of raw query strings.
        lowercase: Convert to lowercase.
        strip_special: Strip special chars.
        expand_abbrevs: Expand abbreviations.
        min_length: Minimum token count to keep.
        max_length: Maximum token count to keep.

    Returns:
        List of dicts with ``original``, ``cleaned``, ``tokens``, ``num_tokens``.
    """
    results = []
    for query in queries:
        cleaned = clean_text(query, lowercase, strip_special, expand_abbrevs)
        tokens = tokenize(cleaned, use_spacy=False)  # whitespace for speed

        if min_length <= len(tokens) <= max_length:
            results.append({
                "original": query,
                "cleaned": cleaned,
                "tokens": tokens,
                "num_tokens": len(tokens),
            })

    logger.info(
        f"Preprocessed {len(queries):,} queries → {len(results):,} kept "
        f"(filtered by length [{min_length}, {max_length}])"
    )
    return results




def preprocess_dataframe(
    df: pd.DataFrame,
    query_col: str = "query",
    lowercase: bool = True,
    strip_special: bool = True,
    expand_abbrevs: bool = True,
    remove_duplicates: bool = True,
    min_length: int = 2,
    max_length: int = 20,
) -> pd.DataFrame:
    """Clean, tokenize and filter a DataFrame of queries.

    Adds columns:
      - ``query_clean``: normalized query string
      - ``query_tokens``: list of tokens (whitespace tokenization)
      - ``num_tokens``: token count

    Args:
        df: Input DataFrame (must contain *query_col*).
        query_col: Name of the raw query column.
        lowercase / strip_special / expand_abbrevs: Cleaning flags.
        remove_duplicates: Drop duplicate cleaned queries.
        min_length / max_length: Token-count filter bounds.

    Returns:
        Cleaned, filtered DataFrame (copy — original is untouched).
    """
    n_start = len(df)
    out = df.copy()

    logger.info(f"Preprocessing {n_start:,} rows …")

    # 1. Clean
    out["query_clean"] = out[query_col].apply(
        lambda q: clean_text(q, lowercase, strip_special, expand_abbrevs)
    )

    # 2. Drop rows where cleaning produced an empty string
    out = out[out["query_clean"].str.len() > 0].copy()
    n_after_clean = len(out)
    if n_after_clean < n_start:
        logger.info(f"  Dropped {n_start - n_after_clean:,} empty-after-clean rows")

    # 3. Tokenize (whitespace — fast; spaCy used downstream for NER)
    out["query_tokens"] = out["query_clean"].str.split()
    out["num_tokens"] = out["query_tokens"].str.len()

    # 4. Filter by token length
    mask = out["num_tokens"].between(min_length, max_length)
    n_before_len = len(out)
    out = out[mask].copy()
    logger.info(
        f"  Length filter [{min_length}, {max_length}]: "
        f"{n_before_len:,} → {len(out):,}"
    )

    # 5. Deduplicate on cleaned query text
    if remove_duplicates:
        n_before_dedup = len(out)
        out = out.drop_duplicates(subset="query_clean").reset_index(drop=True)
        logger.info(
            f"  Deduplicated: {n_before_dedup:,} → {len(out):,}"
        )

    logger.info(f"  Final: {len(out):,} rows (from {n_start:,})")
    return out


def clean_product_metadata(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Light cleaning of product metadata columns.

    Applies unicode normalization and whitespace collapsing to the given
    text columns.  Does **not** lowercase or strip special chars (product
    metadata like brand names should preserve casing for downstream
    matching).

    Args:
        df: DataFrame with product metadata (list-valued columns are OK).
        text_cols: Columns to clean.  Defaults to standard ESCI metadata cols.

    Returns:
        DataFrame with cleaned metadata (copy).
    """
    if text_cols is None:
        text_cols = [
            "product_title",
            "product_brand",
            "product_color",
            "product_bullet_point",
            "product_description",
        ]

    out = df.copy()
    for col in text_cols:
        if col not in out.columns:
            continue

        def _clean_cell(val):
            """Clean a single cell — handles str and list[str]."""
            if isinstance(val, list):
                return [_clean_str(v) for v in val if isinstance(v, str)]
            if isinstance(val, str):
                return _clean_str(val)
            return val

        out[col] = out[col].apply(_clean_cell)

    return out


def _clean_str(s: str) -> str:
    """Light string cleaning: unicode normalize + collapse whitespace."""
    s = normalize_unicode(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s




def run_pipeline(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    config_path: str | Path = "configs/data_config.yaml",
    locale: str = "us",
) -> pd.DataFrame:
    """Run the full preprocessing pipeline: load → clean → save.

    Reads settings from ``data_config.yaml``, loads merged ESCI data via
    :func:`src.data.loader.load_dataset`, preprocesses queries and product
    metadata, and writes the result to ``data/processed/queries_preprocessed.parquet``.

    Args:
        raw_dir: Raw data directory (passed to loader).
        processed_dir: Where to write the output parquet file.
        config_path: Path to the data configuration YAML.
        locale: Locale filter.

    Returns:
        The preprocessed DataFrame.
    """
    from src.data.loader import load_dataset

    # ── Load config ────────────────────────────────────────────────────────
    config_path = Path(config_path)
    if config_path.exists():
        cfg = load_yaml_config(config_path).get("data", {})
        preproc = cfg.get("preprocessing", {})
    else:
        logger.warning(f"Config not found at {config_path}, using defaults.")
        cfg = {}
        preproc = {}

    lowercase = preproc.get("lowercase", True)
    strip_special = preproc.get("strip_special_chars", True)
    expand_abbrevs = preproc.get("expand_abbreviations", True)
    remove_dups = preproc.get("remove_duplicates", True)
    min_len = cfg.get("min_query_length", 2)
    max_len = cfg.get("max_query_length", 20)

    # ── Load data (aggregated: one row per unique query) ───────────────────
    logger.info("Loading dataset …")
    df = load_dataset(raw_dir=raw_dir, locale=locale, aggregate=True)
    logger.info(f"Loaded {len(df):,} unique queries")

    # ── Preprocess queries ─────────────────────────────────────────────────
    df = preprocess_dataframe(
        df,
        query_col="query",
        lowercase=lowercase,
        strip_special=strip_special,
        expand_abbrevs=expand_abbrevs,
        remove_duplicates=remove_dups,
        min_length=min_len,
        max_length=max_len,
    )

    # ── Clean product metadata ─────────────────────────────────────────────
    df = clean_product_metadata(df)

    # ── Save ───────────────────────────────────────────────────────────────
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "queries_preprocessed.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved preprocessed data → {out_path}  ({len(df):,} rows)")

    return df


