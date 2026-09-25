#!/usr/bin/env python3
"""
sentiment_analysis.py
=====================
A reusable sentimentAnalysis() function returning positive / negative / neutral.

Import it, or run it on a CSV from the command line.

QUICK USE
---------
    from sentiment_analysis import sentimentAnalysis

    sentimentAnalysis("Claim settled in four days, very smooth.")
    # -> {'label': 'positive', 'score': 0.5, 'confidence': 0.82, 'backend': 'transformer'}

    sentimentAnalysis("Claim rejected without reason.")['label']
    # -> 'negative'

    # A whole column at once (much faster than looping):
    df['sentiment_label'] = sentimentAnalysis(df['review_text'], as_label=True)

BACKENDS
--------
  transformer  (default if installed) nlptown/bert-base-multilingual-uncased-
               sentiment. Handles negation, word order, and Hinglish. This is
               what took agreement from 28.6% to 90.3% on this dataset.
  vader        Lexicon fallback. Fast, no download, but unreliable on
               insurance vocabulary - it reads "claim", "delay" and "no issues"
               as negative. Use only for a quick plumbing check.

The backend is chosen automatically: transformer if available, else VADER.
Force one with backend="vader" or backend="transformer".

CLI
---
    python sentiment_analysis.py --in reviews.csv
    python sentiment_analysis.py --in reviews.csv --text-col Review_text
    python sentiment_analysis.py --in reviews.csv --backend vader
    python sentiment_analysis.py --text "claim was rejected again"

SETUP
-----
    pip install pandas
    pip install transformers torch      # recommended
    pip install vaderSentiment          # fallback
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Dict, Iterable, List, Optional, Union

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sentiment")

LABELS = ("positive", "neutral", "negative")

# Module-level cache so the model loads once, not per call
_ENGINE = None
_ENGINE_NAME = None


# --------------------------------------------------------------------------- #
# Engine setup
# --------------------------------------------------------------------------- #

def _load_engine(backend: Optional[str] = None):
    """Load and cache a sentiment backend. Returns (engine, name)."""
    global _ENGINE, _ENGINE_NAME

    if _ENGINE is not None and (backend is None or backend == _ENGINE_NAME):
        return _ENGINE, _ENGINE_NAME

    order = [backend] if backend else ["transformer", "vader"]

    for name in order:
        if name == "transformer":
            try:
                from transformers import pipeline
                log.info("Loading transformer sentiment model "
                         "(first run downloads ~700MB)...")
                eng = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    truncation=True, max_length=512)
                _ENGINE, _ENGINE_NAME = eng, "transformer"
                return _ENGINE, _ENGINE_NAME
            except Exception as e:
                if backend == "transformer":
                    raise RuntimeError(
                        "transformer backend requested but unavailable: %s\n"
                        "Install with: pip install transformers torch" % e)
                log.warning("Transformer unavailable (%s). Falling back to VADER.", e)

        elif name == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                _ENGINE, _ENGINE_NAME = SentimentIntensityAnalyzer(), "vader"
                log.info("Using VADER. Note: unreliable on insurance vocabulary.")
                return _ENGINE, _ENGINE_NAME
            except ImportError:
                if backend == "vader":
                    raise RuntimeError("pip install vaderSentiment")

    raise RuntimeError("No sentiment backend available. Install either:\n"
                       "  pip install transformers torch   (recommended)\n"
                       "  pip install vaderSentiment       (fallback)")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")


def _prep(text) -> str:
    if text is None:
        return ""
    t = str(text)
    if t.lower() in ("nan", "none"):
        return ""
    t = _URL.sub("", t)
    return _WS.sub(" ", t).strip()


def _score_one(text: str, engine, name: str,
               neutral_band: float) -> Dict[str, object]:
    """Score a single string. Returns label / score / confidence / backend."""
    text = _prep(text)
    if not text:
        return {"label": "neutral", "score": 0.0,
                "confidence": 0.0, "backend": name}

    if name == "vader":
        s = engine.polarity_scores(text)["compound"]
        label = ("positive" if s >= neutral_band
                 else "negative" if s <= -neutral_band else "neutral")
        return {"label": label, "score": round(s, 4),
                "confidence": round(min(abs(s), 1.0), 3), "backend": name}

    # transformer: model emits "1 star" .. "5 stars"
    res = engine(text[:2000])[0]
    stars = int(str(res["label"])[0])
    score = round((stars - 3) / 2, 4)        # map 1..5 -> -1..1
    label = ("positive" if stars >= 4
             else "negative" if stars <= 2 else "neutral")
    return {"label": label, "score": score,
            "confidence": round(float(res.get("score", 0.0)), 3),
            "backend": name}


def sentimentAnalysis(
    text: Union[str, Iterable, None],
    backend: Optional[str] = None,
    as_label: bool = False,
    neutral_band: float = 0.05,
    batch_size: int = 32,
) -> Union[Dict, List, str]:
    """
    Classify text as positive, negative or neutral.

    Parameters
    ----------
    text : str, or an iterable of str (list / pandas Series)
        The text to classify.
    backend : "transformer" | "vader" | None
        None picks transformer if installed, else VADER.
    as_label : bool
        True returns just the label string(s) instead of full dicts.
        Convenient for assigning straight to a DataFrame column.
    neutral_band : float
        VADER only. Scores within +/- this of zero count as neutral.
    batch_size : int
        Transformer only. Larger is faster but uses more memory.

    Returns
    -------
    For a single string : dict with keys label, score, confidence, backend
                          (or just the label string if as_label=True)
    For an iterable     : list of those, in the same order as the input

    Examples
    --------
    >>> sentimentAnalysis("Claim settled quickly, no problems.")['label']
    'positive'
    >>> sentimentAnalysis(["good service", "terrible delay"], as_label=True)
    ['positive', 'negative']
    """
    engine, name = _load_engine(backend)

    # --- single string ------------------------------------------------- #
    if text is None or isinstance(text, str):
        out = _score_one(text or "", engine, name, neutral_band)
        return out["label"] if as_label else out

    # --- iterable ------------------------------------------------------ #
    try:
        items = list(text)
    except TypeError:
        raise TypeError("sentimentAnalysis expects a string or an iterable of "
                        "strings, got %s" % type(text).__name__)

    prepped = [_prep(t) for t in items]

    # Transformers can score a batch in one pass - much faster than looping
    if name == "transformer":
        results: List[Dict[str, object]] = []
        for i in range(0, len(prepped), batch_size):
            chunk = prepped[i:i + batch_size]
            idx = [j for j, t in enumerate(chunk) if t]
            texts = [chunk[j][:2000] for j in idx]

            chunk_out = [{"label": "neutral", "score": 0.0,
                          "confidence": 0.0, "backend": name}
                         for _ in chunk]
            if texts:
                preds = engine(texts)
                for j, p in zip(idx, preds):
                    stars = int(str(p["label"])[0])
                    chunk_out[j] = {
                        "label": ("positive" if stars >= 4
                                  else "negative" if stars <= 2 else "neutral"),
                        "score": round((stars - 3) / 2, 4),
                        "confidence": round(float(p.get("score", 0.0)), 3),
                        "backend": name,
                    }
            results.extend(chunk_out)
    else:
        results = [_score_one(t, engine, name, neutral_band) for t in prepped]

    return [r["label"] for r in results] if as_label else results


# Snake_case alias, since most Python code expects it
sentiment_analysis = sentimentAnalysis


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _resolve(df, name: str) -> Optional[str]:
    if name in df.columns:
        return name
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    t = norm(name)
    for c in df.columns:
        if norm(c) == t:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default=None, help="CSV to process")
    ap.add_argument("--out", dest="outfile", default=None)
    ap.add_argument("--text-col", default="review_text")
    ap.add_argument("--backend", choices=["transformer", "vader"], default=None)
    ap.add_argument("--text", default=None, help="Score one string and exit")
    args = ap.parse_args()

    if args.text:
        print(sentimentAnalysis(args.text, backend=args.backend))
        return

    if not args.infile:
        ap.error("pass --in <csv> or --text '<string>'")

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas required:  pip install pandas")

    df = pd.read_csv(args.infile)
    col = _resolve(df, args.text_col)
    if col is None:
        sys.exit("Column '%s' not found.\nAvailable: %s"
                 % (args.text_col, list(df.columns)))
    if col != args.text_col:
        log.info("Matched '%s' -> '%s'", args.text_col, col)

    log.info("Scoring %d rows...", len(df))
    results = sentimentAnalysis(df[col].tolist(), backend=args.backend)

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df["sentiment_confidence"] = [r["confidence"] for r in results]
    df["sentiment_backend"] = [r["backend"] for r in results]

    out = args.outfile or args.infile.replace(".csv", "_sentiment.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("Wrote %s", out)

    print()
    print(df["sentiment_label"].value_counts().to_string())
    print()
    pct = df["sentiment_label"].value_counts(normalize=True).mul(100).round(1)
    for lab in LABELS:
        if lab in pct.index:
            print("%-9s %5.1f%%" % (lab, pct[lab]))

    # Agreement check against star ratings, if present
    rc = _resolve(df, "rating_5")
    if rc and df[rc].notna().any():
        d = df[df[rc].notna()]
        truth = d[rc].apply(lambda r: "positive" if r >= 4
                            else "negative" if r <= 2 else "neutral")
        agree = (truth == d["sentiment_label"]).mean() * 100
        print("\nAgreement with star ratings: %.1f%% of %d rated rows"
              % (agree, len(d)))
        if agree < 70:
            print(">> LOW. Switch to --backend transformer before using these.")

    # Brand breakdown, if present
    bc = _resolve(df, "brand")
    if bc:
        try:
            import pandas as pd
            print("\nSentiment by brand (%):")
            t = (pd.crosstab(df[bc], df["sentiment_label"], normalize="index")
                   .mul(100).round(1))
            cols = [c for c in LABELS if c in t.columns]
            print(t.reindex(columns=cols).to_string())
        except Exception:
            pass


if __name__ == "__main__":
    main()
