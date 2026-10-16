"""Turn a prompt into numbers a classifier can learn from.

Two kinds of features, combined side by side:

1. **Hand-crafted features** — things we believe signal difficulty: how many reasoning verbs
   ("why", "compare"), how many constraints ("exactly 6 lines"), whether a context passage
   is attached, how structured the requested output is, and so on.
2. **TF-IDF of the instruction** — every word / word pair becomes a column, weighted by how
   distinctive it is (frequent in this prompt, rare across all prompts). The model then
   learns which words push toward which tier.

Both look at the *instruction*, not the attached context: a passage about Taylor Swift says
nothing about difficulty, while "summarise" vs "why" says a lot. Context only contributes
its length and presence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

CONTEXT_MARKER = "\n\nContext:\n"
CONTEXT_MIN_WORDS = 40  # a paragraph this long, next to a shorter one, is treated as context

_TOKEN = re.compile(r"\w+|[^\w\s]")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _cues(*patterns: str) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE)


# Verb / cue families, counted in the instruction only.
REASONING_CUES = _cues(
    r"why", r"how come", r"explain\w*", r"analy[sz]\w*", r"compar\w*", r"contrast\w*", r"evaluat\w*",
    r"assess\w*", r"justif\w*", r"argu\w*", r"reason\w*", r"critiqu\w*", r"discuss\w*",
    r"implications?", r"trade-?offs?", r"pros and cons", r"should", r"would", r"recommend\w*",
    r"predict\w*", r"what if", r"step by step", r"prove", r"calculat\w*", r"solve", r"estimat\w*",
)
CREATIVE_CUES = _cues(
    r"write", r"story", r"poem", r"essay", r"imagine", r"invent\w*", r"creat\w*", r"compos\w*",
    r"design\w*", r"brainstorm\w*", r"ideas?", r"suggest\w*", r"come up with", r"slogan", r"letter",
    r"song", r"pretend",
)
LOOKUP_CUES = _cues(
    r"extract\w*", r"list", r"identify", r"find", r"name", r"which", r"who", r"when", r"where",
    r"what is", r"what are", r"classify", r"categori[sz]e", r"sort", r"convert", r"rewrite",
    r"reformat\w*", r"given (the|this) (text|paragraph|passage)",
)
SUMMARY_CUES = _cues(r"summari[sz]\w*", r"tl;?dr", r"condens\w*", r"shorten\w*", r"key points", r"main points")

CONSTRAINT_CUES = _cues(
    r"must", r"exactly", r"at least", r"at most", r"no more than", r"no less than", r"only",
    r"without", r"do not", r"don't", r"never", r"always", r"avoid", r"each", r"every",
    r"maximum", r"minimum", r"within",
)
NUMERIC_LIMIT = re.compile(
    r"\b\d+\s*(words?|sentences?|lines?|paragraphs?|bullets?|points?|items?|characters?|examples?|steps?)\b",
    re.IGNORECASE,
)

# Requested output format: structured formats are harder to get exactly right than plain text.
STRUCTURED_FORMATS = _cues(r"json", r"tables?", r"csv", r"yaml", r"xml", r"markdown", r"columns?", r"schema", r"key-value")
SIMPLE_FORMATS = _cues(
    r"list", r"bullets?", r"bullet points?", r"numbered", r"comma[- ]separated", r"one word",
    r"yes or no", r"headings?", r"outline",
)

HANDCRAFTED_FEATURE_NAMES: tuple[str, ...] = (
    "log_tokens_total",
    "log_tokens_instruction",
    "log_tokens_context",
    "has_context",
    "reasoning_verbs",
    "creative_verbs",
    "lookup_verbs",
    "summary_verbs",
    "constraint_count",
    "format_complexity",
    "question_count",
)


def approx_token_count(text: str) -> int:
    """Rough token count (words + punctuation). Close enough for a feature; no tokenizer needed."""
    return len(_TOKEN.findall(text))


def split_prompt(text: str) -> tuple[str, str]:
    """Split a prompt into (instruction, context).

    Uses the ``Context:`` marker our sampler writes. Real API traffic has no marker, so we
    fall back to a heuristic: if there are several paragraphs and the longest has at least
    ``CONTEXT_MIN_WORDS`` words, that paragraph is the context and the rest is the instruction.
    """
    if CONTEXT_MARKER in text:
        instruction, context = text.split(CONTEXT_MARKER, 1)
        return instruction.strip(), context.strip()
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    if len(paragraphs) >= 2:
        longest_index = max(range(len(paragraphs)), key=lambda i: len(paragraphs[i]))
        if len(paragraphs[longest_index].split()) >= CONTEXT_MIN_WORDS:
            instruction = "\n\n".join(p for i, p in enumerate(paragraphs) if i != longest_index)
            return instruction, paragraphs[longest_index]
    return text.strip(), ""


def format_complexity(instruction: str) -> int:
    """0 = no format requested; +2 per distinct structured format, +1 per distinct simple one."""
    structured = {m.lower() for m in STRUCTURED_FORMATS.findall(instruction)}
    simple = {m.lower() for m in SIMPLE_FORMATS.findall(instruction)}
    return 2 * len(structured) + len(simple)


def constraint_count(instruction: str) -> int:
    """Words like "exactly"/"must"/"without" plus numeric limits like "6 lines"."""
    return len(CONSTRAINT_CUES.findall(instruction)) + len(NUMERIC_LIMIT.findall(instruction))


def handcrafted_features(text: str) -> list[float]:
    """The hand-crafted feature vector for one prompt, in ``HANDCRAFTED_FEATURE_NAMES`` order."""
    instruction, context = split_prompt(text)
    return [
        math.log1p(approx_token_count(text)),
        math.log1p(approx_token_count(instruction)),
        math.log1p(approx_token_count(context)),
        1.0 if context else 0.0,
        float(len(REASONING_CUES.findall(instruction))),
        float(len(CREATIVE_CUES.findall(instruction))),
        float(len(LOOKUP_CUES.findall(instruction))),
        float(len(SUMMARY_CUES.findall(instruction))),
        float(constraint_count(instruction)),
        float(format_complexity(instruction)),
        float(instruction.count("?")),
    ]


def instruction_texts(texts: Iterable[str]) -> list[str]:
    """Map prompts to their instruction part (module-level so the pipeline can be pickled)."""
    return [split_prompt(t)[0] for t in texts]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """scikit-learn transformer wrapping :func:`handcrafted_features`. Stateless."""

    def fit(self, texts: Sequence[str], y: object = None) -> HandcraftedFeatures:
        return self

    def transform(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return np.array([handcrafted_features(t) for t in texts], dtype=np.float64)

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.str_]:
        return np.array(HANDCRAFTED_FEATURE_NAMES)


FeatureSet = Literal["tfidf_hc", "embed_hc", "embed_hc_tfidf"]
FEATURE_SETS: tuple[FeatureSet, ...] = get_args(FeatureSet)


def build_feature_union(feature_set: FeatureSet = "tfidf_hc", *, max_tfidf_features: int = 3000) -> FeatureUnion:
    """Combine feature groups side by side.

    - ``tfidf_hc``:       TF-IDF of the instruction + scaled hand-crafted features (v1).
    - ``embed_hc``:       sentence embedding of the prompt + hand-crafted features.
    - ``embed_hc_tfidf``: all three.
    """
    from autopilot.classifier.embeddings import EmbeddingFeatures  # avoid an import cycle

    tfidf = Pipeline(
        [
            ("instruction", FunctionTransformer(instruction_texts)),
            (
                "vectorizer",
                # No stop-word removal: sklearn's English stop list includes "why", "how", "which" —
                # some of our strongest difficulty signals.
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=max_tfidf_features),
            ),
        ]
    )
    handcrafted = Pipeline([("extract", HandcraftedFeatures()), ("scale", StandardScaler())])
    parts: dict[FeatureSet, list[tuple[str, object]]] = {
        "tfidf_hc": [("tfidf", tfidf), ("handcrafted", handcrafted)],
        "embed_hc": [("embedding", EmbeddingFeatures()), ("handcrafted", handcrafted)],
        "embed_hc_tfidf": [("embedding", EmbeddingFeatures()), ("handcrafted", handcrafted), ("tfidf", tfidf)],
    }
    return FeatureUnion(parts[feature_set])


def feature_names(union: FeatureUnion) -> list[str]:
    """Column names of a *fitted* feature union, e.g. ``tfidf:why``, ``hc:has_context``, ``emb:12``."""
    names: list[str] = []
    for name, transformer in union.transformer_list:
        if name == "tfidf":
            names += [f"tfidf:{w}" for w in transformer.named_steps["vectorizer"].get_feature_names_out()]
        elif name == "handcrafted":
            names += [f"hc:{n}" for n in HANDCRAFTED_FEATURE_NAMES]
        elif name == "embedding":
            names += [f"emb:{i}" for i in range(transformer.dim_)]
    return names
