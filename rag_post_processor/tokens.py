"""Token counting.

Chunk sizes in this Actor are expressed in *tokens*, not characters, because
characters are not what any embedding model charges for or limits on. The same
1000-character chunk is ~160 tokens of English prose and ~1080 tokens of
Japanese -- a user who sets a character budget to "stay under my model's limit"
has no idea what they are actually buying.

tiktoken's BPE files are baked into the Docker image at build time (see
Dockerfile / TIKTOKEN_CACHE_DIR), so no network fetch happens at runtime. If the
encoding is somehow unavailable anyway, we degrade to a deliberately
conservative character heuristic rather than failing the run -- see
_HeuristicEncoder.
"""

from __future__ import annotations

import re
from typing import Dict, List, Protocol

# Embedding models: text-embedding-3-small and -large both use cl100k_base,
# NOT o200k_base, despite being newer than the o200k era. Verified by executing
# tiktoken.encoding_for_model(). o200k_base is reached via gpt-4o-/gpt-5-/o1-/o3-.
DEFAULT_ENCODING = "cl100k_base"

SUPPORTED_ENCODINGS = ("cl100k_base", "o200k_base", "p50k_base", "r50k_base")

# Documented max input for text-embedding-3-small / -large. OpenAI's current
# docs say 8192; 8191 is the long-cited safe ceiling, so we use it as the
# validation bound.
MAX_EMBEDDING_TOKENS = 8191

# Measured chars/token on cl100k_base: English prose ~6.1, German ~3.3,
# Python source ~3.6, JSON ~2.4, Japanese ~0.9. A heuristic that guesses high
# on token count is the safe direction (chunks come out smaller than the
# budget, never larger), so we assume a low chars-per-token ratio.
_HEURISTIC_CHARS_PER_TOKEN = 2.5


class Encoder(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def truncate_to(self, text: str, max_tokens: int) -> str: ...


class _TiktokenEncoder:
    """Exact BPE counting via tiktoken."""

    def __init__(self, name: str, enc):
        self.name = name
        self._enc = enc
        self._exact = True

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text, disallowed_special=()))

    def truncate_to(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self._enc.encode(text, disallowed_special=())
        if len(ids) <= max_tokens:
            return text
        return self._enc.decode(ids[:max_tokens])


class _HeuristicEncoder:
    """Fallback used only if tiktoken cannot be loaded at all.

    Biased to over-count tokens so chunks land under the user's budget rather
    than over it. Runs that use this are flagged in the log and in every output
    row via `token_count_method`, so a consumer can tell an exact count from an
    estimate instead of silently trusting a guess.
    """

    def __init__(self, name: str):
        self.name = name
        self._exact = False

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / _HEURISTIC_CHARS_PER_TOKEN))

    def truncate_to(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return text[: int(max_tokens * _HEURISTIC_CHARS_PER_TOKEN)]


_CACHE: Dict[str, Encoder] = {}
_LOAD_ERRORS: List[str] = []


def get_encoder(encoding_name: str = DEFAULT_ENCODING) -> Encoder:
    """Return a cached Encoder. Never raises; falls back to the heuristic."""
    if encoding_name in _CACHE:
        return _CACHE[encoding_name]
    try:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding(encoding_name)
        encoder: Encoder = _TiktokenEncoder(encoding_name, enc)
    except Exception as exc:  # pragma: no cover - only hit if the image is broken
        _LOAD_ERRORS.append(f"{encoding_name}: {exc}")
        encoder = _HeuristicEncoder(encoding_name)
    _CACHE[encoding_name] = encoder
    return encoder


def is_exact(encoder: Encoder) -> bool:
    return bool(getattr(encoder, "_exact", False))


def method_label(encoder: Encoder) -> str:
    """Value emitted on every output row so consumers can trust or distrust it."""
    return f"tiktoken:{encoder.name}" if is_exact(encoder) else "estimated:chars"


def load_errors() -> List[str]:
    return list(_LOAD_ERRORS)


_WORD_SPLIT = re.compile(r"(\s+)")


def split_to_token_budget(text: str, encoder: Encoder, max_tokens: int) -> List[str]:
    """Last-resort splitter for a single unbreakable run of text.

    Splits on whitespace so pieces never begin or end mid-word, which the
    previous character-slicing implementation did on 13 of 17 chunks of
    ordinary prose. Only if a *single* whitespace-free token still exceeds the
    budget (a minified blob, a base64 payload) do we cut inside it.
    """
    if max_tokens <= 0 or not text:
        return [text] if text else []
    if encoder.count(text) <= max_tokens:
        return [text]

    out: List[str] = []
    current = ""
    for piece in _WORD_SPLIT.split(text):
        if not piece:
            continue
        candidate = current + piece
        if encoder.count(candidate) <= max_tokens:
            current = candidate
            continue

        if current.strip():
            out.append(current.strip())
        current = ""
        if piece.isspace():
            continue

        # A single whitespace-free run longer than the whole budget (a minified
        # blob, a base64 payload, unspaced CJK): cut inside it. This is the only
        # place the splitter is allowed to break a word.
        while encoder.count(piece) > max_tokens:
            head = encoder.truncate_to(piece, max_tokens)
            if not head or len(head) >= len(piece):
                break
            out.append(head)
            piece = piece[len(head):]
        current = piece

    if current.strip():
        out.append(current.strip())
    return [c for c in out if c]
