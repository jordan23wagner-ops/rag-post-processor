"""Unit tests for the chunk_text output-count invariant and tail-merge logic.

Dev-only; excluded from the shipped image via .dockerignore. Run with:
    python -m unittest discover -s tests
"""
import math
import sys
import types
import unittest


class _StubLog:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


class _StubActor:
    log = _StubLog()


# main.py does `from apify import Actor` at import time; stub it so these
# tests don't need the real apify package or an actor run context.
sys.modules.setdefault("apify", types.SimpleNamespace(Actor=_StubActor()))

from rag_post_processor.main import chunk_text, ChunkCountExceeded  # noqa: E402


def max_expected_for(text_len: int, chunk_size: int) -> int:
    return math.ceil(text_len / max(1, chunk_size // 2)) + 2


class ChunkCountInvariantTests(unittest.TestCase):
    """The bound must hold (raise or stay under it) for adversarial params,
    even when they bypass resolve_chunk_params' own clamping."""

    def _assert_never_exceeds_bound(self, text, chunk_size, overlap, min_chunk_chars=50):
        expected = max_expected_for(len(text), chunk_size)
        try:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, min_chunk_chars=min_chunk_chars)
        except ChunkCountExceeded as e:
            self.assertLessEqual(e.actual, expected + 1, "exception should fire at or just past the bound")
        else:
            self.assertLessEqual(len(chunks), expected)

    def test_overlap_ge_chunk_size(self):
        self._assert_never_exceeds_bound("x" * 5000, chunk_size=100, overlap=500)

    def test_chunk_size_one(self):
        self._assert_never_exceeds_bound("x" * 500, chunk_size=1, overlap=0)

    def test_chunk_size_zero(self):
        # Would previously infinite-loop (start never advances); must now
        # raise ChunkCountExceeded within a bounded number of attempts.
        with self.assertRaises(ChunkCountExceeded):
            chunk_text("x" * 5000, chunk_size=0, overlap=100)

    def test_chunk_size_negative(self):
        self._assert_never_exceeds_bound("x" * 5000, chunk_size=-10, overlap=5)

    def test_huge_input_with_unclamped_overlap_raises(self):
        # Simulates data arriving via the datasetId chaining path, which
        # never passes through resolve_chunk_params' overlap clamp: a
        # 500,000-char input with chunk_size=800, overlap=750 only advances
        # 50 chars/iteration -> ~10,000 chunks needed, vastly more than the
        # ~1,252 the invariant expects for this chunk_size. Must raise.
        with self.assertRaises(ChunkCountExceeded) as ctx:
            chunk_text("x" * 500_000, chunk_size=800, overlap=750)
        self.assertEqual(ctx.exception.chunk_size, 800)
        self.assertEqual(ctx.exception.overlap, 750)
        self.assertGreater(ctx.exception.actual, ctx.exception.expected)


class TailMergeTests(unittest.TestCase):
    def test_merge_no_runt_tail_on_non_multiple_length(self):
        # 4141 chars: not a clean multiple of chunk_size=100.
        text = "word " * 828 + "tail"
        chunks = chunk_text(text, chunk_size=100, overlap=20, min_chunk_chars=50)

        self.assertGreater(len(chunks), 1)
        # No chunk (including the last) may be a sub-threshold runt.
        for c in chunks:
            self.assertGreaterEqual(len(c), 50, f"found a runt chunk of length {len(c)}")
        # Full text preserved: nothing dropped.
        self.assertIn("tail", chunks[-1])

    def test_short_input_emits_exactly_one_row_not_zero(self):
        chunks = chunk_text("short text under threshold", chunk_size=1000, overlap=100, min_chunk_chars=50)
        self.assertEqual(len(chunks), 1)

    def test_empty_input_is_zero_rows(self):
        self.assertEqual(chunk_text("", chunk_size=1000, overlap=100), [])

    def test_no_previous_chunk_emits_as_is(self):
        # Whole input under min_chunk_chars threshold but also under
        # chunk_size, so it takes the early-return path as a single chunk.
        chunks = chunk_text("hi", chunk_size=1000, overlap=100, min_chunk_chars=50)
        self.assertEqual(chunks, ["hi"])


if __name__ == "__main__":
    unittest.main()
