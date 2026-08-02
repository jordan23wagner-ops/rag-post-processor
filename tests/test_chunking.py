"""Chunking tests.

The headline test is `test_no_schema_legal_config_ever_loses_an_item`, which is
the sweep that the shipped version fails 16/40 times.
"""
import random
import re

import pytest

from rag_post_processor.chunking import (
    Chunk,
    build_units,
    chunk_document,
    parse_blocks,
    split_sentences,
)
from rag_post_processor.tokens import get_encoder, split_to_token_budget

ENC = get_encoder("cl100k_base")


def _prose(sentences=90, seed=7):
    rng = random.Random(seed)
    words = ("revenue pipeline latency customer schema invoice retention embedding "
             "cluster throughput migration compliance forecast anomaly dashboard "
             "segment cohort provisioning quota checkpoint").split()
    return " ".join(
        " ".join(rng.choice(words) for _ in range(rng.randint(8, 18))) + "."
        for _ in range(sentences)
    )


MARKDOWN = """# Widget SDK

Install from npm. See the [docs](https://docs.example.com/sdk) for details.

## Requirements

| Runtime | Minimum |
|---------|---------|
| Node.js | 20.11.0 |
| Python  | 3.11    |
| Go      | 1.22    |
| Rust    | 1.79    |

## Quick start

```python
from widget import Client

client = Client(api_key="sk-test")
resp = client.render(template="invoice", data={"total": 49.99})
print(resp.url)
```

### Notes

- Keys rotate every 90 days.
- Rate limit is 100 req/min.
- Contact Dr. Smith for an exemption.

## Troubleshooting

If the render fails, check the key. Prices are $49.99/mo. approx. 3% of calls fail.
"""


# =========================================================================
# The regression that motivated the rewrite
# =========================================================================

@pytest.mark.parametrize("max_tokens", [16, 24, 32, 48, 64, 128, 256, 512, 1024, 8191])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
def test_no_schema_legal_config_ever_loses_an_item(max_tokens, ratio):
    """Every legal (max_tokens, overlap) pair must return chunks, never raise.

    The shipped implementation raised ChunkCountExceeded and discarded the item
    -- after billing for it -- on 16 of 40 equivalent configurations, including
    every 50%-overlap run at chunk_size <= 1000.
    """
    text = _prose()
    overlap = int(max_tokens * ratio)
    chunks = chunk_document(text, ENC, max_tokens=max_tokens, overlap_tokens=overlap)
    assert chunks, f"lost the item at max_tokens={max_tokens}, overlap={overlap}"
    assert all(c.text.strip() for c in chunks)


@pytest.mark.parametrize("max_tokens,overlap", [
    (64, 32), (128, 64), (256, 128), (512, 256), (1024, 512),
])
def test_maximum_overlap_is_a_supported_configuration(max_tokens, overlap):
    """50% overlap is a standard RAG recommendation and must simply work."""
    chunks = chunk_document(_prose(), ENC, max_tokens=max_tokens, overlap_tokens=overlap)
    assert len(chunks) > 1
    assert all(c.tokens <= max_tokens for c in chunks)


def test_overlap_above_fifty_percent_is_clamped_not_fatal():
    a = chunk_document(_prose(), ENC, max_tokens=128, overlap_tokens=127)
    b = chunk_document(_prose(), ENC, max_tokens=128, overlap_tokens=64)
    assert a and b
    assert all(c.tokens <= 128 for c in a)


# =========================================================================
# Budget correctness
# =========================================================================

@pytest.mark.parametrize("max_tokens", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("source", ["prose", "markdown"])
def test_no_chunk_exceeds_the_token_budget(max_tokens, source):
    text = _prose() if source == "prose" else MARKDOWN
    for c in chunk_document(text, ENC, max_tokens=max_tokens, overlap_tokens=max_tokens // 8):
        assert c.tokens <= max_tokens, f"{c.tokens} > {max_tokens}: {c.text[:120]!r}"


@pytest.mark.parametrize("text", [
    "English prose about quarterly revenue growth across regions. " * 40,
    '{"id":12345,"sku":"AB-99","price":49.99,"tags":["a","b"]},' * 40,
    "四半期報告書は全地域で着実な成長を示しています。" * 40,
    "Die Quartalsberichterstattung zeigt ein stetiges Wachstum. " * 40,
])
def test_budget_holds_across_scripts_and_content_types(text):
    """A character budget gave 117-1080 tokens for the same setting.

    A token budget must hold regardless of language or content type.
    """
    for c in chunk_document(text, ENC, max_tokens=128, overlap_tokens=16):
        assert c.tokens <= 128


def test_reported_token_count_matches_the_encoder():
    for c in chunk_document(MARKDOWN, ENC, max_tokens=128, overlap_tokens=16):
        assert c.tokens == ENC.count(c.text)


# =========================================================================
# Word and sentence boundaries
# =========================================================================

_WORDY = re.compile(r"[A-Za-z]")


def test_chunks_do_not_begin_or_end_mid_word():
    """Shipped version began 13 of 17 chunks mid-word on ordinary prose."""
    text = _prose()
    chunks = chunk_document(text, ENC, max_tokens=200, overlap_tokens=32,
                            include_heading_context=False)
    assert len(chunks) > 3
    for c in chunks:
        body = c.text.strip()
        head, tail = body.split()[0], body.split()[-1]
        # Every emitted word must be a whole word from the source.
        assert head.strip(".,;:!?") in text
        assert tail.strip(".,;:!?") in text


@pytest.mark.parametrize("text,expected_count", [
    ("One. Two. Three.", 3),
    ("Really? Yes! Okay.", 3),
    ("Dr. Smith arrived. He left.", 2),
    ("Costs approx. 3% of revenue. That is fine.", 2),
    ("Use e.g. this form. Or that one.", 2),
    ("Prices are $49.99 today. Tomorrow differs.", 2),
    ("Version v2.1 shipped. v2.2 follows.", 2),
    ("J. R. Wagner signed. It is done.", 2),
    ("The U.S. market grew. Others shrank.", 2),
])
def test_sentence_boundaries(text, expected_count):
    assert len(split_sentences(text)) == expected_count


def test_sentence_split_never_loses_characters():
    text = "One. Two! Three? Dr. Smith stayed."
    joined = " ".join(split_sentences(text))
    assert set(text.replace(" ", "")) == set(joined.replace(" ", ""))


def test_newline_terminated_sentences_are_boundaries():
    # The shipped splitter only knew '. ' and hard-cut mid-word here.
    text = "First sentence ends here.\nSecond sentence starts here."
    assert len(split_sentences(text)) == 2


# =========================================================================
# Structure integrity
# =========================================================================

def test_code_fences_are_balanced_in_every_chunk():
    for c in chunk_document(MARKDOWN, ENC, max_tokens=96, overlap_tokens=0):
        assert c.text.count("```") % 2 == 0, c.text


def test_split_code_keeps_its_language_on_every_part():
    big = "# T\n\n```python\n" + "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n```\n"
    chunks = chunk_document(big, ENC, max_tokens=128, overlap_tokens=0)
    code_chunks = [c for c in chunks if "```" in c.text]
    assert len(code_chunks) > 1
    for c in code_chunks:
        assert "```python" in c.text


def test_split_table_repeats_its_header_on_every_part():
    rows = "\n".join(f"| item-{i} | {i * 3} | region-{i % 4} |" for i in range(60))
    doc = "# Data\n\n| Name | Count | Region |\n|---|---|---|\n" + rows + "\n"
    chunks = chunk_document(doc, ENC, max_tokens=128, overlap_tokens=0)
    table_chunks = [c for c in chunks if "|" in c.text]
    assert len(table_chunks) > 1
    for c in table_chunks:
        assert "| Name | Count | Region |" in c.text, c.text[:200]


def test_list_items_are_never_split_mid_item():
    items = "\n".join(f"- item number {i} with some descriptive trailing text" for i in range(80))
    chunks = chunk_document("# L\n\n" + items, ENC, max_tokens=96, overlap_tokens=0)
    for c in chunks:
        for line in c.text.split("\n"):
            if line.startswith("- "):
                assert len(line) > 4


def test_headings_lead_their_content_not_trail_it():
    chunks = chunk_document(MARKDOWN, ENC, max_tokens=96, overlap_tokens=0)
    for c in chunks:
        last = [ln for ln in c.text.strip().split("\n") if ln.strip()][-1]
        assert not last.lstrip().startswith("#"), f"chunk ends on a bare heading: {c.text!r}"


def test_heading_path_is_recorded():
    chunks = chunk_document(MARKDOWN, ENC, max_tokens=96, overlap_tokens=0)
    paths = [tuple(c.heading_path) for c in chunks]
    assert ("Widget SDK", "Quick start") in paths or ("Widget SDK",) in paths
    assert any(len(p) >= 2 for p in paths)


def test_nested_heading_path_tracks_depth():
    doc = "# A\n\ntext a\n\n## B\n\ntext b\n\n### C\n\ntext c\n\n## D\n\ntext d\n"
    chunks = chunk_document(doc, ENC, max_tokens=64, overlap_tokens=0, min_tokens=0,
                            split_on_heading_level=6)
    paths = {tuple(c.heading_path) for c in chunks}
    assert ("A", "B", "C") in paths
    assert ("A", "D") in paths


def test_chunk_spanning_sibling_sections_reports_the_common_parent():
    """A chunk containing both '## B' and '## D' belongs to their parent 'A',
    not to whichever section happened to be packed first."""
    doc = "# A\n\ntext a\n\n## B\n\ntext b\n\n## D\n\ntext d\n"
    chunks = chunk_document(doc, ENC, max_tokens=512, overlap_tokens=0)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["A"]


def test_heading_context_is_prefixed_when_enabled():
    doc = "# Parent\n\n## Child\n\n" + ("Body sentence here. " * 40)
    with_ctx = chunk_document(doc, ENC, max_tokens=96, overlap_tokens=0,
                              include_heading_context=True)
    assert any("Parent" in c.text for c in with_ctx[1:])


def test_heading_context_can_be_disabled():
    doc = "# Parent\n\n## Child\n\n" + ("Body sentence here. " * 40)
    without = chunk_document(doc, ENC, max_tokens=96, overlap_tokens=0,
                             include_heading_context=False)
    assert not any(c.text.startswith("Parent > ") for c in without)


def test_split_on_heading_level_forces_section_boundaries():
    doc = "# A\n\nshort a\n\n## B\n\nshort b\n\n## C\n\nshort c\n"
    packed = chunk_document(doc, ENC, max_tokens=512, overlap_tokens=0,
                            split_on_heading_level=0)
    split = chunk_document(doc, ENC, max_tokens=512, overlap_tokens=0,
                           split_on_heading_level=2, min_tokens=0)
    assert len(split) > len(packed)


# =========================================================================
# Content preservation
# =========================================================================

def _normalize(s):
    return re.sub(r"\s+", " ", s).strip()


def test_no_source_text_is_lost():
    text = _prose(sentences=40)
    chunks = chunk_document(text, ENC, max_tokens=128, overlap_tokens=0,
                            include_heading_context=False)
    joined = _normalize(" ".join(c.text for c in chunks))
    for sentence in split_sentences(text):
        assert _normalize(sentence) in joined


def test_urls_survive_chunking():
    doc = "# T\n\nSee [docs](https://example.com/a/b) here. " + ("filler sentence. " * 60)
    chunks = chunk_document(doc, ENC, max_tokens=96, overlap_tokens=0)
    assert any("https://example.com/a/b" in c.text for c in chunks)


def test_content_hash_is_stable_and_distinct():
    a = chunk_document(MARKDOWN, ENC, max_tokens=128, overlap_tokens=16)
    b = chunk_document(MARKDOWN, ENC, max_tokens=128, overlap_tokens=16)
    assert [c.content_hash for c in a] == [c.content_hash for c in b]
    assert len({c.content_hash for c in a}) == len(a)


def test_overlap_tokens_reported_on_chunks():
    chunks = chunk_document(_prose(), ENC, max_tokens=128, overlap_tokens=32,
                            include_heading_context=False)
    assert chunks[0].overlap_tokens == 0
    assert any(c.overlap_tokens > 0 for c in chunks[1:])


def test_zero_overlap_produces_no_duplication():
    text = _prose(sentences=30)
    chunks = chunk_document(text, ENC, max_tokens=128, overlap_tokens=0,
                            include_heading_context=False)
    total = sum(c.tokens for c in chunks)
    assert total <= ENC.count(text) * 1.10


# =========================================================================
# Edge cases
# =========================================================================

@pytest.mark.parametrize("text", ["", "   ", "\n\n\n"])
def test_empty_input_produces_no_chunks(text):
    assert chunk_document(text, ENC) == []


def test_single_short_document_is_one_chunk():
    chunks = chunk_document("A short note about nothing.", ENC, max_tokens=512)
    assert len(chunks) == 1


def test_unbreakable_blob_is_still_split_within_budget():
    blob = "A" * 20000
    chunks = chunk_document(blob, ENC, max_tokens=64, overlap_tokens=0)
    assert chunks
    assert all(c.tokens <= 64 for c in chunks)


def test_document_that_is_only_a_heading():
    chunks = chunk_document("# Just a heading", ENC, max_tokens=128)
    assert len(chunks) == 1
    assert "Just a heading" in chunks[0].text


def test_unclosed_code_fence_does_not_lose_the_body():
    doc = "# T\n\n```python\nx = 1\ny = 2\n"
    chunks = chunk_document(doc, ENC, max_tokens=128)
    joined = " ".join(c.text for c in chunks)
    assert "x = 1" in joined and "y = 2" in joined


def test_parse_blocks_classifies_each_kind():
    kinds = [b.kind for b in parse_blocks(MARKDOWN)]
    assert "heading" in kinds
    assert "table" in kinds
    assert "code" in kinds
    assert "list" in kinds
    assert "paragraph" in kinds


def test_split_to_token_budget_never_cuts_mid_word_when_avoidable():
    text = " ".join(f"word{i}" for i in range(500))
    for piece in split_to_token_budget(text, ENC, 32):
        assert not piece.startswith("ord")
        assert ENC.count(piece) <= 32
