"""Billing tests. Revenue-critical: pin the arithmetic and the basis.

Carried over from the previous suite, plus the new invariant that the charge is
computed from the text actually read rather than from whole-item JSON.
"""
import types

import pytest

from rag_post_processor.main import (
    INPUT_KB_EVENT,
    SCRAPER_TEXT_FIELDS,
    charge_for_input,
    compute_billable_bytes,
    compute_billed_units,
    extract_text,
)


class ChargeRecorder:
    def __init__(self, limit_reached=False):
        self.calls = []
        self.limit_reached = limit_reached

    async def __call__(self, event_name, count=1):
        self.calls.append((event_name, count))
        return types.SimpleNamespace(event_charge_limit_reached=self.limit_reached)


@pytest.fixture
def charger(monkeypatch):
    import rag_post_processor.main as main_mod
    rec = ChargeRecorder()
    monkeypatch.setattr(main_mod.Actor, "charge", rec, raising=False)
    return rec


# --- arithmetic -----------------------------------------------------------

@pytest.mark.parametrize("byts,units", [
    (0, 0), (-5, 0), (1, 1), (1023, 1), (1024, 1), (1025, 2), (2048, 2), (2049, 3),
])
def test_billed_unit_arithmetic(byts, units):
    assert compute_billed_units(byts) == units


def test_empty_input_is_zero_bytes():
    assert compute_billable_bytes([]) == 0


def test_bytes_are_utf8_not_characters():
    # A multibyte document must not be under-billed.
    assert compute_billable_bytes(["é" * 100]) == 200
    assert compute_billable_bytes(["四" * 100]) == 300


# --- basis ----------------------------------------------------------------

WCC_ITEM = {
    "url": "https://example.com/pricing",
    "text": "Real content here. " * 30,
    "metadata": {"canonicalUrl": "https://example.com/pricing", "title": "Pricing",
                 "description": "x" * 300, "languageCode": "en"},
    "screenshotUrl": "https://api.apify.com/v2/key-value-stores/abc/records/s.jpg",
    "debug": {"requestHandlerMode": "http", "statusCode": 200},
    "html": "<div>" + "<p>nav footer cookie banner</p>" * 40 + "</div>",
}


def test_charge_is_based_on_text_read_not_whole_item_json():
    """Shipped version billed 3395 bytes to process 572 on this exact item."""
    import json
    text, field = extract_text(WCC_ITEM, None)
    billed = compute_billable_bytes([text])
    whole_item = len(json.dumps(WCC_ITEM, ensure_ascii=False).encode())
    assert billed == len(text.encode("utf-8"))
    assert billed < whole_item / 2


def test_unread_fields_do_not_affect_the_bill():
    lean = {"text": "Real content here. " * 30}
    fat = dict(lean, debug={"a": "x" * 5000}, screenshotUrl="https://x/y.jpg")
    t1, _ = extract_text(lean, None)
    t2, _ = extract_text(fat, None)
    assert compute_billable_bytes([t1]) == compute_billable_bytes([t2])


# --- charge calls ---------------------------------------------------------

@pytest.mark.asyncio
async def test_zero_bytes_makes_no_charge_call(charger):
    units, result = await charge_for_input(0)
    assert units == 0
    assert result is None
    assert charger.calls == []


@pytest.mark.asyncio
async def test_charged_exactly_once_with_the_right_event(charger):
    units, _ = await charge_for_input(3500)
    assert charger.calls == [(INPUT_KB_EVENT, units)]
    assert units == 4


@pytest.mark.asyncio
async def test_charge_is_independent_of_chunk_settings(charger):
    """charge_for_input takes no chunking argument at all, so no chunk_size or
    overlap can inflate a bill. Asserted explicitly anyway."""
    a, _ = await charge_for_input(10_000)
    b, _ = await charge_for_input(10_000)
    assert a == b
    assert {c[1] for c in charger.calls} == {a}


@pytest.mark.asyncio
async def test_charge_limit_reached_is_surfaced(monkeypatch):
    import rag_post_processor.main as main_mod
    rec = ChargeRecorder(limit_reached=True)
    monkeypatch.setattr(main_mod.Actor, "charge", rec, raising=False)
    units, result = await charge_for_input(5000)
    assert result.event_charge_limit_reached is True
    assert units > 0


# --- field selection ------------------------------------------------------

def test_markdown_wins_over_description():
    item = {"description": "SEO meta description.", "markdown": "# Real body\n\ntext"}
    text, field = extract_text(item, None)
    assert field == "markdown"


def test_body_wins_over_summary():
    """Shipped priority list put `summary` and `description` ahead of several
    real content fields, so a 155-character meta description could be indexed
    instead of the article."""
    item = {"summary": "teaser", "body": "the full article body"}
    _, field = extract_text(item, None)
    assert field == "body"
    assert SCRAPER_TEXT_FIELDS.index("body") < SCRAPER_TEXT_FIELDS.index("summary")


def test_explicit_text_field_overrides_detection():
    item = {"markdown": "# ignored", "custom_body": "the real thing"}
    text, field = extract_text(item, "custom_body")
    assert field == "custom_body"
    assert text == "the real thing"


def test_explicit_text_field_falls_back_when_absent():
    item = {"markdown": "# used instead"}
    text, field = extract_text(item, "not_here")
    assert field == "markdown"


def test_item_with_no_text_field_returns_empty():
    text, field = extract_text({"price": 4, "tags": ["a"]}, None)
    assert text == ""
    assert field == ""


def test_non_string_field_is_not_selected():
    text, field = extract_text({"text": {"nested": "object"}, "body": "real"}, None)
    assert field == "body"


# --- v0.x backward compatibility ------------------------------------------

from rag_post_processor.main import apply_legacy_input, resolve_settings  # noqa: E402


def test_legacy_chunk_size_is_translated_to_tokens():
    out = apply_legacy_input({"chunk_size": 1000, "overlap": 100, "min_chunk_chars": 50})
    assert out["max_tokens"] == 250
    assert out["overlap_tokens"] == 25
    assert out["min_tokens"] == 13


def test_new_keys_win_over_legacy_keys():
    out = apply_legacy_input({"chunk_size": 1000, "max_tokens": 512})
    assert out["max_tokens"] == 512


def test_legacy_keys_survive_resolve_settings_end_to_end():
    s = resolve_settings({"chunk_size": 2048, "overlap": 256})
    assert s["max_tokens"] == 512
    assert s["overlap_tokens"] == 64


@pytest.mark.parametrize("bad", [None, True, "1000", -5, 3.5])
def test_invalid_legacy_values_are_ignored_not_fatal(bad):
    out = apply_legacy_input({"chunk_size": bad})
    assert "max_tokens" not in out


def test_absent_legacy_keys_change_nothing():
    assert apply_legacy_input({"data": []}) == {"data": []}
