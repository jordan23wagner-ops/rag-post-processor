"""End-to-end run of main() against a stubbed Actor.

Covers the paths that unit tests can't: charge ordering, the cost-cap exit,
what actually lands in the output dataset, and the credential-leak fallback.
"""
import types

import pytest

import rag_post_processor.main as main_mod


class StubActor:
    """Minimal stand-in for apify.Actor, recording everything main() does."""

    def __init__(self, actor_input, limit_reached=False):
        self._input = actor_input
        self.pushed = []
        self.charges = []
        self.status = None
        self.limit_reached = limit_reached
        self.logs = {"info": [], "warning": [], "error": []}
        parent = self

        class _Log:
            def info(self, m, *a, **k): parent.logs["info"].append(str(m))
            def warning(self, m, *a, **k): parent.logs["warning"].append(str(m))
            def error(self, m, *a, **k): parent.logs["error"].append(str(m))

        self.log = _Log()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_input(self):
        return self._input

    async def charge(self, event_name, count=1):
        self.charges.append((event_name, count))
        return types.SimpleNamespace(event_charge_limit_reached=self.limit_reached)

    async def push_data(self, item):
        self.pushed.append(item)

    async def set_status_message(self, msg):
        self.status = msg

    @staticmethod
    def get_env():
        return {"started_at": "2026-08-02T00:00:00+00:00", "token": ""}


@pytest.fixture
def run(monkeypatch):
    async def _run(actor_input, limit_reached=False):
        stub = StubActor(actor_input, limit_reached=limit_reached)
        monkeypatch.setattr(main_mod, "Actor", stub)
        await main_mod.main()
        return stub
    return _run


DOC = """# Widget SDK

Install from npm. See the [docs](https://docs.example.com/sdk) for details.

## Quick start

```python
from widget import Client
client = Client(api_key="sk-test")
```

## Troubleshooting

If the render fails, check the key. Prices are $49.99/mo. approx. 3% fail.
"""


@pytest.mark.asyncio
async def test_end_to_end_emits_rows_with_the_documented_fields(run):
    stub = await run({"data": [{"url": "https://ex.com/a", "markdown": DOC}],
                      "max_tokens": 96, "overlap_tokens": 0})
    assert stub.pushed
    row = stub.pushed[0]
    for field in ("chunk_id", "original_id", "chunk_index", "total_chunks",
                  "chunk_text", "token_count", "token_count_method",
                  "chunk_length_chars", "content_hash", "overlap_tokens",
                  "block_kinds", "cleaned_at", "source_url", "source_field"):
        assert field in row, f"missing {field}"
    assert row["source_field"] == "markdown"
    assert row["source_url"] == "https://ex.com/a"


@pytest.mark.asyncio
async def test_total_chunks_matches_rows_emitted(run):
    stub = await run({"data": [{"markdown": DOC}], "max_tokens": 64, "overlap_tokens": 0})
    assert len({r["total_chunks"] for r in stub.pushed}) == 1
    assert stub.pushed[0]["total_chunks"] == len(stub.pushed)
    assert [r["chunk_index"] for r in stub.pushed] == list(range(len(stub.pushed)))


@pytest.mark.asyncio
async def test_no_emitted_chunk_exceeds_max_tokens(run):
    stub = await run({"data": [{"markdown": DOC * 4}], "max_tokens": 80,
                      "overlap_tokens": 16})
    assert stub.pushed
    assert all(r["token_count"] <= 80 for r in stub.pushed)


@pytest.mark.asyncio
async def test_charged_exactly_once_per_run(run):
    stub = await run({"data": [{"text": "a " * 3000}, {"text": "b " * 3000}]})
    assert len(stub.charges) == 1
    assert stub.charges[0][0] == main_mod.INPUT_KB_EVENT


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens,overlap", [(64, 32), (128, 64), (512, 256)])
async def test_charge_is_identical_across_chunk_settings(run, max_tokens, overlap):
    stub = await run({"data": [{"text": "word " * 2000}],
                      "max_tokens": max_tokens, "overlap_tokens": overlap})
    assert stub.charges[0][1] == 10   # 10000 bytes -> 10 KB units, always


@pytest.mark.asyncio
async def test_cost_cap_emits_nothing(run):
    stub = await run({"data": [{"text": "x " * 5000}]}, limit_reached=True)
    assert stub.pushed == []
    assert "exceeds this run's max cost" in stub.status


@pytest.mark.asyncio
async def test_maximum_overlap_end_to_end_returns_data(run):
    """The shipped build billed and returned nothing for this exact request."""
    stub = await run({"data": [{"text": "A sentence about revenue. " * 400}],
                      "max_tokens": 128, "overlap_tokens": 64})
    assert stub.charges
    assert len(stub.pushed) > 1


@pytest.mark.asyncio
async def test_non_dict_items_are_skipped_and_not_charged_for(run):
    stub = await run({"data": [5, None, "loose string", {"text": "real content here"}]})
    assert stub.charges[0][1] == 1
    assert all(r["original_id"].startswith("item_") for r in stub.pushed)


@pytest.mark.asyncio
async def test_items_with_no_text_field_are_not_charged_for(run):
    a = await run({"data": [{"text": "same content"}]})
    b = await run({"data": [{"text": "same content"}, {"price": 4}, {"tags": ["x"]}]})
    assert a.charges[0][1] == b.charges[0][1]


@pytest.mark.asyncio
async def test_empty_input_charges_nothing(run):
    stub = await run({"data": []})
    assert stub.charges == []
    assert stub.pushed == []


@pytest.mark.asyncio
async def test_script_content_never_reaches_the_dataset(run):
    html = ("<html><head><script>window.dataLayer=[];</script>"
            "<style>body{margin:0}</style></head><body><h1>Pricing</h1>"
            "<p>Plans start at &pound;99.</p></body></html>")
    stub = await run({"data": [{"html": html}]})
    blob = " ".join(r["chunk_text"] for r in stub.pushed)
    assert "dataLayer" not in blob
    assert "margin:0" not in blob
    assert "£99" in blob


@pytest.mark.asyncio
async def test_token_never_reaches_the_output_dataset(run):
    """The old fallback chunked the Actor's own input, token included."""
    stub = await run({"token": "apify_api_SECRET", "notes": "some loose text to process"})
    blob = " ".join(r["chunk_text"] for r in stub.pushed)
    assert "apify_api_SECRET" not in blob


@pytest.mark.asyncio
async def test_failed_dataset_fetch_is_fatal_and_charges_nothing(run, monkeypatch):
    from rag_post_processor.fetching import FetchResult
    monkeypatch.setattr(main_mod, "fetch_dataset_items",
                        lambda *a, **k: FetchResult(error="TimeoutError: boom"))
    stub = await run({"datasetId": "ds1234567"})
    assert stub.charges == []
    assert stub.pushed == []
    assert "Could not read dataset" in stub.status


@pytest.mark.asyncio
async def test_chained_dataset_items_are_processed(run, monkeypatch):
    from rag_post_processor.fetching import FetchResult
    monkeypatch.setattr(
        main_mod, "fetch_dataset_items",
        lambda *a, **k: FetchResult(items=[{"text": "chained content here"}],
                                    reported_total=1, pages=1),
    )
    stub = await run({"datasetId": "ds1234567"})
    assert len(stub.pushed) == 1
    assert "chained content" in stub.pushed[0]["chunk_text"]


@pytest.mark.asyncio
async def test_truncated_dataset_read_is_reported(run, monkeypatch):
    from rag_post_processor.fetching import FetchResult
    monkeypatch.setattr(
        main_mod, "fetch_dataset_items",
        lambda *a, **k: FetchResult(items=[{"text": "only one"}],
                                    reported_total=500, pages=1, truncated=True),
    )
    stub = await run({"datasetId": "ds1234567"})
    assert any("INCOMPLETE" in w for w in stub.logs["warning"])


@pytest.mark.asyncio
async def test_token_amplification_is_reported(run):
    stub = await run({"data": [{"text": "A sentence about revenue growth. " * 200}],
                      "max_tokens": 128, "overlap_tokens": 48})
    assert "amplification" in stub.status
    assert any("embedding provider will bill you" in m for m in stub.logs["info"])


@pytest.mark.asyncio
async def test_invalid_settings_fall_back_with_a_warning(run):
    stub = await run({"data": [{"text": "content here"}],
                      "max_tokens": "big", "overlap_tokens": True,
                      "encoding": "not_a_real_encoding"})
    assert stub.pushed
    assert any("max_tokens" in w for w in stub.logs["warning"])
    assert any("Unknown encoding" in w for w in stub.logs["warning"])
    assert stub.pushed[0]["token_count_method"] == "tiktoken:cl100k_base"


@pytest.mark.asyncio
async def test_explicit_text_field_is_honoured(run):
    stub = await run({"data": [{"markdown": "# ignored", "custom": "the real content"}],
                      "text_field": "custom"})
    assert stub.pushed[0]["source_field"] == "custom"
    assert "real content" in stub.pushed[0]["chunk_text"]


@pytest.mark.asyncio
async def test_one_bad_item_does_not_abort_the_run(run, monkeypatch):
    calls = {"n": 0}
    real = main_mod.chunk_document

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic failure")
        return real(*a, **k)

    monkeypatch.setattr(main_mod, "chunk_document", flaky)
    stub = await run({"data": [{"text": "first item"}, {"text": "second item here"}]})
    assert stub.pushed
    assert any("failed and was skipped" in w for w in stub.logs["warning"])
