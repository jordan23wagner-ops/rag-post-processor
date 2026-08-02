"""Dataset fetching tests: pagination, retries, timeouts, auth, redaction."""
import io
import json
import urllib.error

import pytest

from rag_post_processor import fetching
from rag_post_processor.fetching import FetchResult, fetch_dataset_items, redact


class FakeResponse(io.BytesIO):
    def __init__(self, payload, headers=None, status=200):
        super().__init__(json.dumps(payload).encode())
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def make_opener(pages, total=None, recorder=None, fail_first=0, fail_with=None):
    state = {"calls": 0, "failures": 0}

    def _request(url, token, timeout):
        state["calls"] += 1
        if recorder is not None:
            recorder.append({"url": url, "token": token, "timeout": timeout})
        if state["failures"] < fail_first:
            state["failures"] += 1
            raise (fail_with or urllib.error.HTTPError(url, 503, "busy", {}, None))
        offset = 0
        for part in url.split("?", 1)[-1].split("&"):
            if part.startswith("offset="):
                offset = int(part.split("=")[1])
        flat = [item for page in pages for item in page]
        limit = 1000
        for part in url.split("?", 1)[-1].split("&"):
            if part.startswith("limit="):
                limit = int(part.split("=")[1])
        headers = {}
        if total is not None:
            headers["X-Apify-Pagination-Total"] = str(total)
        return FakeResponse(flat[offset:offset + limit], headers)

    _request.state = state
    return _request


@pytest.fixture
def no_sleep():
    return lambda _: None


def test_single_page(monkeypatch, no_sleep):
    monkeypatch.setattr(fetching, "_request", make_opener([[{"text": "a"}, {"text": "b"}]]))
    r = fetch_dataset_items("ds123456", "tok", page_size=1000, sleep=no_sleep)
    assert len(r.items) == 2
    assert r.pages == 1
    assert r.error is None
    assert r.complete


def test_pagination_reads_every_page(monkeypatch, no_sleep):
    """The shipped version issued one unbounded request and read no pagination
    headers, so any server-side cap truncated silently."""
    items = [{"text": f"item {i}"} for i in range(2500)]
    monkeypatch.setattr(fetching, "_request", make_opener([items], total=2500))
    r = fetch_dataset_items("ds123456", "tok", page_size=1000, sleep=no_sleep)
    assert len(r.items) == 2500
    assert r.pages == 3
    assert not r.truncated
    assert r.complete


def test_truncation_is_detected_from_the_total_header(monkeypatch, no_sleep):
    items = [{"text": f"i{i}"} for i in range(10)]
    monkeypatch.setattr(fetching, "_request", make_opener([items], total=99))
    r = fetch_dataset_items("ds123456", "tok", page_size=1000, sleep=no_sleep)
    assert r.truncated is True
    assert r.reported_total == 99
    assert not r.complete


def test_max_items_caps_and_flags_truncation(monkeypatch, no_sleep):
    items = [{"text": f"i{i}"} for i in range(500)]
    monkeypatch.setattr(fetching, "_request", make_opener([items], total=500))
    r = fetch_dataset_items("ds123456", "tok", max_items=50, page_size=1000, sleep=no_sleep)
    assert len(r.items) == 50
    assert r.truncated is True


def test_token_goes_in_a_header_not_the_query_string(monkeypatch, no_sleep):
    calls = []
    monkeypatch.setattr(fetching, "_request", make_opener([[{"text": "a"}]], recorder=calls))
    fetch_dataset_items("ds123456", "secret-token-value", sleep=no_sleep)
    assert "secret-token-value" not in calls[0]["url"]
    assert calls[0]["token"] == "secret-token-value"


def test_a_timeout_is_always_passed(monkeypatch, no_sleep):
    calls = []
    monkeypatch.setattr(fetching, "_request", make_opener([[{"text": "a"}]], recorder=calls))
    fetch_dataset_items("ds123456", "tok", sleep=no_sleep)
    assert calls[0]["timeout"] > 0


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_errors_are_retried(monkeypatch, no_sleep, code):
    opener = make_opener(
        [[{"text": "a"}]], fail_first=2,
        fail_with=urllib.error.HTTPError("u", code, "e", {}, None),
    )
    monkeypatch.setattr(fetching, "_request", opener)
    r = fetch_dataset_items("ds123456", "tok", sleep=no_sleep)
    assert r.error is None
    assert len(r.items) == 1
    assert opener.state["calls"] == 3


def test_permanent_errors_are_not_retried(monkeypatch, no_sleep):
    opener = make_opener(
        [[{"text": "a"}]], fail_first=1,
        fail_with=urllib.error.HTTPError("u", 404, "gone", {}, None),
    )
    monkeypatch.setattr(fetching, "_request", opener)
    r = fetch_dataset_items("ds123456", "tok", sleep=no_sleep)
    assert r.error == "HTTP 404"
    assert opener.state["calls"] == 1


def test_network_errors_are_retried_then_reported(monkeypatch, no_sleep):
    opener = make_opener([[{"text": "a"}]], fail_first=99,
                         fail_with=TimeoutError("timed out"))
    monkeypatch.setattr(fetching, "_request", opener)
    r = fetch_dataset_items("ds123456", "tok", max_retries=3, sleep=no_sleep)
    assert r.error is not None
    assert "TimeoutError" in r.error
    assert opener.state["calls"] == 3


def test_failure_is_distinguishable_from_empty(monkeypatch, no_sleep):
    """The shipped version returned [] for empty, failed and truncated alike."""
    monkeypatch.setattr(fetching, "_request", make_opener([[]]))
    empty = fetch_dataset_items("ds123456", "tok", sleep=no_sleep)
    assert empty.items == [] and empty.error is None

    monkeypatch.setattr(fetching, "_request",
                        make_opener([[]], fail_first=99, fail_with=TimeoutError("x")))
    failed = fetch_dataset_items("ds123456", "tok", max_retries=1, sleep=no_sleep)
    assert failed.items == [] and failed.error is not None


# --- redaction ------------------------------------------------------------

def test_redact_removes_credentials():
    """The shipped fallback chunked the Actor's own input, token included, and
    pushed it to the output dataset."""
    out = redact({"token": "apify_api_xxx", "text": "keep me",
                  "nested": {"api_key": "sk-secret", "ok": 1}})
    assert out["token"] == "<redacted>"
    assert out["nested"]["api_key"] == "<redacted>"
    assert out["text"] == "keep me"
    assert out["nested"]["ok"] == 1


@pytest.mark.parametrize("key", [
    "token", "TOKEN", "apiToken", "api_key", "apikey", "password",
    "secret", "Authorization", "access_token",
])
def test_redact_is_case_and_style_insensitive(key):
    assert redact({key: "sensitive"})[key] == "<redacted>"


def test_redact_walks_lists():
    out = redact([{"token": "x"}, {"text": "y"}])
    assert out[0]["token"] == "<redacted>"
    assert out[1]["text"] == "y"


def test_redact_passes_scalars_through():
    assert redact("plain") == "plain"
    assert redact(42) == 42
    assert redact(None) is None
