"""Apify dataset fetching.

The previous implementation was one line:

    url = f"https://api.apify.com/v2/datasets/{id}/items?token={token}&clean=true&format=json"
    with urllib.request.urlopen(url) as response: ...

with no timeout (so a hung connection burned paid compute indefinitely), no
retry, no pagination, the API token in the query string where it lands in
access logs, and no read of `X-Apify-Pagination-Total` -- so silent truncation
was undetectable even in principle.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

API_BASE = "https://api.apify.com/v2"
PAGE_SIZE = 1000
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 4
BACKOFF_BASE_S = 1.5


@dataclass
class FetchResult:
    items: List[Any] = field(default_factory=list)
    reported_total: Optional[int] = None
    pages: int = 0
    truncated: bool = False
    error: Optional[str] = None

    @property
    def complete(self) -> bool:
        if self.error:
            return False
        if self.reported_total is None:
            return True
        return len(self.items) >= self.reported_total


def _request(url: str, token: str, timeout: int):
    req = urllib.request.Request(url)
    if token:
        # Header auth, not a query parameter: query strings are recorded in
        # access logs, proxy logs and error reports.
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "rag-post-processor/1.0 (+apify actor)")
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_dataset_items(
    dataset_id: str,
    token: str,
    *,
    max_items: Optional[int] = None,
    page_size: int = PAGE_SIZE,
    timeout: int = REQUEST_TIMEOUT_S,
    max_retries: int = MAX_RETRIES,
    log: Optional[Callable[[str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Fetch every item in a dataset, paging until exhausted.

    Retries 5xx and 429 with exponential backoff. Returns a FetchResult so the
    caller can tell "empty dataset" from "fetch failed" from "partially read" --
    the old code collapsed all three into an empty list.
    """
    result = FetchResult()
    offset = 0

    while True:
        remaining = None if max_items is None else max_items - len(result.items)
        if remaining is not None and remaining <= 0:
            result.truncated = True
            break

        limit = page_size if remaining is None else min(page_size, remaining)
        query = urllib.parse.urlencode(
            {"clean": "true", "format": "json", "offset": offset, "limit": limit}
        )
        url = f"{API_BASE}/datasets/{urllib.parse.quote(dataset_id)}/items?{query}"

        page: Optional[List[Any]] = None
        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                with _request(url, token, timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    total_header = response.headers.get("X-Apify-Pagination-Total")
                    if total_header and result.reported_total is None:
                        try:
                            result.reported_total = int(total_header)
                        except ValueError:
                            pass
                data = json.loads(body) if body.strip() else []
                page = data if isinstance(data, list) else []
                break
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    delay = BACKOFF_BASE_S * (2 ** attempt)
                    if log:
                        log(f"Dataset fetch got {last_error}; retrying in {delay:.1f}s "
                            f"(attempt {attempt + 2}/{max_retries}).")
                    sleep(delay)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries - 1:
                    delay = BACKOFF_BASE_S * (2 ** attempt)
                    if log:
                        log(f"Dataset fetch failed ({last_error}); retrying in {delay:.1f}s "
                            f"(attempt {attempt + 2}/{max_retries}).")
                    sleep(delay)
                    continue
                break

        if page is None:
            result.error = last_error or "unknown error"
            break

        result.pages += 1
        result.items.extend(page)

        if len(page) < limit:
            break
        offset += len(page)

    if result.reported_total is not None and len(result.items) < result.reported_total:
        result.truncated = True

    return result


def redact(value: Any) -> Any:
    """Strip credentials from a structure before it is ever logged or emitted.

    The old fallback path (`items = [actor_input]` when the fetch failed) chunked
    the Actor's own input -- including the API token -- and pushed it into the
    output dataset.
    """
    secret_keys = {"token", "apitoken", "api_key", "apikey", "password", "secret",
                   "authorization", "auth", "bearer", "access_token"}
    if isinstance(value, dict):
        return {
            k: ("<redacted>" if k.lower().replace("-", "_") in secret_keys else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
