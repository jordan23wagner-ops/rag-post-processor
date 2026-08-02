"""Text cleaning.

Replaces the previous four-regex pipeline, which had three defects that put
garbage into customers' vector indexes:

  1. `<[^>]+>` -> ' ' removed the *tags* but not what was *between* them, so
     `<script>` and `<style>` bodies (GA snippets, CSS rules) were embedded
     verbatim as retrievable content.
  2. `https?://\\S+` -> '' deleted every URL in the body. RAG's worst failure
     mode is a confident answer with no citation, and the link inside the source
     passage is the artifact that fixes it.
  3. `&[a-zA-Z]+;` -> ' ' replaced named entities with a *space* instead of
     decoding them (so `&pound;99` became ` 99` and the price was wrong) and
     ignored numeric entities entirely (`&#8220;` was embedded as literal text).

The whitespace collapse (`\\s+` -> ' ') also flattened every document to one
line, destroying paragraph, list, heading and table structure while leaving the
now-meaningless markdown *syntax* behind. Structure is preserved here because
the chunker downstream needs it.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List, Optional

# Tags whose *contents* are not document text and must be dropped entirely.
_DROP_CONTENT_TAGS = {
    "script", "style", "noscript", "template", "svg", "canvas",
    "head", "iframe", "object", "embed", "map", "audio", "video",
}

# Tags that force a block break in the text output. thead/tbody/tfoot are
# deliberately absent: they are grouping containers, and breaking on them
# would separate a table's header row from its body rows.
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "blockquote", "pre", "form", "fieldset", "figure", "figcaption",
    "table", "ul", "ol", "dl", "hr", "address",
}

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Navigation chrome that is almost never wanted in a RAG index.
_CHROME_TAGS = {"nav", "menu"}

_HTML_HINT = re.compile(
    r"<\s*(?:/\s*)?(?:html|body|div|p|span|br|table|tr|td|th|ul|ol|li|a|h[1-6]|"
    r"script|style|img|section|article|header|footer|strong|em|pre|code)\b[^>]*>",
    re.IGNORECASE,
)


def looks_like_html(text: str) -> bool:
    """True when the input is HTML rather than markdown/plain text.

    Deliberately requires a recognized *structural* tag, so a plain-text
    document that happens to contain `a < b` or `<3` is not mangled by the
    HTML path.
    """
    if not text:
        return False
    return bool(_HTML_HINT.search(text))


class _HTMLToText(HTMLParser):
    """Convert HTML to structured plain text.

    Uses the stdlib parser rather than regex so that malformed markup,
    attributes containing `>`, and comments are handled correctly -- all three
    broke the previous `<[^>]+>` approach.
    """

    def __init__(self, preserve_links: bool = True, drop_nav: bool = True):
        super().__init__(convert_charrefs=True)
        self.preserve_links = preserve_links
        self.drop_nav = drop_nav
        self._parts: List[str] = []
        self._suppress_depth = 0
        self._suppressed_tag: Optional[str] = None
        self._pre_depth = 0
        self._href: Optional[str] = None
        self._link_text: List[str] = []
        self._in_link = False
        self._list_stack: List[str] = []
        self._ol_counter: List[int] = []
        self._in_cell = False
        self._row_cells = 0
        self._row_is_header = False
        self._divider_done = False
        self._table_depth = 0

    # -- helpers ---------------------------------------------------------
    def _emit(self, s: str) -> None:
        if s:
            self._parts.append(s)

    def _break(self, hard: bool = False) -> None:
        self._emit("\n\n" if hard else "\n")

    # -- parser hooks ----------------------------------------------------
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if self._suppress_depth:
            if tag == self._suppressed_tag:
                self._suppress_depth += 1
            return

        if tag in _DROP_CONTENT_TAGS or (self.drop_nav and tag in _CHROME_TAGS):
            self._suppress_depth = 1
            self._suppressed_tag = tag
            return

        if tag == "pre":
            self._pre_depth += 1
            self._break(hard=True)
            self._emit("```\n")
            return

        if tag in _HEADING_TAGS:
            self._break(hard=True)
            self._emit("#" * _HEADING_TAGS[tag] + " ")
            return

        if tag == "br":
            self._emit("\n")
            return

        if tag == "a" and self.preserve_links:
            attr = dict(attrs)
            href = (attr.get("href") or "").strip()
            # Skip in-page anchors, javascript: and empty hrefs -- they are not
            # citations and add noise to the embedded text.
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self._href = href
                self._in_link = True
                self._link_text = []
            return

        if tag in ("ul", "ol"):
            self._list_stack.append(tag)
            self._ol_counter.append(0)
            self._break(hard=True)
            return

        if tag == "li":
            self._emit("\n")
            depth = max(0, len(self._list_stack) - 1)
            indent = "  " * depth
            if self._list_stack and self._list_stack[-1] == "ol":
                self._ol_counter[-1] += 1
                self._emit(f"{indent}{self._ol_counter[-1]}. ")
            else:
                self._emit(f"{indent}- ")
            return

        if tag == "tr":
            # No leading newline: the previous </tr> already emitted one. A
            # blank line between rows would break them into separate blocks,
            # and the chunker's "repeat the header row on every part" rule
            # would never fire for HTML-sourced tables.
            self._emit("| ")
            self._row_cells = 0
            self._row_is_header = False
            self._in_cell = True
            return

        if tag in ("td", "th"):
            self._row_cells += 1
            if tag == "th":
                self._row_is_header = True
            self._in_cell = True
            return

        if tag == "img" and self.preserve_links:
            attr = dict(attrs)
            alt = (attr.get("alt") or "").strip()
            if alt:
                self._emit(f" [image: {alt}] ")
            return

        if tag == "table":
            self._table_depth += 1
            self._divider_done = False
            self._break(hard=True)
            return

        if tag in _BLOCK_TAGS:
            self._break(hard=True)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if self._suppress_depth:
            if tag == self._suppressed_tag:
                self._suppress_depth -= 1
                if self._suppress_depth == 0:
                    self._suppressed_tag = None
            return

        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self._emit("\n```\n")
            return

        if tag == "a" and self._in_link:
            text = "".join(self._link_text).strip()
            href = self._href or ""
            if text and href:
                # Markdown link form keeps the citation attached to the words
                # it belongs to, so the chunker can keep them in the same chunk.
                self._emit(f"[{text}]({href})")
            elif href:
                self._emit(href)
            elif text:
                self._emit(text)
            self._in_link = False
            self._href = None
            self._link_text = []
            return

        if tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            if self._ol_counter:
                self._ol_counter.pop()
            self._break(hard=True)
            return

        if tag in ("td", "th"):
            self._emit(" | ")
            self._in_cell = False
            return

        if tag == "tr":
            self._emit("\n")
            # HTML has no divider row, but markdown needs one for the table to
            # be recognisable -- and the chunker uses it to identify the header
            # it must repeat when a long table is split across chunks.
            if self._row_is_header and self._row_cells and not self._divider_done:
                self._emit("|" + ("---|" * self._row_cells) + "\n")
                self._divider_done = True
            self._in_cell = False
            return

        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            self._break(hard=True)
            return

        if tag in _HEADING_TAGS or tag in _BLOCK_TAGS:
            self._break(hard=True)

    def handle_data(self, data):
        if self._suppress_depth:
            return
        # Whitespace between </tr> and <tr> is markup formatting, not content.
        # Emitting it would put a blank line between rows, splitting one table
        # into several blocks and defeating the header-repeat rule downstream.
        if self._table_depth and not self._in_cell and not data.strip():
            return
        if self._in_link:
            self._link_text.append(data)
            return
        if self._pre_depth:
            self._emit(data)
            return
        self._emit(data)

    def handle_comment(self, data):
        return  # comments are never document text

    def get_text(self) -> str:
        if self._in_link:  # unclosed <a>
            self.handle_endtag("a")
        return "".join(self._parts)


def html_to_text(raw: str, preserve_links: bool = True, drop_nav: bool = True) -> str:
    parser = _HTMLToText(preserve_links=preserve_links, drop_nav=drop_nav)
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        # Malformed beyond recovery: fall back to whatever was parsed so far
        # rather than losing the item.
        pass
    return parser.get_text()


_SPACES = re.compile(r"[^\S\n]+")          # horizontal whitespace only
_TRAILING = re.compile(r"[^\S\n]+\n")
_LEADING = re.compile(r"\n[^\S\n]+")
_MANY_BREAKS = re.compile(r"\n{3,}")
# Zero-width joiners, ZWSP, BOM and soft hyphen. Written as escapes rather
# than literal characters so they are visible to anyone reading this file.
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]{4,}\|?\s*$")


def normalize_whitespace(text: str, preserve_structure: bool = True) -> str:
    """Collapse runs of spaces without destroying line/paragraph structure."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub(" ", text)
    if not preserve_structure:
        return re.sub(r"\s+", " ", text).strip()
    text = _SPACES.sub(" ", text)
    text = _TRAILING.sub("\n", text)
    text = _LEADING.sub("\n", text)
    text = _MANY_BREAKS.sub("\n\n", text)
    return text.strip()


def clean_text(
    text: str,
    *,
    preserve_links: bool = True,
    preserve_structure: bool = True,
    drop_nav: bool = True,
    force_html: Optional[bool] = None,
) -> str:
    """Clean one document.

    force_html=None auto-detects; True/False overrides detection.
    """
    if not text or not text.strip():
        return ""

    is_html = looks_like_html(text) if force_html is None else force_html
    if is_html:
        text = html_to_text(text, preserve_links=preserve_links, drop_nav=drop_nav)
    else:
        # Markdown / plain text still routinely contains entities from a
        # careless scraper. html.unescape handles named AND numeric forms,
        # which the old `&[a-zA-Z]+;` regex did not.
        text = html.unescape(text)

    if not preserve_links:
        text = re.sub(r"\[([^\]]*)\]\((?:https?|ftp)://[^)\s]+\)", r"\1", text)
        text = re.sub(r"\b(?:https?|ftp)://\S+", "", text)

    return normalize_whitespace(text, preserve_structure=preserve_structure)
