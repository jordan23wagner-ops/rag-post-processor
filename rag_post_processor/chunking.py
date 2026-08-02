"""Structure-aware, token-budgeted chunking.

Three things distinguish this from `RecursiveCharacterTextSplitter` and from
the previous implementation in this Actor:

  1. **Budgets are in tokens.** A character budget is not what any embedding
     model limits on. Measured on cl100k_base, a 1000-character chunk is
     117-160 tokens of English and 1080 tokens of Japanese -- a 9x spread from
     an identical setting.

  2. **Blocks are respected.** Fenced code, tables and list items are parsed as
     units. Code that must be split keeps its fence and language on every part;
     a table that must be split repeats its header row on every part, so each
     chunk is independently interpretable after retrieval.

  3. **Termination is structural, not heuristic.** The previous version guarded
     a `while start < len(text)` loop with a chunk-count bound derived from an
     assumed stride. Sentence-aware shortening made the real stride smaller
     than assumed, so 16 of 40 schema-legal (chunk_size, overlap) pairs raised
     and the item was discarded *after being billed* -- including every
     50%-overlap run at chunk_size <= 1000. Here the input is decomposed into a
     finite list of units and each iteration consumes at least one, so the loop
     cannot run away and no arbitrary bound is needed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .tokens import Encoder, split_to_token_budget

# --------------------------------------------------------------------------
# Block parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)\s*$")
_ATX_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_H1 = re.compile(r"^\s{0,3}={2,}\s*$")
_SETEXT_H2 = re.compile(r"^\s{0,3}-{3,}\s*$")
_TABLE_ROW = re.compile(r"^\s{0,3}\|.*\|?\s*$")
_TABLE_DIVIDER = re.compile(r"^\s{0,3}\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_LIST_ITEM = re.compile(r"^(\s*)(?:[-*+]|\d{1,3}[.)])\s+")
_QUOTE = re.compile(r"^\s{0,3}>\s?")


@dataclass
class Block:
    kind: str                      # heading | code | table | list | quote | paragraph
    lines: List[str]
    level: int = 0                 # heading level, when kind == "heading"
    lang: str = ""                 # fence language, when kind == "code"

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")


def parse_blocks(text: str) -> List[Block]:
    """Decompose cleaned text into structural blocks.

    Fenced code is captured verbatim, including blank lines, so indentation and
    formatting survive to the output.
    """
    blocks: List[Block] = []
    lines = text.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            marker, lang = fence.group(1), fence.group(2)
            body: List[str] = []
            i += 1
            closed = False
            while i < n:
                close = _FENCE.match(lines[i])
                if close and close.group(1)[0] == marker[0] and len(close.group(1)) >= len(marker):
                    i += 1
                    closed = True
                    break
                body.append(lines[i])
                i += 1
            # An unclosed fence is common in truncated scrapes; treat the rest
            # of the document as code rather than losing the fence entirely.
            blocks.append(Block(kind="code", lines=body, lang=lang))
            if not closed and body:
                pass
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            blocks.append(
                Block(kind="heading", lines=[heading.group(2).strip()],
                      level=len(heading.group(1)))
            )
            i += 1
            continue

        # Setext headings: a line of text followed by ===== or -----
        if i + 1 < n and line.strip():
            if _SETEXT_H1.match(lines[i + 1]):
                blocks.append(Block(kind="heading", lines=[line.strip()], level=1))
                i += 2
                continue
            if _SETEXT_H2.match(lines[i + 1]) and not _LIST_ITEM.match(line):
                blocks.append(Block(kind="heading", lines=[line.strip()], level=2))
                i += 2
                continue

        if _TABLE_ROW.match(line):
            rows: List[str] = []
            while i < n and _TABLE_ROW.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="table", lines=rows))
            continue

        if _LIST_ITEM.match(line):
            items: List[str] = []
            while i < n and (_LIST_ITEM.match(lines[i]) or
                             (lines[i].strip() and lines[i].startswith((" ", "\t")))):
                items.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="list", lines=items))
            continue

        if _QUOTE.match(line):
            quoted: List[str] = []
            while i < n and _QUOTE.match(lines[i]):
                quoted.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="quote", lines=quoted))
            continue

        para: List[str] = []
        while i < n and lines[i].strip():
            if (_FENCE.match(lines[i]) or _ATX_HEADING.match(lines[i])
                    or _TABLE_ROW.match(lines[i]) or _LIST_ITEM.match(lines[i])):
                break
            para.append(lines[i].rstrip())
            i += 1
        if para:
            blocks.append(Block(kind="paragraph", lines=para))
        else:
            i += 1

    return blocks


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

# Abbreviations that end in a period and are followed by a space. The previous
# implementation only recognized '. ' as a boundary and had no abbreviation
# guard at all, so it split inside "Dr. Smith" and "approx. 3%" and ignored
# '!', '?' and '.\n' entirely.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "sr", "jr",
    "inc", "ltd", "co", "corp", "dept", "univ", "est", "fig", "figs",
    "no", "nos", "vol", "vols", "ed", "eds", "pp", "al", "etc", "vs", "viz",
    "cf", "ibid", "approx", "min", "max", "avg", "sec", "eq", "ref", "refs",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "ph.d", "b.a", "m.a",
}

_SENT_BOUNDARY = re.compile(r'(?<=[.!?…])(["\'’”)\]]*)(\s+)')
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)[.!?…][\"'’”)\]]*$")

# CJK and other scripts terminate sentences without a following space, so the
# whitespace-anchored rule above never fires. Measured: Japanese runs ~0.9
# chars/token, so a document with no recognised boundary became one enormous
# unbreakable unit.
_CJK_BOUNDARY = re.compile(r"(?<=[。．！？；…])(?![。．！？；…])")


def _is_false_boundary(left: str) -> bool:
    """True when a period is an abbreviation/initial, not a sentence end."""
    m = _WORD_BEFORE.search(left)
    if not m:
        return False
    word = m.group(1).rstrip(".").lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1:                       # initials: "J. Wagner"
        return True
    if len(word) <= 3 and word.isupper():    # "U.S. Inc"
        return True
    return False


def _split_latin_sentences(text: str) -> List[str]:
    out: List[str] = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.end(1)
        left = text[start:end]
        if _is_false_boundary(left):
            continue
        piece = left.strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences.

    Honours '.', '!', '?', '…' followed by whitespace (with an abbreviation
    guard), newline-terminated sentences, and CJK full stops that have no
    following space. The shipped implementation recognised only '. '.
    """
    if not text.strip():
        return []
    out: List[str] = []
    for segment in _CJK_BOUNDARY.split(text):
        if not segment.strip():
            continue
        out.extend(_split_latin_sentences(segment))
    return out or [text.strip()]


# --------------------------------------------------------------------------
# Unit construction
# --------------------------------------------------------------------------

@dataclass
class Unit:
    """One indivisible piece of output text, with the heading path it sits under."""
    text: str
    kind: str
    heading_path: List[str] = field(default_factory=list)
    tokens: int = 0
    starts_section: bool = False
    section_level: int = 0


def _fence_wrap(lines: Sequence[str], lang: str, part: int, total: int) -> str:
    label = lang or ""
    header = f"```{label}"
    if total > 1:
        header = f"```{label}\n# (part {part} of {total})" if label else f"```\n# (part {part} of {total})"
    return f"{header}\n" + "\n".join(lines) + "\n```"


def _split_code(block: Block, encoder: Encoder, budget: int) -> List[str]:
    """Split a code block on line boundaries, re-fencing every part."""
    whole = _fence_wrap(block.lines, block.lang, 1, 1)
    if encoder.count(whole) <= budget:
        return [whole]

    groups: List[List[str]] = []
    current: List[str] = []
    for line in block.lines:
        trial = current + [line]
        if current and encoder.count(_fence_wrap(trial, block.lang, 1, 2)) > budget:
            groups.append(current)
            current = [line]
        else:
            current = trial
    if current:
        groups.append(current)

    total = len(groups)
    return [_fence_wrap(g, block.lang, i + 1, total) for i, g in enumerate(groups)]


def _split_table(block: Block, encoder: Encoder, budget: int) -> List[str]:
    """Split a table on row boundaries, repeating the header on every part.

    Without this, a table split across chunks leaves every part after the first
    as a wall of unlabeled pipe-separated values -- unusable after retrieval.
    """
    whole = "\n".join(block.lines)
    if encoder.count(whole) <= budget:
        return [whole]

    header: List[str] = []
    body = list(block.lines)
    if body and _TABLE_ROW.match(body[0]):
        header = [body[0]]
        if len(body) > 1 and _TABLE_DIVIDER.match(body[1]):
            header.append(body[1])
        body = body[len(header):]

    parts: List[str] = []
    current: List[str] = []
    for row in body:
        trial = current + [row]
        if current and encoder.count("\n".join(header + trial)) > budget:
            parts.append("\n".join(header + current))
            current = [row]
        else:
            current = trial
    if current:
        parts.append("\n".join(header + current))
    return parts or [whole]


def _split_list(block: Block, encoder: Encoder, budget: int) -> List[str]:
    """Split a list on item boundaries, never mid-item."""
    whole = "\n".join(block.lines)
    if encoder.count(whole) <= budget:
        return [whole]

    items: List[List[str]] = []
    for line in block.lines:
        if _LIST_ITEM.match(line) or not items:
            items.append([line])
        else:
            items[-1].append(line)

    parts: List[str] = []
    current: List[str] = []
    for item in items:
        trial = current + item
        if current and encoder.count("\n".join(trial)) > budget:
            parts.append("\n".join(current))
            current = list(item)
        else:
            current = trial
    if current:
        parts.append("\n".join(current))

    out: List[str] = []
    for p in parts:
        out.extend(split_to_token_budget(p, encoder, budget) if encoder.count(p) > budget else [p])
    return out


def _split_prose(text: str, encoder: Encoder, budget: int) -> List[str]:
    """Sentence-first, then word-boundary. Never cuts mid-word."""
    if encoder.count(text) <= budget:
        return [text]
    out: List[str] = []
    for sentence in split_sentences(text):
        if encoder.count(sentence) <= budget:
            out.append(sentence)
        else:
            out.extend(split_to_token_budget(sentence, encoder, budget))
    return out


def build_units(blocks: Sequence[Block], encoder: Encoder, budget: int) -> List[Unit]:
    """Flatten blocks into a finite list of budget-sized units."""
    units: List[Unit] = []
    stack: List[tuple] = []   # (level, title)

    for block in blocks:
        if block.kind == "heading":
            level = block.level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text))
            units.append(Unit(
                text=("#" * level + " " + block.text),
                kind="heading",
                heading_path=[t for _, t in stack],
                tokens=encoder.count("#" * level + " " + block.text),
                starts_section=True,
                section_level=level,
            ))
            continue

        path = [t for _, t in stack]
        if block.kind == "code":
            pieces = _split_code(block, encoder, budget)
        elif block.kind == "table":
            pieces = _split_table(block, encoder, budget)
        elif block.kind == "list":
            pieces = _split_list(block, encoder, budget)
        else:
            pieces = _split_prose(block.text, encoder, budget)

        for piece in pieces:
            if piece.strip():
                units.append(Unit(
                    text=piece.strip("\n"),
                    kind=block.kind,
                    heading_path=list(path),
                    tokens=encoder.count(piece),
                ))
    return units


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    tokens: int
    heading_path: List[str]
    block_kinds: List[str]
    overlap_tokens: int = 0

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


_OVERLAPPABLE = {"paragraph", "quote", "list"}


def _breadcrumb(path: Sequence[str]) -> str:
    return " > ".join(p for p in path if p)


def _common_prefix(paths: Sequence[Sequence[str]]) -> List[str]:
    """Deepest heading path shared by every unit in a chunk.

    A chunk that spans two sibling sections belongs to their common parent, not
    to whichever section happened to come first.
    """
    paths = [list(p) for p in paths if p]
    if not paths:
        return []
    prefix = list(paths[0])
    for path in paths[1:]:
        keep = 0
        for a, b in zip(prefix, path):
            if a != b:
                break
            keep += 1
        prefix = prefix[:keep]
        if not prefix:
            break
    return prefix


def chunk_document(
    text: str,
    encoder: Encoder,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    min_tokens: int = 24,
    split_on_heading_level: int = 0,
    include_heading_context: bool = True,
) -> List[Chunk]:
    """Chunk one cleaned document.

    split_on_heading_level: 0 disables; N forces a new chunk at every heading of
    level <= N, so a section never bleeds into its neighbour.
    include_heading_context: prefix each chunk with its heading breadcrumb so
    the embedded text carries the context a bare passage loses.
    """
    if not text or not text.strip():
        return []

    max_tokens = max(16, int(max_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), max_tokens // 2))
    min_tokens = max(0, int(min_tokens))

    # Reserve room so the breadcrumb never pushes a chunk over the user's
    # budget. Measured per-path rather than assumed, so short heading paths
    # don't waste budget.
    def reserve_for(path: Sequence[str]) -> int:
        if not include_heading_context or not path:
            return 0
        return encoder.count(_breadcrumb(path)) + 2

    # Units are built against the tightest plausible budget so that no single
    # unit can be unplaceable later. Capped as a fraction of the budget so a
    # small max_tokens is not consumed entirely by breadcrumb reserve.
    max_reserve = min(32, max_tokens // 4) if include_heading_context else 0
    body_budget = max(8, max_tokens - max_reserve)

    blocks = parse_blocks(text)
    units = build_units(blocks, encoder, body_budget)
    if not units:
        return []

    chunks: List[Chunk] = []
    current: List[Unit] = []
    carried = 0

    def flush() -> None:
        nonlocal current, carried
        if not current:
            return
        body = "\n\n".join(u.text for u in current).strip()
        if not body:
            current = []
            carried = 0
            return
        content = [u for u in current if not getattr(u, "_is_overlap", False)] or current
        path = _common_prefix([u.heading_path for u in content])
        prefix = ""
        if include_heading_context and path:
            # If the chunk already opens with its own leaf heading, prepend only
            # the ancestors, so context is added without duplicating a line.
            visible = path[:-1] if body.lstrip("#").strip().startswith(path[-1]) else path
            crumb = _breadcrumb(visible)
            if crumb:
                prefix = f"{crumb}\n\n"
        final = prefix + body
        # Hard guarantee: the emitted chunk never exceeds the user's budget.
        # If the breadcrumb would push it over, drop the breadcrumb -- the
        # heading path is still on the output row.
        if prefix and encoder.count(final) > max_tokens:
            final = body
        chunks.append(Chunk(
            text=final,
            tokens=encoder.count(final),
            heading_path=list(path),
            block_kinds=sorted({u.kind for u in current}),
            overlap_tokens=carried,
        ))
        current = []
        carried = 0

    def current_tokens() -> int:
        if not current:
            return 0
        body = sum(u.tokens for u in current) + 2 * (len(current) - 1)
        content = [u for u in current if not getattr(u, "_is_overlap", False)] or current
        return body + reserve_for(_common_prefix([u.heading_path for u in content]))

    def make_overlap(prev: Sequence[Unit]) -> List[Unit]:
        """Carry trailing prose units into the next chunk, measured in tokens.

        Only whole units are carried, so a chunk never begins mid-sentence --
        the previous implementation began 13 of 17 chunks mid-word.
        """
        if overlap_tokens <= 0:
            return []
        carry: List[Unit] = []
        total = 0
        for u in reversed(prev):
            if u.kind not in _OVERLAPPABLE:
                break
            if total + u.tokens > overlap_tokens:
                # Try a partial carry at sentence granularity.
                if u.kind == "paragraph" and not carry:
                    sents = split_sentences(u.text)
                    tail: List[str] = []
                    t = 0
                    for s in reversed(sents):
                        st = encoder.count(s)
                        if t + st > overlap_tokens:
                            break
                        tail.insert(0, s)
                        t += st
                    if tail:
                        carry.insert(0, Unit(" ".join(tail), u.kind, list(u.heading_path), t))
                break
            carry.insert(0, u)
            total += u.tokens
        for u in carry:
            setattr(u, "_is_overlap", True)
        return carry

    def projected_with(unit: Unit) -> int:
        trial = current + [unit]
        body = sum(u.tokens for u in trial) + 2 * (len(trial) - 1)
        content = [u for u in trial if not getattr(u, "_is_overlap", False)] or trial
        return body + reserve_for(_common_prefix([u.heading_path for u in content]))

    def strip_trailing_headings() -> List[Unit]:
        """Headings belong to what follows them, not to what precedes them.

        Without this, a chunk ends with a bare '## Troubleshooting' and the
        troubleshooting text lands in the next chunk with its heading gone.
        """
        held: List[Unit] = []
        while current and current[-1].kind == "heading":
            held.insert(0, current.pop())
        return held

    for unit in units:
        forced = (
            split_on_heading_level > 0
            and unit.starts_section
            and unit.section_level <= split_on_heading_level
            and bool(current)
        )
        if forced:
            held = strip_trailing_headings()
            if current:
                flush()
            # No overlap across an explicit section break.
            current = list(held)
            carried = 0

        if current and projected_with(unit) > max_tokens:
            held = strip_trailing_headings()
            if current:
                prev = list(current)
                flush()
                # No overlap when the next chunk opens a new section: lead-in
                # text from the previous section is noise under a new heading.
                carry = [] if held else make_overlap(prev)
                current = list(carry) + held
                carried = sum(u.tokens for u in carry)
            else:
                current = held

        current.append(unit)

    flush()

    # A heading with nothing after it produces a chunk that is only a heading;
    # fold it forward rather than emitting a contentless row.
    merged: List[Chunk] = []
    for c in chunks:
        if (merged and c.tokens < min_tokens
                and merged[-1].tokens + c.tokens <= max_tokens):
            prev = merged[-1]
            text_ = prev.text + "\n\n" + c.text
            merged[-1] = Chunk(
                text=text_,
                tokens=encoder.count(text_),
                heading_path=prev.heading_path,
                block_kinds=sorted(set(prev.block_kinds) | set(c.block_kinds)),
                overlap_tokens=prev.overlap_tokens,
            )
        else:
            merged.append(c)

    # Never drop text: if the very first chunk is under min_tokens and is the
    # only one, keep it anyway.
    return merged
