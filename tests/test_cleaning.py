"""Cleaning tests.

Every test here corresponds to a defect measured in the shipped version.
"""
import pytest

from rag_post_processor.cleaning import (
    clean_text,
    html_to_text,
    looks_like_html,
    normalize_whitespace,
)


# --- script/style bodies must not reach the index -------------------------

SCRIPTY = """<html><head>
<style>body{margin:0;font-family:Inter}.hdr{color:#fff}</style>
<script>window.dataLayer=[];function gtag(){dataLayer.push(arguments)}</script>
</head><body><h1>Pricing</h1><p>Plans start at &pound;99.</p>
<script>console.log("tracking pixel fired");</script></body></html>"""


def test_script_bodies_are_dropped():
    out = clean_text(SCRIPTY)
    assert "dataLayer" not in out
    assert "gtag" not in out
    assert "tracking pixel" not in out


def test_style_bodies_are_dropped():
    out = clean_text(SCRIPTY)
    assert "font-family" not in out
    assert "margin:0" not in out
    assert "#fff" not in out


def test_document_text_survives_the_script_purge():
    out = clean_text(SCRIPTY)
    assert "Pricing" in out
    assert "Plans start at" in out


def test_noscript_and_template_and_svg_are_dropped():
    html = ("<div><noscript>enable javascript</noscript>"
            "<template>hidden {{row}}</template>"
            "<svg><path d='M0 0'/></svg><p>real text</p></div>")
    out = clean_text(html)
    assert "enable javascript" not in out
    assert "{{row}}" not in out
    assert "M0 0" not in out
    assert "real text" in out


def test_comments_are_dropped():
    out = clean_text("<p>keep<!-- drop this analytics note -->me</p>")
    assert "analytics" not in out
    assert "keep" in out and "me" in out


def test_attribute_containing_angle_bracket_does_not_leak():
    # The old `<[^>]+>` regex mis-parsed this and leaked attribute text.
    out = clean_text('<div data-tip="a > b">visible</div>')
    assert "visible" in out
    assert "data-tip" not in out


# --- URLs are citations, not noise ---------------------------------------

def test_urls_survive_in_plain_text():
    src = "See https://ir.example.com/q3 for the filing."
    assert "https://ir.example.com/q3" in clean_text(src)


def test_anchor_hrefs_become_markdown_links():
    out = clean_text('<p>Read our <a href="https://example.com/terms">terms</a>.</p>')
    assert "[terms](https://example.com/terms)" in out


def test_markdown_links_survive_unchanged():
    src = "See the [official docs](https://docs.example.com/sdk) for details."
    assert "[official docs](https://docs.example.com/sdk)" in clean_text(src)


def test_in_page_anchors_and_javascript_hrefs_are_not_emitted_as_links():
    out = clean_text('<a href="#top">Top</a> <a href="javascript:void(0)">Go</a>')
    assert "#top" not in out
    assert "javascript:" not in out
    assert "Top" in out and "Go" in out


def test_preserve_links_false_strips_urls_but_keeps_link_text():
    src = "See the [official docs](https://docs.example.com/sdk) at https://x.example.com"
    out = clean_text(src, preserve_links=False)
    assert "https://" not in out
    assert "official docs" in out


# --- entity handling ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("&pound;99/month", "£99/month"),
    ("rates &amp; limits", "rates & limits"),
    ("24&nbsp;hours", "24"),                 # nbsp normalises to a space
    ("&#8220;Fair use&#8221;", "“Fair use”"),
    ("&#x27;quoted&#x27;", "'quoted'"),
    ("caf&eacute;", "café"),
])
def test_entities_are_decoded_not_deleted(raw, expected):
    out = clean_text(raw)
    assert expected in out


def test_currency_symbol_is_not_silently_destroyed():
    # Shipped behaviour turned "£99" into " 99" -- a wrong price in the index.
    assert "£99" in clean_text("<p>Plans start at &pound;99/month.</p>")


def test_numeric_entities_do_not_survive_as_literal_text():
    out = clean_text("see &#8220;Fair use&#8221; policy")
    assert "&#8220;" not in out
    assert "&#" not in out


# --- structure preservation ----------------------------------------------

def test_paragraph_breaks_survive():
    src = "First paragraph here.\n\nSecond paragraph here."
    out = clean_text(src)
    assert "\n\n" in out


def test_list_structure_survives():
    src = "Breakdown:\n\n- US: $4.2M\n- EU: 3.1M\n"
    out = clean_text(src)
    assert out.count("\n- ") >= 1 or out.count("- ") >= 2


def test_html_lists_become_markdown_lists():
    out = clean_text("<ul><li>alpha</li><li>beta</li></ul>")
    assert "- alpha" in out
    assert "- beta" in out


def test_html_ordered_lists_are_numbered():
    out = clean_text("<ol><li>first</li><li>second</li></ol>")
    assert "1. first" in out
    assert "2. second" in out


def test_html_headings_become_atx_headings():
    out = clean_text("<h1>Title</h1><h2>Sub</h2><p>body</p>")
    assert "# Title" in out
    assert "## Sub" in out


def test_html_tables_become_pipe_rows():
    out = clean_text("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    assert "|" in out
    assert out.count("\n") >= 1


def test_pre_blocks_become_fenced_code():
    out = clean_text("<pre>def f():\n    return 1</pre>")
    assert "```" in out
    assert "return 1" in out


def test_nav_is_dropped_by_default_and_kept_when_asked():
    html = "<nav>Home About Pricing</nav><p>real content</p>"
    assert "Home About" not in clean_text(html)
    assert "Home About" in clean_text(html, drop_nav=False)


# --- detection and normalisation -----------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("<p>hello</p>", True),
    ("<div class='x'>hi</div>", True),
    ("plain text with a < b comparison", False),
    ("I <3 markdown", False),
    ("# A markdown heading\n\ntext", False),
    ("", False),
])
def test_html_detection(text, expected):
    assert looks_like_html(text) is expected


def test_normalize_collapses_horizontal_space_only():
    out = normalize_whitespace("a    b\n\n\n\nc   d")
    assert out == "a b\n\nc d"


def test_normalize_without_structure_flattens():
    out = normalize_whitespace("a\n\nb", preserve_structure=False)
    assert out == "a b"


def test_empty_and_whitespace_input():
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""


def test_zero_width_and_control_characters_removed():
    out = clean_text("he​llo\x07 world")
    assert "​" not in out
    assert "\x07" not in out
    assert "hello" in out


def test_malformed_html_does_not_raise():
    assert clean_text("<p>unclosed <b>bold <a href='https://x.io'>link") is not None


def test_unclosed_anchor_still_emits_its_text():
    out = clean_text("<p>see <a href='https://x.io/docs'>the docs")
    assert "the docs" in out
    assert "x.io/docs" in out


# --- HTML tables must survive as one markdown table ------------------------

HTML_TABLE = ("<h2>Requirements</h2>"
              "<table><thead><tr><th>Runtime</th><th>Minimum</th></tr></thead><tbody>\n"
              "<tr><td>Node.js</td><td>20.11.0</td></tr>\n"
              "<tr><td>Python</td><td>3.11</td></tr>\n"
              "</tbody></table><p>after</p>")


def test_html_table_rows_are_contiguous():
    """Blank lines between rows would split one table into several blocks and
    defeat the chunker's repeat-the-header rule."""
    out = clean_text(HTML_TABLE)
    rows = [l for l in out.split("\n") if l.startswith("|")]
    assert len(rows) == 4          # header, divider, two body rows
    block = "\n".join(rows)
    assert block in out            # contiguous, no blank line between them


def test_html_table_gets_a_markdown_divider_row():
    out = clean_text(HTML_TABLE)
    lines = [l for l in out.split("\n") if l.startswith("|")]
    assert lines[0] == "| Runtime | Minimum |"
    assert set(lines[1]) <= set("|-")


def test_thead_and_tbody_do_not_split_the_table():
    from rag_post_processor.chunking import parse_blocks
    kinds = [(b.kind, len(b.lines)) for b in parse_blocks(clean_text(HTML_TABLE))]
    tables = [k for k in kinds if k[0] == "table"]
    assert len(tables) == 1
    assert tables[0][1] == 4


def test_divider_is_only_emitted_for_header_rows():
    out = clean_text("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>")
    assert "---" not in out


def test_each_table_gets_its_own_divider():
    two = ("<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
           "<table><tr><th>B</th></tr><tr><td>2</td></tr></table>")
    out = clean_text(two)
    assert out.count("|---|") == 2
