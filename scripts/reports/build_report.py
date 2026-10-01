"""Build the weekly meeting reports (documentation/reports/) from report.yaml.

WHY THIS EXISTS
    Each week's research meeting needs the same artefact: what was asked, what
    was delivered, results, open questions, next steps, reproduce commands, and
    a few interactive proof-of-concept panels. Writing that by hand in HTML
    every week would drift in layout and style. Instead every week is a folder
    `documentation/reports/YYYY-MM-DD/` holding a `report.yaml` (the content)
    and `data/*.json` (what the widgets draw); this script renders them into a
    single self-contained `index.html` using the shared shell in
    `documentation/reports/_template/`.

    The output has to work in two places with no build step: GitHub Pages
    (served from the repo root, so a report lives at
    https://oeleel.github.io/epiForecasting/documentation/reports/YYYY-MM-DD/)
    and from disk over file:// during the meeting. Hence everything (CSS,
    widget code, data) is inlined; the only external loads are the Tailwind
    Play CDN script and the Inter font.

CONTRACT
    report.yaml:
        title:    str      page title, e.g. "Weekly report - 2026-10-01"
        date:     str      YYYY-MM-DD, must equal the folder name
        summary:  str      one or two sentences under the title and on the index
        sections: list of blocks, each with `id` and `heading`

    A section additionally takes `label:` (short eyebrow text above the heading;
    defaults to the two-digit section number).

    A block has a `kind`. Kinds and their fields:
        markdown   body (the small subset: paragraphs, "- " bullets, **bold**,
                   `code`, [text](url), and the semantic marks [[+good]]
                   [[-bad]] [[!caution]] [[*key phrase]])
        table      columns [str | {label, format?, better?, baseline?, tones?}],
                   rows [[cell, ...]] (cells are inline markdown; a column with
                   `format: num` right-aligns tabular numbers, `format: delta`
                   additionally colours them: lower is better unless
                   `better: higher`; with `baseline: <row index>` the colour
                   and arrow compare against that row's value instead of zero,
                   and `better: zero` (nearer zero wins) becomes available;
                   `format: badge` renders the cell as a Badge coloured by
                   `tones: {text: tone}`, unlisted text as an outline badge)
        cards      cards [{title, badge?, body, tone?}], tone? (body is markdown;
                   a block-level tone applies to every card, a card's own wins)
        stats      stats [{value, label, tone?}]
        callout    title?, body (markdown), tone?
        code       code (verbatim text, escaped)
        accordion  items [{title, body}]
        widget     name (a ReportWidgets function), data? (path relative to the
                   report folder), options? (map passed to the widget)
        tabs       tabs [{id, label, blocks: [block, ...]}]
        stack      blocks [block, ...]
    `tone` is one of info | success | warning | destructive and maps onto the
    semantic classes in tokens.css (callout-<tone>, card.tone-<tone>,
    badge-<tone>, ink-<tone>). Colour carries meaning, never decoration.
    Unknown kinds, unknown tones, missing data files, missing required fields,
    non-numeric cells in numeric columns and em dashes (U+2014) anywhere in the
    content fail loudly: a half-built report is worse than no report.

USAGE
    python scripts/reports/build_report.py documentation/reports/2026-10-01
    python scripts/reports/build_report.py --index
    python scripts/reports/build_report.py --all     # every week, then the index
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

__all__ = [
    "ReportError",
    "build_index",
    "build_report",
    "load_report_spec",
    "render_blocks",
    "render_index",
    "render_markdown",
    "render_report",
    "write_index",
    "write_report",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "documentation" / "reports"
TEMPLATE_DIR_NAME = "_template"
REPORT_SPEC_NAME = "report.yaml"
OUTPUT_NAME = "index.html"
REPORT_FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EM_DASH = "\u2014"  # written as an escape so this file itself passes the rule
PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
# Pages URL prefix for the README and the index subtitle.
PAGES_BASE_URL = "https://oeleel.github.io/epiForecasting/documentation/reports/"


class ReportError(ValueError):
    """Anything wrong with a report spec or its data. Always names the block."""


# ---- loading ----------------------------------------------------------------------

def load_report_spec(report_dir: Path) -> Dict[str, Any]:
    """Read and sanity-check report.yaml for one week's folder."""
    spec_path = report_dir / REPORT_SPEC_NAME
    if not spec_path.is_file():
        raise ReportError(f"{spec_path} not found")
    with spec_path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    if not isinstance(spec, dict):
        raise ReportError(f"{spec_path}: top level must be a mapping")
    for key in ("title", "date", "summary", "sections"):
        if key not in spec:
            raise ReportError(f"{spec_path}: missing required key '{key}'")
    if not isinstance(spec["sections"], list) or not spec["sections"]:
        raise ReportError(f"{spec_path}: 'sections' must be a non-empty list")
    if REPORT_FOLDER_PATTERN.match(report_dir.name) and str(spec["date"]) != report_dir.name:
        raise ReportError(
            f"{spec_path}: date {spec['date']!r} does not match folder name {report_dir.name!r}"
        )
    _reject_em_dashes(spec, spec_path)
    return spec


def _reject_em_dashes(obj: Any, where: Path) -> None:
    """Project rule: plain dashes only. Walk every string in the spec."""
    if isinstance(obj, str):
        if EM_DASH in obj:
            raise ReportError(f"{where}: em dash found in {obj[:60]!r}; use '-' instead")
    elif isinstance(obj, dict):
        for value in obj.values():
            _reject_em_dashes(value, where)
    elif isinstance(obj, list):
        for value in obj:
            _reject_em_dashes(value, where)


def _load_template(template_dir: Path, name: str) -> str:
    path = template_dir / name
    if not path.is_file():
        raise ReportError(f"template file {path} not found")
    return path.read_text(encoding="utf-8")


# ---- markdown subset -------------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_SAFE_HREF = re.compile(r"^(https?://|\.{1,2}/|/|#|[\w./-]+$)")
# Semantic marks: [[+good]] [[-bad]] [[!caution]] [[*key phrase]]. Applied to the
# already-escaped text, before links and bold, so the inner text may still use
# **bold** but cannot carry a `code` span (code is split out first).
_MARK = re.compile(r"\[\[([+\-!*])(.+?)\]\]")
MARK_CLASSES: Dict[str, str] = {
    "+": "mark-pos",
    "-": "mark-neg",
    "!": "mark-warn",
    "*": "mark-key",
}
TONES = ("info", "success", "warning", "destructive")


def _check_tone(tone: Any, where: str) -> str | None:
    """Validate an optional `tone:` field. None means untoned."""
    if tone is None:
        return None
    if tone not in TONES:
        raise ReportError(f"{where}: tone {tone!r} must be one of {list(TONES)}")
    return str(tone)


def render_inline(text: str) -> str:
    """Escape, then apply inline code, marks, bold and links. Code wins over bold."""
    escaped = html.escape(str(text), quote=True)
    pieces: List[str] = []
    last = 0
    for match in _INLINE_CODE.finditer(escaped):
        pieces.append(_bold_and_links(escaped[last:match.start()]))
        pieces.append(f'<code class="inline">{match.group(1)}</code>')
        last = match.end()
    pieces.append(_bold_and_links(escaped[last:]))
    return "".join(pieces)


def _bold_and_links(escaped: str) -> str:
    def link(match: "re.Match[str]") -> str:
        label, href = match.group(1), match.group(2)
        if not _SAFE_HREF.match(href):
            raise ReportError(f"link target {href!r} is not an http(s), relative or anchor URL")
        return f'<a href="{href}">{label}</a>'

    def mark(match: "re.Match[str]") -> str:
        return f'<span class="{MARK_CLASSES[match.group(1)]}">{match.group(2)}</span>'

    out = _MARK.sub(mark, escaped)
    out = _LINK.sub(link, out)
    return _BOLD.sub(r"<strong>\1</strong>", out)


def render_markdown(body: str) -> str:
    """Paragraphs separated by blank lines; lines starting with '- ' form a list."""
    blocks: List[str] = []
    paragraph: List[str] = []
    items: List[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if items:
            blocks.append("<ul>" + "".join(f"<li>{render_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in str(body).splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue
        if line.lstrip().startswith("- "):
            flush_paragraph()
            items.append(line.lstrip()[2:])
            continue
        if items and line.startswith("  "):
            # continuation of the previous bullet
            items[-1] += " " + line.strip()
            continue
        flush_list()
        paragraph.append(line.strip())
    flush_paragraph()
    flush_list()
    return "".join(blocks)


# ---- block renderers -------------------------------------------------------------------

class _Context:
    """Per-build state: where data files live and which ones got embedded."""

    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.data: Dict[str, Any] = {}
        self._paths: Dict[str, Path] = {}
        self.widget_counter = 0

    def embed_data(self, rel_path: str, where: str) -> str:
        """Load a JSON data file once and return its REPORT_DATA key (the file stem)."""
        path = (self.report_dir / rel_path).resolve()
        if not path.is_file():
            raise ReportError(f"{where}: data file {rel_path!r} not found under {self.report_dir}")
        key = path.stem
        if key in self._paths and self._paths[key] != path:
            raise ReportError(f"{where}: two data files share the stem {key!r}: {self._paths[key]} and {path}")
        if key in self._paths:
            return key
        with path.open("r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ReportError(f"{where}: {path} is not valid JSON: {exc}") from exc
        if EM_DASH in json.dumps(payload, ensure_ascii=False):
            raise ReportError(f"{where}: em dash found inside {path.name}; use '-' instead")
        self.data[key] = payload
        self._paths[key] = path
        return key


def _require(block: Dict[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise ReportError(f"{where}: kind {block.get('kind')!r} requires '{key}'")
    return block[key]


def _render_markdown_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    return f'<div class="prose">{render_markdown(_require(block, "body", where))}</div>'


TABLE_FORMATS = ("text", "num", "delta", "badge")
TABLE_BETTER = ("lower", "higher", "zero")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")


class _Column:
    """One parsed table column spec: `"label"` or `{label, format?, better?, baseline?, tones?}`.

    format text   inline markdown (the default)
    format num    right-aligned tabular number
    format delta  num, coloured: better/worse than zero (or than row `baseline`)
    format badge  the cell text as a Badge; `tones: {text: tone}` picks the colour,
                  anything unlisted renders as an outline badge
    """

    def __init__(self, spec: Any, where: str) -> None:
        if isinstance(spec, str):
            spec = {"label": spec}
        if not isinstance(spec, dict) or "label" not in spec:
            raise ReportError(f"{where}: a column is a string or a mapping with 'label', got {spec!r}")
        self.label = str(spec["label"])
        self.format = str(spec.get("format", "text"))
        if self.format not in TABLE_FORMATS:
            raise ReportError(f"{where}: column {self.label!r} format {self.format!r} must be one of {list(TABLE_FORMATS)}")
        self.better = str(spec.get("better", "lower"))
        if self.better not in TABLE_BETTER:
            raise ReportError(f"{where}: column {self.label!r} better {self.better!r} must be one of {list(TABLE_BETTER)}")
        self.baseline = spec.get("baseline")
        if self.baseline is not None and not isinstance(self.baseline, int):
            raise ReportError(f"{where}: column {self.label!r} baseline must be a row index, got {self.baseline!r}")
        if self.better == "zero" and self.baseline is None:
            raise ReportError(f"{where}: column {self.label!r}: better: zero needs a baseline row to compare against")
        if self.format != "delta" and ("better" in spec or "baseline" in spec):
            raise ReportError(f"{where}: column {self.label!r}: better/baseline only apply to format: delta")
        tones = spec.get("tones") or {}
        if tones and self.format != "badge":
            raise ReportError(f"{where}: column {self.label!r}: tones only apply to format: badge")
        if not isinstance(tones, dict):
            raise ReportError(f"{where}: column {self.label!r}: tones must be a mapping of cell text to tone")
        self.tones: Dict[str, str] = {str(k): _check_tone(v, f"{where} column {self.label!r}") or "" for k, v in tones.items()}

    @property
    def numeric(self) -> bool:
        return self.format in ("num", "delta")


def _parse_number(cell: Any, column: _Column, where: str) -> float:
    text = str(cell).strip().replace(",", "")
    if not _NUMBER.match(text):
        raise ReportError(f"{where}: column {column.label!r} is {column.format} but cell {cell!r} is not a number")
    return float(text)


def _delta_class(value: float, column: _Column, reference: float) -> str:
    """Colour = better/worse than the reference, arrow = direction of the number."""
    diff = value - reference
    if diff == 0:
        return "delta-flat"
    if column.better == "zero":
        good = abs(value) < abs(reference)
    elif column.better == "higher":
        good = diff > 0
    else:
        good = diff < 0
    arrow = "delta-up" if diff > 0 else "delta-down"
    return f"{'delta-pos' if good else 'delta-neg'} {arrow}"


def _render_cell(cell: Any, column: _Column, row_index: int, col_index: int, rows: List[Any], where: str) -> str:
    """One <td>. Text columns take inline markdown; numeric columns take a plain number."""
    if column.format == "badge":
        tone = column.tones.get(str(cell).strip())
        badge_class = f"badge-{tone}" if tone else "badge-outline"
        return f'<td><span class="badge {badge_class}">{render_inline(str(cell).strip())}</span></td>'
    if not column.numeric:
        return f"<td>{render_inline(cell)}</td>"
    value = _parse_number(cell, column, where)
    text = html.escape(str(cell).strip(), quote=True)
    if column.format == "num":
        return f'<td class="num">{text}</td>'
    if column.baseline is None:
        return f'<td class="num"><span class="{_delta_class(value, column, 0.0)}">{text}</span></td>'
    if column.baseline < 0 or column.baseline >= len(rows):
        raise ReportError(f"{where}: column {column.label!r} baseline row {column.baseline} is out of range")
    if row_index == column.baseline:
        return f'<td class="num">{text}</td>'
    reference = _parse_number(rows[column.baseline][col_index], column, where)
    return f'<td class="num"><span class="{_delta_class(value, column, reference)}">{text}</span></td>'


def _render_table_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    columns = [_Column(c, where) for c in _require(block, "columns", where)]
    rows = _require(block, "rows", where)
    head = "".join(
        f'<th class="num">{render_inline(c.label)}</th>' if c.numeric else f"<th>{render_inline(c.label)}</th>"
        for c in columns
    )
    highlight_rows = {c.baseline for c in columns if c.baseline is not None}
    body_rows = []
    for r, row in enumerate(rows):
        if len(row) != len(columns):
            raise ReportError(f"{where}: table row {row!r} has {len(row)} cells, expected {len(columns)}")
        cells = []
        for col_index, (cell, column) in enumerate(zip(row, columns)):
            cells.append(_render_cell(cell, column, r, col_index, rows, where))
        row_class = ' class="row-highlight"' if r in highlight_rows else ""
        body_rows.append(f"<tr{row_class}>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap rounded-md border"><table class="table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _render_cards_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    cards = _require(block, "cards", where)
    columns = int(block.get("columns", 2))
    grid_class = {1: "grid gap-4", 2: "grid gap-4 md:grid-cols-2", 3: "grid gap-4 md:grid-cols-2 xl:grid-cols-3"}.get(columns)
    if grid_class is None:
        raise ReportError(f"{where}: cards.columns must be 1, 2 or 3, got {columns}")
    block_tone = _check_tone(block.get("tone"), where)
    out = []
    for i, card in enumerate(cards):
        title = render_inline(_require(card, "title", where))
        tone = _check_tone(card.get("tone"), f"{where} > card {i}") or block_tone
        badge = card.get("badge")
        badge_class = f"badge-{tone}" if tone else "badge-secondary"
        badge_html = f'<span class="badge {badge_class}">{render_inline(badge)}</span>' if badge else ""
        body = render_markdown(_require(card, "body", where))
        card_class = f"card tone-{tone}" if tone else "card"
        out.append(
            f'<div class="{card_class}"><div class="card-header !p-5 !pb-3">'
            f'<div class="flex flex-wrap items-start justify-between gap-2"><h3 class="card-title-sm">{title}</h3>{badge_html}</div>'
            f'</div><div class="card-content !p-5 !pt-0 prose text-sm">{body}</div></div>'
        )
    return f'<div class="{grid_class}">{"".join(out)}</div>'


def _render_stats_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    stats = _require(block, "stats", where)
    tiles = []
    for i, s in enumerate(stats):
        tone = _check_tone(s.get("tone"), f"{where} > stat {i}")
        value_class = f"kpi-value ink-{tone}" if tone else "kpi-value"
        card_class = f"card tone-{tone}" if tone else "card"
        tiles.append(
            f'<div class="{card_class}"><div class="card-content !p-5">'
            f'<div class="{value_class}">{render_inline(_require(s, "value", where))}</div>'
            f'<div class="stat-label">{render_inline(_require(s, "label", where))}</div></div></div>'
        )
    return f'<div class="grid gap-4 grid-cols-2 lg:grid-cols-4">{"".join(tiles)}</div>'


def _render_callout_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    title = block.get("title")
    tone = _check_tone(block.get("tone"), where)
    title_html = f'<h5 class="alert-title">{render_inline(title)}</h5>' if title else ""
    body = render_markdown(_require(block, "body", where))
    alert_class = f"alert callout-{tone}" if tone else "alert"
    return f'<div class="{alert_class}" role="note">{title_html}<div class="alert-description prose">{body}</div></div>'


def _render_code_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    text = html.escape(str(_require(block, "code", where)), quote=False)
    # Lines starting with "# " read as comments; colour them without a highlighter.
    lines = [
        f'<span class="comment">{line}</span>' if line.lstrip().startswith("#") else line
        for line in text.splitlines()
    ]
    return f'<pre class="code-block"><code>{chr(10).join(lines)}</code></pre>'


def _render_accordion_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    items = _require(block, "items", where)
    chevron = (
        '<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>'
    )
    out = []
    for item in items:
        title = render_inline(_require(item, "title", where))
        body = render_markdown(_require(item, "body", where))
        out.append(
            f'<details class="accordion-item"><summary class="accordion-trigger"><span>{title}</span>{chevron}</summary>'
            f'<div class="accordion-content prose">{body}</div></details>'
        )
    return f'<div class="accordion border-t">{"".join(out)}</div>'


def _render_widget_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    name = _require(block, "name", where)
    if not re.match(r"^[A-Za-z][A-Za-z0-9]*$", str(name)):
        raise ReportError(f"{where}: widget name {name!r} must be an identifier")
    attrs = [f'data-widget="{html.escape(str(name))}"']
    if block.get("data"):
        key = ctx.embed_data(str(block["data"]), where)
        attrs.append(f'data-source="{html.escape(key)}"')
    options = block.get("options") or {}
    if options:
        if not isinstance(options, dict):
            raise ReportError(f"{where}: widget options must be a mapping")
        attrs.append(f"data-options='{html.escape(json.dumps(options), quote=True)}'")
    ctx.widget_counter += 1
    attrs.append(f'id="widget-{ctx.widget_counter}"')
    return f'<div class="widget min-w-0" {" ".join(attrs)}></div>'


def _render_tabs_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    tabs = _require(block, "tabs", where)
    if not tabs:
        raise ReportError(f"{where}: tabs must not be empty")
    triggers, panels = [], []
    for i, tab in enumerate(tabs):
        tab_id = _require(tab, "id", where)
        label = render_inline(_require(tab, "label", where))
        state = "active" if i == 0 else "inactive"
        safe_id = html.escape(str(tab_id), quote=True)
        triggers.append(
            f'<button type="button" class="tabs-trigger" role="tab" data-tab="{safe_id}" data-state="{state}" '
            f'aria-selected="{"true" if i == 0 else "false"}" aria-controls="panel-{safe_id}" id="tab-{safe_id}">{label}</button>'
        )
        inner = render_blocks(_require(tab, "blocks", where), ctx, f"{where} > tab {tab_id}")
        panels.append(
            f'<div class="tabs-content" role="tabpanel" data-tab="{safe_id}" data-state="{state}" id="panel-{safe_id}" '
            f'aria-labelledby="tab-{safe_id}"{"" if i == 0 else " hidden"}>{inner}</div>'
        )
    return (
        f'<div data-tabs><div class="tabs-list" role="tablist">{"".join(triggers)}</div>{"".join(panels)}</div>'
    )


def _render_stack_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    return render_blocks(_require(block, "blocks", where), ctx, where)


BLOCK_RENDERERS: Dict[str, Callable[[Dict[str, Any], _Context, str], str]] = {
    "markdown": _render_markdown_block,
    "table": _render_table_block,
    "cards": _render_cards_block,
    "stats": _render_stats_block,
    "callout": _render_callout_block,
    "code": _render_code_block,
    "accordion": _render_accordion_block,
    "widget": _render_widget_block,
    "tabs": _render_tabs_block,
    "stack": _render_stack_block,
}


def render_block(block: Dict[str, Any], ctx: _Context, where: str) -> str:
    if not isinstance(block, dict):
        raise ReportError(f"{where}: block must be a mapping, got {type(block).__name__}")
    kind = block.get("kind")
    renderer = BLOCK_RENDERERS.get(str(kind))
    if renderer is None:
        raise ReportError(f"{where}: unknown block kind {kind!r}; known: {sorted(BLOCK_RENDERERS)}")
    parts = []
    if block.get("intro"):
        parts.append(f'<div class="prose prose-muted text-sm">{render_markdown(block["intro"])}</div>')
    parts.append(renderer(block, ctx, where))
    if block.get("note"):
        parts.append(f'<div class="prose prose-muted text-sm">{render_markdown(block["note"])}</div>')
    return "".join(parts) if len(parts) == 1 else f'<div class="space-y-4">{"".join(parts)}</div>'


def render_blocks(blocks: List[Dict[str, Any]], ctx: _Context, where: str) -> str:
    if not isinstance(blocks, list):
        raise ReportError(f"{where}: expected a list of blocks")
    return f'<div class="space-y-4">{"".join(render_block(b, ctx, f"{where}[{i}]") for i, b in enumerate(blocks))}</div>'


def _render_section(section: Dict[str, Any], ctx: _Context, index: int) -> str:
    where = f"sections[{index}]"
    section_id = _require(section, "id", where)
    if not re.match(r"^[a-z][a-z0-9-]*$", str(section_id)):
        raise ReportError(f"{where}: id {section_id!r} must be lower-case letters, digits and dashes")
    heading = render_inline(_require(section, "heading", where))
    # Eyebrow: a short label from report.yaml, else the two-digit section number.
    label = section.get("label")
    eyebrow = render_inline(label) if label else f"{index + 1:02d}"
    body = render_block(section, ctx, f"{where} ({section_id})")
    return (
        f'<section id="{section_id}" class="scroll-mt-24">'
        f'<div class="section-head"><div class="eyebrow">{eyebrow}</div>'
        f'<h2 class="section-title">{heading}<a class="section-anchor" href="#{section_id}" aria-label="Link to this section">#</a></h2></div>'
        f"{body}</section>"
    )


def _render_nav(sections: List[Dict[str, Any]]) -> str:
    return "".join(
        f'<a href="#{html.escape(str(s["id"]))}">{render_inline(s["heading"])}</a>' for s in sections
    )


# ---- page assembly ----------------------------------------------------------------------

def _fill_shell(template_dir: Path, values: Dict[str, str]) -> str:
    shell = _load_template(template_dir, "shell.html")
    values = dict(values)
    values.setdefault("TOKENS_CSS", _load_template(template_dir, "tokens.css"))
    values.setdefault("WIDGETS_JS", _load_template(template_dir, "widgets.js"))
    out = shell
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    leftovers = PLACEHOLDER_PATTERN.findall(out)
    if leftovers:
        raise ReportError(f"shell placeholders left unfilled: {sorted(set(leftovers))}")
    if EM_DASH in out:
        raise ReportError("em dash found in rendered page; the templates must use '-'")
    return out


def _json_for_script(data: Any) -> str:
    # Inside a <script>, "</" could end the element early and "<!--" opens a comment.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")


def render_report(report_dir: Path, template_dir: Path | None = None) -> str:
    """The full HTML page for one week. Pure: reads, never writes."""
    report_dir = Path(report_dir)
    template_dir = Path(template_dir) if template_dir else report_dir.parent / TEMPLATE_DIR_NAME
    spec = load_report_spec(report_dir)
    ctx = _Context(report_dir)
    sections_html = "\n".join(_render_section(s, ctx, i) for i, s in enumerate(spec["sections"]))
    return _fill_shell(template_dir, {
        "TITLE": html.escape(str(spec["title"])),
        "DATE": html.escape(str(spec["date"])),
        "SUMMARY": render_inline(str(spec["summary"])),
        "HOME_HREF": "../index.html",
        "HEADER_LINKS": '<a href="../index.html" class="btn btn-ghost btn-sm">All reports</a>',
        "NAV": _render_nav(spec["sections"]),
        "SECTIONS": sections_html,
        "DATA_JSON": _json_for_script(ctx.data),
    })


def write_report(report_dir: Path, template_dir: Path | None = None) -> Path:
    out_path = Path(report_dir) / OUTPUT_NAME
    out_path.write_text(render_report(report_dir, template_dir), encoding="utf-8")
    return out_path


def build_report(report_dir: Path, template_dir: Path | None = None) -> Path:
    """Alias kept for the CLI and tests: render + write."""
    return write_report(report_dir, template_dir)


# ---- index -------------------------------------------------------------------------------

def _report_folders(reports_dir: Path) -> List[Path]:
    folders = [p for p in Path(reports_dir).iterdir() if p.is_dir() and REPORT_FOLDER_PATTERN.match(p.name)]
    return sorted(folders, key=lambda p: p.name, reverse=True)


def render_index(reports_dir: Path, template_dir: Path | None = None) -> str:
    """One card per YYYY-MM-DD folder, newest first, in the same shell."""
    reports_dir = Path(reports_dir)
    template_dir = Path(template_dir) if template_dir else reports_dir / TEMPLATE_DIR_NAME
    folders = _report_folders(reports_dir)
    cards = []
    for folder in folders:
        spec = load_report_spec(folder)
        href = f"{folder.name}/{OUTPUT_NAME}"
        cards.append(
            '<a class="card block transition-colors hover:bg-accent/40" href="' + html.escape(href) + '">'
            '<div class="card-header !p-5 !pb-3"><div class="flex flex-wrap items-center gap-3">'
            f'<span class="badge badge-outline">{html.escape(str(spec["date"]))}</span>'
            f'<h3 class="card-title-sm">{render_inline(str(spec["title"]))}</h3></div></div>'
            f'<div class="card-content !p-5 !pt-0 text-sm text-muted-foreground">{render_inline(str(spec["summary"]))}</div></a>'
        )
    if not cards:
        body = '<p class="muted text-sm">No reports yet. Add a YYYY-MM-DD folder with a report.yaml and run the builder.</p>'
    else:
        body = f'<div class="grid gap-4">{"".join(cards)}</div>'
    sections_html = (
        '<section id="reports" class="scroll-mt-24"><div class="section-head"><div class="eyebrow">Index</div>'
        '<h2 class="section-title">All weeks</h2></div>'
        f"{body}</section>"
    )
    return _fill_shell(template_dir, {
        "TITLE": "Weekly reports",
        "DATE": f"{len(folders)} report{'' if len(folders) == 1 else 's'}",
        "SUMMARY": "Standing meeting reports for the agentic flu-forecasting project, newest first. Every week uses the same layout; only the content changes.",
        "HOME_HREF": "index.html",
        "HEADER_LINKS": "",
        "NAV": '<a href="#reports">All weeks</a>',
        "SECTIONS": sections_html,
        "DATA_JSON": "{}",
    })


def write_index(reports_dir: Path, template_dir: Path | None = None) -> Path:
    out_path = Path(reports_dir) / OUTPUT_NAME
    out_path.write_text(render_index(reports_dir, template_dir), encoding="utf-8")
    return out_path


def build_index(reports_dir: Path, template_dir: Path | None = None) -> Path:
    return write_index(reports_dir, template_dir)


# ---- CLI -----------------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report_dir", nargs="?", help="documentation/reports/YYYY-MM-DD")
    parser.add_argument("--index", action="store_true", help="regenerate documentation/reports/index.html")
    parser.add_argument("--all", action="store_true", help="rebuild every week's report, then the index")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="override the reports root (tests)")
    args = parser.parse_args(argv)

    reports_dir = Path(args.reports_dir)
    template_dir = reports_dir / TEMPLATE_DIR_NAME
    if not (args.report_dir or args.index or args.all):
        parser.error("give a report folder, --index, or --all")
    try:
        if args.report_dir:
            out = write_report(Path(args.report_dir), template_dir)
            print(f"wrote {out}")
        if args.all:
            for folder in _report_folders(reports_dir):
                print(f"wrote {write_report(folder, template_dir)}")
        if args.index or args.all:
            print(f"wrote {write_index(reports_dir, template_dir)}")
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
