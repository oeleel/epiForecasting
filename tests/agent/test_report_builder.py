"""Unit tests for scripts/reports/build_report.py (the weekly report builder).

What a wrong build would quietly get away with, and is therefore pinned here:
a placeholder left in the page, an em dash slipping through, a widget whose
data file is missing, a typo in a section kind, user text rendered as live
HTML, and an index that lists weeks in the wrong order. The real template
folder is used so the tests also catch a broken shell or widgets.js.

Run with:
    PYTHONPATH=. python tests/agent/test_report_builder.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reports.build_report import (  # noqa: E402
    ReportError,
    build_index,
    build_report,
    render_index,
    render_markdown,
    render_report,
)

TEMPLATE_DIR = PROJECT_ROOT / "documentation" / "reports" / "_template"
THIS_WEEK_DIR = PROJECT_ROOT / "documentation" / "reports" / "2026-10-01"
EM_DASH = "\u2014"  # written as an escape so this file itself passes the rule

# ---- factories --------------------------------------------------------------

MINIMAL_SPEC = """\
title: "Weekly report - {date}"
date: "{date}"
summary: "A tiny report used by the tests."
sections:
  - id: summary
    heading: Summary
    kind: markdown
    body: |-
      First paragraph with **bold** and `code`.

      - one bullet
      - two bullets
  - id: numbers
    heading: Numbers
    kind: table
    columns: [a, b]
    rows:
      - ["1", "2"]
{extra}
"""

WIDGET_SECTION = """\
  - id: widget
    heading: Widget
    kind: widget
    name: sweepTable
    data: data/sweep.json
"""


def _write_report(root: Path, date: str, extra: str = "", with_data: bool = True) -> Path:
    """One YYYY-MM-DD folder with a report.yaml (and a tiny sweep.json)."""
    folder = root / date
    folder.mkdir(parents=True)
    (folder / "report.yaml").write_text(MINIMAL_SPEC.format(date=date, extra=extra), encoding="utf-8")
    if with_data:
        (folder / "data").mkdir()
        (folder / "data" / "sweep.json").write_text(json.dumps([{"config_id": "baseline", "peak": {"wis": 1.0}}]))
    return folder


# ---- tests ------------------------------------------------------------------

def test_build_minimal_report_writes_index_html():
    with tempfile.TemporaryDirectory() as tmp:
        folder = _write_report(Path(tmp), "2026-01-03", extra=WIDGET_SECTION)

        out = build_report(folder, TEMPLATE_DIR)

        html = out.read_text(encoding="utf-8")
        assert out == folder / "index.html"
        assert "<title>Weekly report - 2026-01-03</title>" in html
        assert '<section id="summary"' in html
        assert "<strong>bold</strong>" in html
        assert '<code class="inline">code</code>' in html
        assert "<li>one bullet</li>" in html
        assert 'data-widget="sweepTable"' in html
        assert 'data-source="sweep"' in html
        assert '"config_id":"baseline"' in html
        assert "ReportWidgets" in html


def test_unknown_kind_raises():
    extra = "  - id: odd\n    heading: Odd\n    kind: carousel\n    body: x\n"
    with tempfile.TemporaryDirectory() as tmp:
        folder = _write_report(Path(tmp), "2026-01-03", extra=extra)

        try:
            render_report(folder, TEMPLATE_DIR)
        except ReportError as exc:
            assert "carousel" in str(exc)
            assert "sections[2]" in str(exc)
        else:
            raise AssertionError("unknown kind must raise ReportError")


def test_missing_data_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        folder = _write_report(Path(tmp), "2026-01-03", extra=WIDGET_SECTION, with_data=False)

        try:
            render_report(folder, TEMPLATE_DIR)
        except ReportError as exc:
            assert "data/sweep.json" in str(exc)
        else:
            raise AssertionError("missing data file must raise ReportError")


def test_em_dash_in_spec_raises():
    extra = f"  - id: dash\n    heading: Dash\n    kind: markdown\n    body: \"a {EM_DASH} b\"\n"
    with tempfile.TemporaryDirectory() as tmp:
        folder = _write_report(Path(tmp), "2026-01-03", extra=extra)

        try:
            render_report(folder, TEMPLATE_DIR)
        except ReportError as exc:
            assert "em dash" in str(exc)
        else:
            raise AssertionError("em dash must raise ReportError")


def test_date_must_match_folder_name():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "2026-01-10"
        folder.mkdir()
        (folder / "report.yaml").write_text(MINIMAL_SPEC.format(date="2026-01-03", extra=""), encoding="utf-8")

        try:
            render_report(folder, TEMPLATE_DIR)
        except ReportError as exc:
            assert "does not match folder name" in str(exc)
        else:
            raise AssertionError("date / folder mismatch must raise ReportError")


def test_markdown_escapes_html_in_user_text():
    html = render_markdown("<script>alert(1)</script> and **<b>x</b>**")

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<strong>&lt;b&gt;x&lt;/b&gt;</strong>" in html


def test_markdown_link_rejects_javascript_scheme():
    try:
        render_markdown("[click](javascript:alert(1))")
    except ReportError as exc:
        assert "javascript:" in str(exc)
    else:
        raise AssertionError("javascript: links must raise ReportError")


def test_index_lists_folders_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_report(root, "2026-01-03")
        _write_report(root, "2026-02-14")
        _write_report(root, "2025-12-20")
        (root / "_template").mkdir()  # must be ignored: not a date folder
        (root / "notes").mkdir()

        html = render_index(root, TEMPLATE_DIR)

        first = html.index("2026-02-14/index.html")
        second = html.index("2026-01-03/index.html")
        third = html.index("2025-12-20/index.html")
        assert first < second < third
        assert "_template/index.html" not in html
        assert "3 reports" in html


def test_build_index_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_report(root, "2026-01-03")

        out = build_index(root, TEMPLATE_DIR)

        assert out == root / "index.html"
        assert "Weekly report - 2026-01-03" in out.read_text(encoding="utf-8")


def test_this_weeks_report_has_no_placeholders_or_em_dashes():
    html = render_report(THIS_WEEK_DIR, TEMPLATE_DIR)

    assert "{{" not in html
    assert EM_DASH not in html
    assert '"id":"rectify-peak-loss-weight-approaching-peak"' in html
    assert '"run_id":"20261001-084414-7d97"' in html
    assert '"config_id":"baseline"' in html
    for widget in ("bankExplorer", "runStepper", "sweepCharts", "sweepTable", "bankLifecycle", "schemaReference", "architectureDiagram"):
        assert f'data-widget="{widget}"' in html, widget


def test_render_is_deterministic():
    first = render_report(THIS_WEEK_DIR, TEMPLATE_DIR)
    second = render_report(THIS_WEEK_DIR, TEMPLATE_DIR)

    assert first == second


# ---- semantic colour layer (2026-10-01) -------------------------------------

def _render_section(extra: str) -> str:
    """Render MINIMAL_SPEC plus one extra section and return the page."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = _write_report(Path(tmp), "2026-01-03", extra=extra)
        return render_report(folder, TEMPLATE_DIR)


def _expect_error(extra: str, fragment: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = _write_report(Path(tmp), "2026-01-03", extra=extra)
        try:
            render_report(folder, TEMPLATE_DIR)
        except ReportError as exc:
            assert fragment in str(exc), str(exc)
        else:
            raise AssertionError(f"expected ReportError mentioning {fragment!r}")


def test_section_renders_numbered_eyebrow_by_default():
    html = _render_section("")

    assert '<div class="eyebrow">01</div>' in html
    assert '<div class="eyebrow">02</div>' in html
    assert '<div class="section-head">' in html


def test_section_label_overrides_eyebrow():
    extra = "  - id: lab\n    heading: Labelled\n    label: Results\n    kind: markdown\n    body: x\n"
    html = _render_section(extra)

    assert '<div class="eyebrow">Results</div>' in html
    assert '<div class="eyebrow">03</div>' not in html


def test_inline_marks_render_semantic_spans():
    html = render_markdown("[[+good]] [[-bad]] [[!careful]] [[*remember this]]")

    assert '<span class="mark-pos">good</span>' in html
    assert '<span class="mark-neg">bad</span>' in html
    assert '<span class="mark-warn">careful</span>' in html
    assert '<span class="mark-key">remember this</span>' in html


def test_inline_marks_keep_escaping_and_allow_bold():
    html = render_markdown("[[*<b>x</b> and **bold**]]")

    assert "<b>" not in html
    assert '<span class="mark-key">&lt;b&gt;x&lt;/b&gt; and <strong>bold</strong></span>' in html


def test_unmatched_mark_is_left_as_text():
    html = render_markdown("[[+open and never closed, with **bold** after")

    assert "mark-pos" not in html
    assert "[[+open and never closed" in html
    assert "<strong>bold</strong>" in html


def test_callout_tone_maps_to_class():
    extra = "  - id: c\n    heading: C\n    kind: callout\n    tone: warning\n    title: Caveats\n    body: x\n"
    html = _render_section(extra)

    assert 'class="alert callout-warning"' in html


def test_untoned_callout_keeps_plain_alert():
    extra = "  - id: c\n    heading: C\n    kind: callout\n    body: x\n"
    html = _render_section(extra)
    section = html.split('<section id="c"')[1].split("</section>")[0]

    assert 'class="alert" role="note"' in section
    assert "callout-" not in section


def test_unknown_tone_raises():
    extra = "  - id: c\n    heading: C\n    kind: callout\n    tone: purple\n    body: x\n"
    _expect_error(extra, "purple")


def test_cards_block_tone_applies_and_card_tone_wins():
    extra = (
        "  - id: k\n    heading: K\n    kind: cards\n    tone: info\n    cards:\n"
        "      - {title: A, badge: ask, body: a}\n"
        "      - {title: B, badge: risk, body: b, tone: destructive}\n"
    )
    html = _render_section(extra)

    assert '<div class="card tone-info">' in html
    assert '<span class="badge badge-info">ask</span>' in html
    assert '<div class="card tone-destructive">' in html
    assert '<span class="badge badge-destructive">risk</span>' in html


def test_stats_render_kpi_value_with_optional_tone():
    extra = (
        "  - id: s\n    heading: S\n    kind: stats\n    stats:\n"
        "      - {value: '27', label: entries}\n"
        "      - {value: '369', label: tests, tone: success}\n"
    )
    html = _render_section(extra)

    assert '<div class="kpi-value">27</div>' in html
    assert '<div class="kpi-value ink-success">369</div>' in html
    assert '<div class="card tone-success">' in html


def test_table_delta_colours_by_sign_lower_is_better():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: d, format: delta}]\n"
        "    rows: [[a, '-2.5'], [b, '+1.0'], [c, '0']]\n"
    )
    html = _render_section(extra)

    assert '<span class="delta-pos delta-down">-2.5</span>' in html
    assert '<span class="delta-neg delta-up">+1.0</span>' in html
    assert '<span class="delta-flat">0</span>' in html
    assert '<th class="num">d</th>' in html


def test_table_delta_better_higher_flips_colour():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: d, format: delta, better: higher}]\n"
        "    rows: [[a, '-2.5'], [b, '+1.0']]\n"
    )
    html = _render_section(extra)

    assert '<span class="delta-neg delta-down">-2.5</span>' in html
    assert '<span class="delta-pos delta-up">+1.0</span>' in html


def test_table_delta_with_baseline_row():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns:\n"
        "      - name\n"
        "      - {label: wis, format: delta, baseline: 0}\n"
        "      - {label: bias, format: delta, baseline: 0, better: zero}\n"
        "      - {label: cov, format: delta, baseline: 0, better: higher}\n"
        "    rows:\n"
        "      - [base, '121.01', '-58.8', '0.855']\n"
        "      - [worse, '123.74', '+67.5', '0.746']\n"
        "      - [better, '119.00', '+7.9', '0.900']\n"
    )
    html = _render_section(extra)

    assert '<tr class="row-highlight"><td>base</td><td class="num">121.01</td><td class="num">-58.8</td><td class="num">0.855</td></tr>' in html
    assert '<span class="delta-neg delta-up">123.74</span>' in html
    assert '<span class="delta-neg delta-up">+67.5</span>' in html      # further from zero than -58.8
    assert '<span class="delta-neg delta-down">0.746</span>' in html
    assert '<span class="delta-pos delta-down">119.00</span>' in html
    assert '<span class="delta-pos delta-up">+7.9</span>' in html       # nearer zero than -58.8
    assert '<span class="delta-pos delta-up">0.900</span>' in html


def test_table_num_format_right_aligns_without_colour():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: n, format: num}]\n"
        "    rows: [[a, '1,234.5']]\n"
    )
    html = _render_section(extra)

    assert '<td class="num">1,234.5</td>' in html
    assert "delta-" not in html.split('<section id="t"')[1].split("</section>")[0]


def test_table_badge_format_uses_tones_map():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: status, format: badge, tones: {done: success, 'not done': warning}}]\n"
        "    rows: [[a, done], [b, not done], [c, later]]\n"
    )
    html = _render_section(extra)

    assert '<span class="badge badge-success">done</span>' in html
    assert '<span class="badge badge-warning">not done</span>' in html
    assert '<span class="badge badge-outline">later</span>' in html


def test_table_non_numeric_cell_in_delta_column_raises():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: d, format: delta}]\n"
        "    rows: [[a, 'n/a']]\n"
    )
    _expect_error(extra, "not a number")


def test_table_better_zero_without_baseline_raises():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: d, format: delta, better: zero}]\n"
        "    rows: [[a, '1']]\n"
    )
    _expect_error(extra, "better: zero")


def test_table_unknown_format_raises():
    extra = (
        "  - id: t\n    heading: T\n    kind: table\n"
        "    columns: [name, {label: d, format: money}]\n"
        "    rows: [[a, '1']]\n"
    )
    _expect_error(extra, "money")


def _hsl_tokens(css_block: str) -> dict:
    """{name: (h, s, l)} for every `--name: H S% L%;` line in one CSS block."""
    import re
    out = {}
    for m in re.finditer(r"--([a-z0-9-]+):\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*;", css_block):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def _contrast(a, b) -> float:
    import colorsys

    def lum(hsl):
        r, g, b_ = colorsys.hls_to_rgb(hsl[0] / 360, hsl[2] / 100, hsl[1] / 100)
        lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b_)

    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# text token -> its soft background token
SEMANTIC_TEXT_TOKENS = {
    "success": "success-bg",
    "warning": "warning-bg",
    "info": "info-bg",
    "destructive-ink": "destructive-bg",
}
MIN_TEXT_CONTRAST = 4.5


def test_semantic_text_tokens_clear_aa_contrast_in_both_themes():
    css = (TEMPLATE_DIR / "tokens.css").read_text(encoding="utf-8")
    light_block = css.split(":root {")[1].split("}")[0]
    dark_block = css.split(".dark {")[1].split("}")[0]
    light = _hsl_tokens(light_block)
    dark = _hsl_tokens(dark_block)

    for name, bg_name in SEMANTIC_TEXT_TOKENS.items():
        assert name in light, f"--{name} missing from :root"
        assert name in dark, f"--{name} missing from .dark"
        # Text on the page, and (the same pair inverted) the page-background text on a filled badge.
        assert _contrast(light[name], light["background"]) >= MIN_TEXT_CONTRAST, (name, "light")
        assert _contrast(dark[name], dark["background"]) >= MIN_TEXT_CONTRAST, (name, "dark")
        assert f"--{bg_name}:" in light_block, f"--{bg_name} missing from :root"
        assert f"--{bg_name}:" in dark_block, f"--{bg_name} missing from .dark"


def test_this_weeks_report_uses_the_semantic_layer():
    html = render_report(THIS_WEEK_DIR, TEMPLATE_DIR)

    assert '<span class="mark-key">The weight fixes the level, not the score</span>' in html
    assert '<span class="badge badge-success">done</span>' in html
    assert '<span class="badge badge-warning">not done</span>' in html
    assert 'class="alert callout-warning"' in html
    assert 'class="alert callout-success"' in html
    assert '<div class="card tone-info">' in html
    assert '<span class="delta-neg delta-up">238.26</span>' in html   # window 12 peak WIS vs baseline
    assert '<span class="delta-pos delta-up">-6.2</span>' in html     # calendar lambda 3 bias nearer zero
    assert "delta-good" not in html.split("<main")[1].split("</main>")[0]


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
