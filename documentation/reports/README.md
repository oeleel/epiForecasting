# Weekly meeting reports

One folder per week, one `report.yaml` per folder, one builder. Every report gets the
same shell (shadcn/ui look, Tailwind from the Play CDN, no build step) so the weekly work is
only the content. The built `index.html` is self-contained: it opens from GitHub Pages and
from disk (`file://`) alike. The reports target desktop viewing only (a 1200-1440px viewport);
wide tables and side-by-side panels are fine, and no effort goes into phone layouts.

```
documentation/reports/
  README.md              this file
  index.html             generated: every week, newest first
  _template/
    shell.html           page shell with {{PLACEHOLDERS}}
    tokens.css           shadcn variables + component recipes (documented at the top)
    widgets.js           interactive panels: bankExplorer, runStepper, sweepCharts, ...
  2026-10-01/
    report.yaml          the content
    data/*.json          what the widgets draw
    index.html           generated
```

## URLs

Pages serves the repo root, so a week lives at

    https://oeleel.github.io/epiForecasting/documentation/reports/YYYY-MM-DD/

and the list of all weeks at

    https://oeleel.github.io/epiForecasting/documentation/reports/

(`.nojekyll` at the repo root keeps Pages from mangling the HTML.)

## Add a week

1. Copy the most recent folder to `documentation/reports/YYYY-MM-DD/` (the meeting date).
2. Edit `report.yaml`: `title`, `date` (must equal the folder name), `summary`, `sections`.
   Keep the section order; it is what makes weeks comparable.
3. Drop the week's data into `data/` as JSON. Current producers:
   - `python -m agent knowledge list --json` style exports for the bank entries
   - the experiment harness summary for `sweep.json`
   - a run's iterations (from `runs.db` or the run report JSON) for `live_run.json`
4. Build, then rebuild the index:

   ```bash
   python scripts/reports/build_report.py documentation/reports/YYYY-MM-DD
   python scripts/reports/build_report.py --index
   # or everything at once
   python scripts/reports/build_report.py --all
   ```

5. Open `documentation/reports/YYYY-MM-DD/index.html` locally in both themes, commit the
   folder plus the regenerated `documentation/reports/index.html`.

The builder fails loudly on an unknown section kind, a missing data file, a `date` that does
not match the folder, an unfilled placeholder, or an em dash anywhere (use `-`).

## Section kinds

Every section is `{id, heading, kind, ...}`. `id` is lower-case with dashes and becomes the
anchor. A section renders an eyebrow above its heading: `label:` if given (a short word or
two), else the two-digit section number. Any block may also carry `intro` and `note`
(markdown rendered above and below).

| kind | fields | renders as |
|---|---|---|
| `markdown` | `body` | prose (see the subset below) |
| `table` | `columns: [..]`, `rows: [[..], ..]` | shadcn table in a horizontal-scroll wrapper; column specs below |
| `cards` | `cards: [{title, badge?, body, tone?}]`, `columns?` (1-3), `tone?` | card grid, badge top right; a tone adds a left accent and colours the badge |
| `stats` | `stats: [{value, label, tone?}]` | four-up stat tiles, value in primary (or the tone) |
| `callout` | `title?`, `body`, `tone?` | Alert box; a tone adds a left border and a soft background |
| `code` | `code` | code block, `#` lines dimmed |
| `accordion` | `items: [{title, body}]` | details/summary accordion |
| `widget` | `name`, `data?`, `options?` | a `widgets.js` function mounted on a div |
| `tabs` | `tabs: [{id, label, blocks: [..]}]` | shadcn Tabs; each pane holds blocks |
| `stack` | `blocks: [..]` | several blocks in one section |

`tone` is one of `info`, `success`, `warning`, `destructive`. On `cards`, a block-level tone
applies to every card and a card's own `tone` wins. Anything else fails the build.

Markdown subset: paragraphs separated by blank lines, `- ` bullet lists, `**bold**`,
`` `inline code` ``, `[text](url)` (http(s), relative or `#anchor` only), and four semantic
marks:

| mark | renders as | use for |
|---|---|---|
| `[[+text]]` | emerald, semibold | an improvement, a confirmation |
| `[[-text]]` | red, semibold | a regression, a negative finding |
| `[[!text]]` | amber, semibold | a caveat, something not done |
| `[[*text]]` | soft primary highlight | the sentence to remember |

Everything is HTML-escaped first; raw HTML is not supported on purpose. A mark may contain
`**bold**` but not a code span (code is split out before the marks are applied).

### Table columns

A column is a plain label or a mapping `{label, format?, better?, baseline?, tones?}`:

| format | cell | renders as |
|---|---|---|
| `text` (default) | inline markdown | left-aligned text |
| `num` | a number | right-aligned tabular number |
| `delta` | a number | `num`, coloured by sign: negative is good (`better: lower`, the default) or positive is good (`better: higher`); a triangle shows the direction |
| `delta` + `baseline: <row index>` | a number | compared against that row's value instead of zero; the baseline row is highlighted and `better: zero` (nearer zero wins) is allowed |
| `badge` | short text | a Badge, coloured by `tones: {text: tone}`; unlisted text is an outline badge |

A non-numeric cell in a `num` or `delta` column fails the build.

## Widgets

| name | data | shows |
|---|---|---|
| `bankExplorer` | bank entries (list) | the KNOWN FACTS block for a chosen phase / model / metric, real matching and ordering rules, click a line for the full entry |
| `runStepper` | one run `{run_id, cutoff, iterations}`; `options.entries` names the bank data key for cross-references | diagnosis, retrieval, proposal, citations, outcome per iteration |
| `sweepCharts` | sweep rows | peak bias and peak WIS against weight, training-window bars |
| `sweepTable` | sweep rows | metric-switchable table with deltas against the baseline |
| `architectureDiagram` | none | data, pipeline, YAML, knowledge.db, experiment log, loop, runs.db; click a box |
| `bankLifecycle` | none | add, rebuild, retrieve, cite, remove; click a stage for the command |
| `schemaReference` | bank entries; `options.examples` picks ids | field table, constants, ordering rule, two entries in full |

Data files are embedded into the page under their file stem (`data/sweep.json` becomes
`REPORT_DATA.sweep`), so two files in one report cannot share a stem.

To add a widget: write `fn(el, data, options, allData)` in `widgets.js`, add it to the
`WIDGETS` map at the bottom, and reference it by name in `report.yaml`. Build text with DOM
text nodes (the `h()` helper), never `innerHTML` with data.

## Style

`tokens.css` documents each shadcn recipe (Card, Badge, Tabs, Table, Accordion, Button, Alert,
Separator) next to the utility string it reproduces, so a new piece of markup can either use
the class or paste the utilities. Colours are the zinc palette in light and dark; the theme
follows the OS and the header button pins a choice. Tables and code blocks scroll inside their
own wrapper.

On top of the neutrals sits a small semantic layer, defined once in `tokens.css` for both
themes with text contrast of at least 4.5:1 and inherited by every report:

| token | colour | means |
|---|---|---|
| `--success` | emerald | improvements, "best", a supported proposal, done |
| `--warning` | amber | caveats, not done, experiential `[E]` knowledge, a parameter mismatch |
| `--info` | sky | asks, next steps, derived `[D]` knowledge |
| `--destructive-ink` | red (text-safe) | regressions, the advisory override |
| `--primary` | zinc | curated `[C]` knowledge, the baseline reference, the active stage, eyebrows, KPI values |

Each has a `-bg` variant at 12% alpha for soft backgrounds. The classes (`.delta-pos`,
`.delta-neg`, `.kpi-value`, `.callout-*`, `.badge-*`, `.mark-*`, `.eyebrow`, `.chip-*`) are
listed in the header of `tokens.css`. Colour carries meaning, never decoration: do not add a
tone to a block because it looks plain.

## Tests

`tests/agent/test_report_builder.py` builds a minimal report in a temp dir, checks that an
unknown kind, an unknown tone, a missing data file and a non-numeric delta cell raise, that
the inline marks and the table formats render the right classes, that every semantic text
colour in `tokens.css` clears 4.5:1 on its background in both themes, that the index lists
weeks newest first, and that this week's rendered page has no `{{` left and no em dash.
