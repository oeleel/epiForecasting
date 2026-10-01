/*
 * widgets.js - the interactive panels a weekly report can mount.
 *
 * WHY: the reports are static pages (GitHub Pages, or file:// on a laptop in a
 * meeting), so every interactive piece is a plain function that takes a mount
 * element and already-embedded data. No framework, no build step, no fetch.
 * The builder inlines this file into the shell and emits
 *   <div data-widget="bankExplorer" data-source="bank_entries" data-options='{...}'></div>
 * for every `kind: widget` block in report.yaml; `ReportWidgets.init(data)`
 * mounts them all and wires the tabs.
 *
 * Every widget has the signature  fn(el, data, options, allData)  where `data`
 * is the parsed JSON of the block's data file (or undefined for widgets that
 * need none), `options` is the block's `options:` map, and `allData` is the
 * whole embedded map (so one widget can cross-reference another's file).
 *
 * The bank explorer reproduces the real retrieval rules so what the page shows
 * is what the loop shows the LLM:
 *   - an entry matches when its context lacks the key or lists the value
 *   - order: provenance trust (curated 3, derived 2, experiential 1) desc,
 *     confidence (high 3, medium 2, low 1) desc, n_observations desc, id asc
 *   - lines render as "- [C, high] statement -> try action(k=v)" and
 *     "[E, low, 1 run]" for experiential entries
 *   - the proposal limit (30) reserves up to 6 slots for experiential entries
 *
 * All text goes through DOM text nodes (never innerHTML with data), so the
 * embedded JSON cannot inject markup.
 */
(function (global) {
  "use strict";

  // ---- schema constants mirrored from agent/knowledge (keep in sync by hand) ----
  const KNOWN_PHASES = ["onset", "peak", "decline", "surge", "plateau", "approaching_peak", "off_season"];
  const DEFAULT_QUERY_LIMIT = 30;
  const DEFAULT_PROPOSAL_FACTS_LIMIT = 30;
  const DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS = 6;
  const MIN_SEASON_WEEKS = 40;
  const DEFAULT_APPROACHING_PEAK_WEEKS = 6;

  const TRUST = { curated: 3, derived: 2, experiential: 1 };
  const CONF = { high: 3, medium: 2, low: 1 };
  const TAG = { curated: "C", derived: "D", experiential: "E" };

  // ---- tiny DOM helpers -------------------------------------------------------
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v == null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "text") el.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
        else if (k === "dataset") Object.assign(el.dataset, v);
        else el.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return el;
  }
  function svgEl(n, attrs, text) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", n);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function code(text, cls) { return h("code", { class: cls || "inline", text }); }
  function pre(text) { return h("pre", { class: "code-block" }, h("code", { text })); }
  function card(titleText, bodyNodes, opts) {
    const o = opts || {};
    const header = h("div", { class: "card-header" + (o.compact ? " !p-4 !pb-2" : "") },
      h("h4", { class: o.compact ? "card-title-sm" : "card-title", text: titleText }),
      o.description ? h("p", { class: "card-description", text: o.description }) : null);
    const content = h("div", { class: "card-content" + (o.compact ? " !p-4 !pt-0" : "") }, bodyNodes);
    return h("div", { class: "card" }, header, content);
  }
  function field(labelText, control) {
    return h("label", { class: "field" }, h("span", { class: "field-label", text: labelText }), control);
  }
  function select(options, value, onChange) {
    const s = h("select", { class: "select", onChange });
    for (const [v, label] of options) s.appendChild(h("option", { value: v, text: label, selected: v === value }));
    return s;
  }
  function checkbox(labelText, checked, onChange) {
    const input = h("input", { type: "checkbox", class: "checkbox", onChange });
    input.checked = checked;
    return h("label", { class: "checkbox-label" }, input, labelText);
  }
  function fmtNum(v, dec) { return typeof v === "number" ? v.toFixed(dec) : String(v); }

  // ---- bank explorer ----------------------------------------------------------
  function matches(e, ctx) {
    for (const k of ["phase", "model", "metric"]) {
      const want = ctx[k]; if (!want) continue;
      const have = e.context && e.context[k];
      if (have && have.length && !have.includes(want)) return false;
    }
    return true;
  }
  function orderKey(a, b) {
    return (TRUST[b.provenance] - TRUST[a.provenance]) || (CONF[b.confidence] - CONF[a.confidence]) ||
      (((b.evidence || {}).n_observations || 0) - ((a.evidence || {}).n_observations || 0)) || a.id.localeCompare(b.id);
  }
  function fmtParam(v) {
    if (typeof v === "string" && !/^[\w.-]+$/.test(v)) return "'" + v + "'";
    return JSON.stringify(v).replace(/^"|"$/g, "");
  }
  // The recommendation suffix of a KNOWN FACTS line, with its semantic class:
  // an adapter action the loop can take is success, a placeholder is muted.
  function recPart(e) {
    const r = e.payload && e.payload.recommendation; if (!r) return null;
    if (r.action === "not_yet_available") return { text: " -> no adapter action yet", cls: "rec-none" };
    const ps = Object.entries(r.params || {}).map(([k, v]) => k + "=" + fmtParam(v)).join(", ");
    return { text: " -> try " + r.action + "(" + ps + ")", cls: "rec-action" };
  }
  function recText(e) { const r = recPart(e); return r ? r.text : ""; }
  function lineFor(e) {
    const n = (e.evidence || {}).n_observations;
    const tag = e.provenance === "experiential"
      ? "[E, " + e.confidence + ", " + n + " run" + (n === 1 ? "" : "s") + "]"
      : "[" + (TAG[e.provenance] || "?") + ", " + e.confidence + "]";
    const rec = recPart(e);
    // `text` keeps the full line for callers that want one string; `statement` and `rec` split it for colouring.
    return { tag, tagClass: "tag tag-" + (TAG[e.provenance] || "c").toLowerCase(), statement: e.statement, rec, text: e.statement + (rec ? rec.text : "") };
  }
  // The orchestrator's representation guarantee, see Orchestrator._retrieve_for_proposal.
  function applyProposalLimit(ranked, limit, slots) {
    if (ranked.length <= limit) return { shown: ranked, omitted: 0 };
    const experiential = ranked.filter(e => e.provenance === "experiential");
    const others = ranked.filter(e => e.provenance !== "experiential");
    if (!experiential.length) return { shown: ranked.slice(0, limit), omitted: ranked.length - limit };
    let nExp = Math.min(experiential.length, slots);
    const nOthers = Math.min(others.length, limit - nExp);
    nExp = Math.min(experiential.length, limit - nOthers);
    const shown = others.slice(0, nOthers).concat(experiential.slice(0, nExp));
    return { shown, omitted: ranked.length - shown.length };
  }
  function entryDetail(e) {
    const rows = [
      ["id", e.id], ["provenance", e.provenance], ["category", e.category], ["statement", e.statement],
      ["entities", JSON.stringify(e.entities)], ["context", JSON.stringify(e.context)],
      ["payload", JSON.stringify(e.payload, null, 1)], ["evidence", JSON.stringify(e.evidence)],
      ["confidence", e.confidence], ["created_at", e.created_at],
    ];
    const dl = h("dl");
    for (const [k, v] of rows) { dl.appendChild(h("dt", { text: k })); dl.appendChild(h("dd", {}, h("code", { text: v == null ? "null" : String(v) }))); }
    return h("div", { class: "entry-detail" }, dl);
  }
  function uniqueContextValues(entries, key) {
    const s = new Set();
    for (const e of entries) for (const v of ((e.context || {})[key] || [])) s.add(String(v));
    return [...s].sort();
  }

  function bankExplorer(el, entries) {
    entries = entries || [];
    const provenances = [...new Set(entries.map(e => e.provenance))];
    const models = uniqueContextValues(entries, "model");
    const metrics = uniqueContextValues(entries, "metric");
    const state = {
      phase: "peak",
      model: models.includes("xgboost_direct") ? "xgboost_direct" : "",
      metric: "",
      show: Object.fromEntries(provenances.map(p => [p, true])),
      onlyRec: false,
      applyLimit: false,
      selectedId: null,
    };
    const countEl = h("div", { class: "muted text-sm" });
    const out = h("div", { class: "facts", role: "list" });
    const detailHost = h("div", { hidden: true });

    function render() {
      const ctx = { phase: state.phase, model: state.model, metric: state.metric };
      const matching = entries.filter(e => matches(e, ctx)).sort(orderKey);
      let rows = matching.filter(e => state.show[e.provenance] !== false).filter(e => !state.onlyRec || (e.payload && e.payload.recommendation));
      let omitted = 0;
      if (state.applyLimit) { const r = applyProposalLimit(rows, DEFAULT_PROPOSAL_FACTS_LIMIT, DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS); rows = r.shown; omitted = r.omitted; }
      countEl.textContent = rows.length + " shown of " + matching.length + " matching entries (bank holds " + entries.length + ")";
      out.textContent = "";
      out.appendChild(h("div", { text: "KNOWN FACTS (knowledge bank; [C]=curated [D]=derived [E]=experiential):" }));
      if (!rows.length) out.appendChild(h("div", { text: "(knowledge bank: no matching entries)" }));
      for (const e of rows) {
        const { tag, tagClass, statement, rec } = lineFor(e);
        const line = h("span", {
          class: "fact-line" + (e.id === state.selectedId ? " is-selected" : ""), role: "listitem button", tabindex: "0",
          onClick: () => { state.selectedId = e.id; render(); detailHost.hidden = false; detailHost.textContent = ""; detailHost.appendChild(card(e.id, [entryDetail(e)], { compact: true, description: "full entry as stored" })); },
          onKeydown: ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); line.click(); } },
        }, "- ", h("span", { class: tagClass, text: tag }), " " + statement, rec ? h("span", { class: rec.cls, text: rec.text }) : null);
        out.appendChild(line);
      }
      if (omitted) out.appendChild(h("div", { class: "muted", text: "(" + omitted + " more matching entr" + (omitted === 1 ? "y" : "ies") + " not shown: proposal limit " + DEFAULT_PROPOSAL_FACTS_LIMIT + ", " + DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS + " experiential slots)" }));
    }

    const phaseOpts = [["", "any"]].concat(KNOWN_PHASES.map(p => [p, p]));
    const controls = h("div", { class: "grid gap-3 sm:grid-cols-3" },
      field("Phase", select(phaseOpts, state.phase, ev => { state.phase = ev.target.value; render(); })),
      field("Model", select([["", "any"]].concat(models.map(m => [m, m])), state.model, ev => { state.model = ev.target.value; render(); })),
      field("Metric", select([["", "any"]].concat(metrics.map(m => [m, m])), state.metric, ev => { state.metric = ev.target.value; render(); })));
    const toggles = h("div", { class: "flex flex-wrap gap-x-5 gap-y-2" },
      provenances.map(p => checkbox(p + " [" + (TAG[p] || "?") + "]", true, ev => { state.show[p] = ev.target.checked; render(); })),
      checkbox("only entries with an action", false, ev => { state.onlyRec = ev.target.checked; render(); }),
      checkbox("apply proposal limit (" + DEFAULT_PROPOSAL_FACTS_LIMIT + ", " + DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS + " experiential slots)", false, ev => { state.applyLimit = ev.target.checked; render(); }));
    el.appendChild(h("div", { class: "space-y-4" },
      h("p", { class: "text-sm muted", text: "The retrieval context the loop builds at proposal time. Entries whose context lacks a key, or lists the chosen value, match. Click a line to see the stored entry." }),
      controls, toggles, countEl, out, detailHost));
    render();
  }

  // ---- loop stepper -----------------------------------------------------------
  // Chips are coloured by provenance (curated primary, experiential amber, derived sky) unless
  // a role tone is given: supported_by is success, a params mismatch is warning.
  function idChips(ids, opts) {
    const o = opts || {}; const byId = o.byId || {};
    if (!ids || !ids.length) return h("div", { class: "muted text-xs", text: "none" });
    return h("div", { class: "ids" }, ids.map(i => {
      const prov = (byId[i] || {}).provenance || (String(i).startsWith("exp-") ? "experiential" : "curated");
      const cls = o.tone ? "chip-" + o.tone : "chip-" + (TAG[prov] || "C").toLowerCase();
      return h("span", { class: "chip " + cls, text: i });
    }));
  }
  function stageCard(eyebrow, headline, bodyNodes, tone) {
    return h("div", { class: "card" + (tone ? " tone-" + tone : "") }, h("div", { class: "card-content !p-4 space-y-1.5" },
      h("div", { class: "eyebrow", text: eyebrow }),
      headline ? h("div", { class: "text-sm font-semibold break-words", text: headline }) : null,
      bodyNodes));
  }
  function callout(tone, titleText, bodyNodes) {
    return h("div", { class: "alert callout-" + tone, role: "note" },
      titleText ? h("h5", { class: "alert-title", text: titleText }) : null,
      h("div", { class: "alert-description" }, bodyNodes));
  }
  function runStepper(el, run, options, allData) {
    const opts = options || {};
    const entries = (allData && opts.entries && allData[opts.entries]) || [];
    const byId = Object.fromEntries(entries.map(e => [e.id, e]));
    const iterations = run.iterations || [];
    const last = iterations.length - 1;
    let step = Math.min(1, last);
    const host = h("div", { class: "space-y-4" });
    el.appendChild(host);

    // The outcome explanation as a toned callout: supported = success, with any parameter
    // mismatch as a warning sentence; an unsupported action = the override case, destructive-soft.
    function explain(a) {
      const cited = a.cited_entries || [], sup = a.supported_by || [], mis = a.supported_by_params_mismatch || [];
      const nodes = [];
      if (cited.length) {
        const kinds = cited.map(i => (byId[i] || {}).provenance || (String(i).startsWith("exp-") ? "experiential" : "curated"));
        nodes.push("Agent 2 cited " + cited.length + " entr" + (cited.length === 1 ? "y" : "ies") + " (" + [...new Set(kinds)].join(", ") + "). ");
      } else nodes.push("Agent 2 cited no entries. ");
      if (sup.length) {
        nodes.push(h("span", { class: "mark-pos", text: "A retrieved curated rule recommends this action" }), ", so the proposal is recorded as supported.");
        if (mis.length) {
          const rule = byId[mis[0]]; const rp = rule && rule.payload && rule.payload.recommendation && rule.payload.recommendation.params;
          nodes.push(" ", h("span", { class: "mark-warn", text: "The parameters differ from the rule's" + (rp ? " (rule: " + Object.entries(rp).map(([k, v]) => k + "=" + fmtParam(v)).join(", ") + ")" : "") }), ", which is recorded in supported_by_params_mismatch.");
        }
        return callout("success", "Supported by a curated rule", [h("p", { class: "text-sm" }, nodes)]);
      }
      nodes.push(h("span", { class: "mark-neg", text: "No retrieved rule recommends this action." }), " The loop allows it and logs an advisory override: exploration is permitted, but visible.");
      return callout("destructive", "Advisory override", [h("p", { class: "text-sm" }, nodes)]);
    }

    function render() {
      const it = iterations[step] || {}; const a = it.action || {}; const d = it.diagnosis || {};
      const weakPhase = (d.weak_segments || []).find(w => w.dimension === "phase");
      const params = Object.entries(a.params || {}).map(([k, v]) => k + "=" + fmtParam(v)).join(", ");
      host.textContent = "";
      host.appendChild(h("div", { class: "flex flex-wrap items-center gap-3" },
        h("button", { type: "button", class: "btn btn-outline btn-sm", disabled: step <= 1, onClick: () => { step = Math.max(1, step - 1); render(); } }, "Previous"),
        h("button", { type: "button", class: "btn btn-sm", disabled: step >= last, onClick: () => { step = Math.min(last, step + 1); render(); } }, "Next iteration"),
        h("span", { class: "text-sm muted" }, "run ", code(run.run_id), ", cutoff " + run.cutoff + ", iteration " + step + " of " + last + " (iteration 0 is the baseline, WIS " + fmtNum((iterations[0] || {}).wis, 2) + ")")));
      const summary = String(d.summary || "");
      host.appendChild(h("div", { class: "grid gap-3 md:grid-cols-2" },
        stageCard("1. Diagnosis (agent 1)", d.suggested_focus || "", [
          h("p", { class: "text-sm", text: summary.length > 220 ? summary.slice(0, 220) + "..." : summary }),
          h("p", { class: "text-xs muted" }, "weak phase named: ", h("b", { text: weakPhase ? weakPhase.value : "none" }))]),
        stageCard("2. Retrieval context", "phase = " + (weakPhase ? weakPhase.value : "any") + ", model = xgboost_direct, metric = wis", [
          h("p", { class: "text-sm", text: (a.retrieved_entry_ids || []).length + " entries shown to agent 2, each with its id" }),
          h("p", { class: "text-xs muted", text: "curated first, then experiential; up to " + DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS + " experiential slots are reserved when the limit truncates" })]),
        stageCard("3. Proposal (agent 2)", (a.name || "") + "(" + params + ")", [h("p", { class: "text-sm", text: a.rationale || "" })]),
        stageCard("4. Citations recorded", null, [
          h("div", { class: "text-xs muted", text: "cited_entries (what it says it relied on; curated in primary, experiential in amber)" }), idChips(a.cited_entries, { byId }),
          h("div", { class: "text-xs muted mt-2", text: "supported_by (shown entries recommending this action)" }), idChips(a.supported_by, { byId, tone: "success" }),
          h("div", { class: "text-xs muted mt-2", text: "supported_by_params_mismatch" }), idChips(a.supported_by_params_mismatch, { byId, tone: "warning" })]),
        stageCard("5. Outcome", "WIS " + fmtNum(it.wis, 2) + " (unchanged: fake pipeline)", [explain(a)], (a.supported_by || []).length ? "success" : "destructive")));
      host.appendChild(h("details", { class: "text-sm" }, h("summary", { class: "cursor-pointer muted", text: "retrieved_entry_ids for this iteration" }), idChips(a.retrieved_entry_ids, { byId })));
    }
    render();
  }

  // ---- charts -------------------------------------------------------------------
  function tooltip() { return h("div", { class: "chart-tip", role: "tooltip" }); }
  function placeTip(tip, host, target, text) {
    tip.textContent = text; tip.style.opacity = 1;
    const b = host.getBoundingClientRect(), r = target.getBoundingClientRect();
    tip.style.left = (r.left - b.left + r.width / 2) + "px"; tip.style.top = (r.top - b.top) + "px";
  }
  function lineChart(host, series, opts) {
    // right margin leaves room for the direct series labels drawn after the last point
    const W = 360, H = 230, m = { l: 44, r: 78, t: 12, b: 34 };
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.aria });
    const xs = [...new Set(series.flatMap(s => s.pts.map(p => p.x)))].sort((a, b) => a - b);
    const ys = series.flatMap(s => s.pts.map(p => p.y)).concat(opts.ref != null ? [opts.ref] : []);
    let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.15 || 1; y0 -= pad; y1 += pad;
    if (opts.includeZero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = v => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const v = y0 + (y1 - y0) * i / ticks;
      svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "hsl(var(--border))", "stroke-width": 1 }));
      svg.appendChild(svgEl("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, v.toFixed(opts.dec ?? 0)));
    }
    for (const v of xs) svg.appendChild(svgEl("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, String(v)));
    svg.appendChild(svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, opts.xlabel));
    // Reference semantics: the baseline is primary and dashed, the zero line is the foreground.
    if (opts.ref != null) {
      svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(opts.ref), y2: Y(opts.ref), stroke: "hsl(var(--primary))", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
      svg.appendChild(svgEl("text", { x: W - m.r, y: Y(opts.ref) - 5, "text-anchor": "end", class: "lbl" }, opts.refLabel));
    }
    if (opts.includeZero) svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: "hsl(var(--foreground))", "stroke-width": 1.25 }));
    const tip = tooltip();
    series.forEach(s => {
      const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p.x) + "," + Y(p.y)).join(" ");
      svg.appendChild(svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
      s.pts.forEach(p => {
        svg.appendChild(svgEl("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: s.color, stroke: "hsl(var(--card))", "stroke-width": 2 }));
        const hit = svgEl("circle", { cx: X(p.x), cy: Y(p.y), r: 12, fill: "transparent" });
        hit.addEventListener("mouseenter", () => placeTip(tip, host, hit, s.name + ": " + opts.xname + " " + p.x + " -> " + p.y.toFixed(opts.dec ?? 1)));
        hit.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
        svg.appendChild(hit);
      });
      const lastPt = s.pts[s.pts.length - 1];
      svg.appendChild(svgEl("text", { x: X(lastPt.x) + 7, y: Y(lastPt.y) + 3.5, class: "lbl" }, s.short));
    });
    host.appendChild(svg); host.appendChild(tip);
    const legend = h("div", { class: "legend" }, series.map(s => h("span", {}, h("i", { style: "background:" + s.color }), s.name)));
    if (opts.ref != null) legend.appendChild(h("span", {}, h("i", { class: "ref ref-primary" }), opts.refLabel));
    if (opts.includeZero) legend.appendChild(h("span", {}, h("i", { class: "ref", style: "background:hsl(var(--foreground))" }), "zero (unbiased)"));
    host.appendChild(legend);
  }
  function barChart(host, rows, opts) {
    const W = 360, H = 230, m = { l: 44, r: 16, t: 12, b: 34 };
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.aria });
    const y1 = Math.max(...rows.map(r => r.v)) * 1.12, Y = v => H - m.b - (v / y1) * (H - m.t - m.b); const bw = (W - m.l - m.r) / rows.length;
    for (let i = 0; i <= 4; i++) {
      const v = y1 * i / 4;
      svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "hsl(var(--border))" }));
      svg.appendChild(svgEl("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, v.toFixed(0)));
    }
    const tip = tooltip();
    rows.forEach((r, i) => {
      const x = m.l + i * bw + bw * 0.2, w = bw * 0.6, y = Y(r.v), hh = H - m.b - y;
      const rect = svgEl("rect", { x, y, width: w, height: hh, fill: r.color, rx: 4 });
      svg.appendChild(rect);
      svg.appendChild(svgEl("rect", { x, y: H - m.b - 4, width: w, height: 4, fill: r.color }));
      svg.appendChild(svgEl("text", { x: x + w / 2, y: y - 5, "text-anchor": "middle", class: "lbl" }, r.v.toFixed(1)));
      svg.appendChild(svgEl("text", { x: x + w / 2, y: H - m.b + 16, "text-anchor": "middle" }, r.short));
      rect.addEventListener("mouseenter", () => placeTip(tip, host, rect, r.label + ": " + opts.yname + " " + r.v.toFixed(2)));
      rect.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
    });
    svg.appendChild(svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, opts.xlabel));
    host.appendChild(svg); host.appendChild(tip);
  }
  function chartCard(title, sub) {
    const host = h("div", { class: "chart-host" });
    return { host, node: card(title, [host], { compact: true, description: sub }) };
  }
  function numericSuffix(configId, prefix) { return +String(configId).slice(prefix.length); }
  function sweepCharts(el, sweep) {
    const base = sweep.find(r => r.config_id === "baseline");
    if (!base) { el.appendChild(h("p", { class: "muted text-sm", text: "sweep.json has no baseline row; nothing to chart." })); return; }
    const lam = sweep.filter(r => r.arm === "lambda").map(r => ({ x: numericSuffix(r.config_id, "lambda_"), r }));
    const cal = sweep.filter(r => r.arm === "lambda_calendar").map(r => ({ x: numericSuffix(r.config_id, "lambda_calendar_"), r }));
    const win = sweep.filter(r => r.arm === "window").map(r => ({ x: numericSuffix(r.config_id, "window_"), r }));
    const grid = h("div", { class: "grid gap-4 md:grid-cols-2 xl:grid-cols-3" });
    const biasRef = base.peak.bias, wisRef = base.peak.wis;
    const c1 = chartCard("Peak bias against weight", "zero is unbiased; dashed line is the baseline, " + biasRef.toFixed(1));
    lineChart(c1.host, [
      { name: "approaching-peak label (" + DEFAULT_APPROACHING_PEAK_WEEKS + " weeks before max)", short: "approaching", color: "var(--chart-1)", pts: lam.map(p => ({ x: p.x, y: p.r.peak.bias })) },
      { name: "calendar label (Dec-Jan)", short: "calendar", color: "var(--chart-2)", pts: cal.map(p => ({ x: p.x, y: p.r.peak.bias })) },
    ].filter(s => s.pts.length), { aria: "peak bias by weight", xlabel: "weight on pre-peak rows", xname: "weight", includeZero: true, dec: 0, ref: biasRef, refLabel: "baseline " + biasRef.toFixed(1) });
    const c2 = chartCard("Peak WIS against weight", "lower is better; dashed line is the baseline, " + wisRef.toFixed(1));
    lineChart(c2.host, [
      { name: "approaching-peak label", short: "approaching", color: "var(--chart-1)", pts: lam.map(p => ({ x: p.x, y: p.r.peak.wis })) },
      { name: "calendar label", short: "calendar", color: "var(--chart-2)", pts: cal.map(p => ({ x: p.x, y: p.r.peak.wis })) },
    ].filter(s => s.pts.length), { aria: "peak WIS by weight", xlabel: "weight on pre-peak rows", xname: "weight", dec: 1, ref: wisRef, refLabel: "baseline " + wisRef.toFixed(1) });
    const c3 = chartCard("Training window", "peak WIS by rows kept; the baseline uses all history");
    barChart(c3.host, [{ label: "all history (baseline)", short: "all history", v: base.peak.wis, color: "hsl(var(--primary))" }]
      .concat(win.map(p => ({ label: p.x + " weeks", short: p.x + " weeks", v: p.r.peak.wis, color: "var(--chart-3)" }))),
      { aria: "peak WIS by training window", xlabel: "training rows kept before each cutoff", yname: "peak WIS" });
    grid.appendChild(c1.node); grid.appendChild(c2.node); grid.appendChild(c3.node);
    el.appendChild(grid);
  }

  // ---- results table --------------------------------------------------------------
  function get(o, path) { return path.split(".").reduce((x, k) => (x == null ? x : x[k]), o); }
  function sweepTable(el, sweep) {
    const base = sweep.find(r => r.config_id === "baseline");
    if (!base) { el.appendChild(h("p", { class: "muted text-sm", text: "sweep.json has no baseline row." })); return; }
    const METRICS = [
      ["peak.wis", "peak WIS"], ["overall.wis", "overall WIS"], ["peak.bias", "peak bias"], ["peak.coverage_95", "peak coverage 95"],
      ["peak.mae", "peak MAE"], ["onset.wis", "onset WIS"], ["decline.wis", "decline WIS"], ["by_horizon.4", "horizon-4 WIS"],
    ].filter(([p]) => get(base, p) != null);
    let path = METRICS[0][0];
    const fmt = (v, p) => /coverage/.test(p) ? v.toFixed(3) : /bias/.test(p) ? v.toFixed(1) : v.toFixed(2);
    // "Best" per metric: lowest WIS/MAE, bias nearest zero, coverage nearest the nominal 0.95.
    const score = (v, p) => /bias/.test(p) ? Math.abs(v) : /coverage/.test(p) ? Math.abs(v - 0.95) : v;
    const bestFor = p => { let best = null; for (const r of sweep) { const s = score(get(r, p), p); if (best == null || s < best) best = s; } return best; };
    const ALWAYS = [["peak.wis", "peak WIS"], ["overall.wis", "overall WIS"], ["peak.bias", "peak bias"], ["peak.coverage_95", "peak coverage 95"]];
    const table = h("table", { class: "table table-compact table-dense" });
    function metricCell(r, p, best) {
      const v = get(r, p); const isBest = score(v, p) === best;
      return h("td", { class: "num" + (isBest ? " is-best" : "") }, fmt(v, p), isBest ? h("span", { class: "badge badge-success badge-xs", text: "best" }) : null);
    }
    function render() {
      const bv = get(base, path); const label = METRICS.find(m => m[0] === path)[1];
      // The selected metric leads; the always-on columns skip it so nothing is shown twice.
      const fixed = ALWAYS.filter(([p]) => p !== path);
      const bestSel = bestFor(path); const bestFixed = fixed.map(([p]) => bestFor(p));
      table.textContent = "";
      table.appendChild(h("thead", {}, h("tr", {}, h("th", { text: "arm" }), h("th", { text: "config" }), h("th", { class: "num", text: label }),
        h("th", { class: "num", text: "delta vs baseline" }), fixed.map(([, l]) => h("th", { class: "num", text: l })))));
      const tbody = h("tbody");
      for (const r of sweep) {
        const v = get(r, path); const d = v - bv; const isBase = r.config_id === "baseline";
        const better = isBase ? null : (/bias/.test(path) ? Math.abs(v) < Math.abs(bv) : /coverage/.test(path) ? Math.abs(v - 0.95) < Math.abs(bv - 0.95) : d < 0);
        // colour = better/worse than the baseline, arrow = direction of the change
        const deltaClass = isBase ? "" : (d === 0 ? "delta-flat" : (better ? "delta-pos" : "delta-neg") + (d > 0 ? " delta-up" : " delta-down"));
        tbody.appendChild(h("tr", { class: isBase ? "row-highlight" : "" },
          h("td", { text: r.arm }), h("td", {}, code(r.config_id)),
          metricCell(r, path, bestSel),
          h("td", { class: "num" }, isBase ? h("span", { class: "muted", text: "reference" }) : h("span", { class: deltaClass, text: (d >= 0 ? "+" : "") + fmt(d, path) })),
          fixed.map(([p], i) => metricCell(r, p, bestFixed[i]))));
      }
      table.appendChild(tbody);
    }
    el.appendChild(h("div", { class: "space-y-3" },
      h("div", { class: "max-w-xs" }, field("Table metric", select(METRICS, path, ev => { path = ev.target.value; render(); }))),
      h("div", { class: "table-wrap rounded-md border" }, table),
      h("p", { class: "text-xs muted" }, h("span", { class: "delta-pos", text: "green" }), " deltas improve on the baseline for the selected metric (lower WIS and MAE, bias nearer zero, coverage nearer 0.95), ",
        h("span", { class: "delta-neg", text: "red" }), " deltas do not; the triangle is the direction of the change. The highlighted row is the baseline; ",
        h("span", { class: "badge badge-success badge-xs !ml-0", text: "best" }), " marks the best value in each metric column.")));
    render();
  }

  // ---- architecture diagram ---------------------------------------------------------
  const ARCH_NODES = [
    { id: "data", x: 16, y: 24, label: "CDC FluSight data", sub: "data/raw/", path: "src/data_loader.py",
      text: "Weekly influenza hospitalization counts fetched from the CDC FluSight GitHub repository and cached locally for a week. Everything downstream, including the experiment and the bank's experiential entries, is scored against this series." },
    { id: "pipeline", x: 16, y: 148, label: "Forecast pipeline", sub: "src/pipeline.py", path: "src/pipeline.py, src/direct_forecast.py",
      text: "Feature engineering plus the XGBoost direct ensemble (or any registered model family). It takes a config dict and returns a forecast CSV; the two new knobs (approaching-peak sample weights, post-feature training window) live here and are what the seventh action set_training_window drives." },
    { id: "curated", x: 308, y: 24, label: "Curated YAML", sub: "knowledge/curated/*.yaml", path: "knowledge/curated/, agent/knowledge/curated.py",
      text: "Human-authored guardrails: the five rectification actions, training-strategy rules and the migrated domain context (18 entries). Each file is validated against the schema; a pull request is the review step. The store rebuilds these rows from YAML every time it opens, so the YAML is the source of truth." },
    { id: "db", x: 308, y: 148, label: "knowledge.db", sub: "agent/knowledge/store.py", path: "knowledge/knowledge.db, agent/knowledge/store.py",
      text: "SQLite store with one row per entry. Curated rows are replaced on every open; experiential rows persist. Retrieval is a context match (an entry matches when its context lacks the key or lists the value) ordered by provenance trust, confidence, n_observations and id, capped at 30." },
    { id: "explog", x: 600, y: 24, label: "Experiment log", sub: "outputs/experiments/.../log.jsonl", path: "scripts/experiments/peak_rectification.py, agent/knowledge/experiment_import.py",
      text: "One JSON line per configuration from the controlled sweep (10 configs x 9 cutoffs). `knowledge import-experiment` turns each outcome into an experiential entry: statement, reward delta against the baseline, bias and coverage before and after, confidence low with one observation." },
    { id: "loop", x: 308, y: 272, label: "Improve loop", sub: "agent/orchestrator.py", path: "agent/orchestrator.py, agent/prompt_templates.py",
      text: "Evaluate, diagnose (agent 1), propose (agent 2), validate, apply, retrain. Before every proposal the loop queries the bank with the diagnosed phase and renders a KNOWN FACTS block with entry ids. The proposal must cite what it relied on; the loop records cited_entries, supported_by and a params mismatch, and logs an advisory override when no retrieved rule backs the action." },
    { id: "runs", x: 600, y: 272, label: "runs.db + report", sub: "outputs/agent_runs/", path: "agent/run_tracker.py, agent/run_report.py",
      text: "Every iteration, with its action, metrics, citations and retrieved ids, goes into SQLite. The end-of-run report (report.md and report.json) prints a cites: line per iteration so a reviewer can trace each move back to a bank entry." },
  ];
  const NODE_W = 184, NODE_H = 56;
  const ARCH_EDGES = [
    { from: "data", to: "pipeline", label: "" },
    { from: "curated", to: "db", label: "rebuild on open" },
    { from: "explog", to: "db", label: "import-experiment" },
    { from: "db", to: "loop", label: "KNOWN FACTS block" },
    { from: "pipeline", to: "loop", label: "retrain, re-evaluate", both: true },
    { from: "loop", to: "runs", label: "cited_entries" },
  ];
  function nodeCenter(n) { return { cx: n.x + NODE_W / 2, cy: n.y + NODE_H / 2 }; }
  function edgePath(a, b) {
    // Straight connectors that leave from the nearest side of a box.
    const A = nodeCenter(a), B = nodeCenter(b);
    const dx = B.cx - A.cx, dy = B.cy - A.cy;
    if (Math.abs(dx) < 1) return { x1: A.cx, y1: dy > 0 ? a.y + NODE_H : a.y, x2: B.cx, y2: dy > 0 ? b.y : b.y + NODE_H };
    if (Math.abs(dy) < 1) return { x1: dx > 0 ? a.x + NODE_W : a.x, y1: A.cy, x2: dx > 0 ? b.x : b.x + NODE_W, y2: B.cy };
    // diagonal: leave horizontally, arrive on the top/bottom or side depending on direction
    return { x1: dx > 0 ? a.x + NODE_W : a.x, y1: A.cy, x2: dx > 0 ? b.x : b.x + NODE_W, y2: B.cy };
  }
  function architectureDiagram(el) {
    const W = 800, H = 352;
    const byId = Object.fromEntries(ARCH_NODES.map(n => [n.id, n]));
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "How the knowledge bank sits between the data, the pipeline and the improve loop" });
    const defs = svgEl("defs");
    const marker = svgEl("marker", { id: "arch-arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
    marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "arrow" }));
    defs.appendChild(marker); svg.appendChild(defs);
    for (const e of ARCH_EDGES) {
      const p = edgePath(byId[e.from], byId[e.to]);
      const attrs = { x1: p.x1, y1: p.y1, x2: p.x2, y2: p.y2, class: "edge", "marker-end": "url(#arch-arrow)" };
      if (e.both) attrs["marker-start"] = "url(#arch-arrow)";
      svg.appendChild(svgEl("line", attrs));
      if (e.label) {
        const mx = (p.x1 + p.x2) / 2, my = (p.y1 + p.y2) / 2;
        const vertical = Math.abs(p.x1 - p.x2) < 1;
        svg.appendChild(svgEl("text", { x: vertical ? mx + 8 : mx, y: vertical ? my + 3 : my - 7, "text-anchor": vertical ? "start" : "middle", class: "edge-label" }, e.label));
      }
    }
    const detail = h("div", { class: "card" });
    const groups = {};
    function show(id) {
      const n = byId[id];
      for (const [k, g] of Object.entries(groups)) g.classList.toggle("is-active", k === id);
      detail.textContent = "";
      detail.appendChild(h("div", { class: "card-header !p-4 !pb-2" }, h("h4", { class: "card-title-sm", text: n.label }), h("p", { class: "card-description" }, code(n.path))));
      detail.appendChild(h("div", { class: "card-content !p-4 !pt-0" }, h("p", { class: "text-sm", text: n.text })));
    }
    for (const n of ARCH_NODES) {
      const g = svgEl("g", { class: "node", role: "button", tabindex: "0", "aria-label": n.label });
      g.appendChild(svgEl("rect", { x: n.x, y: n.y, width: NODE_W, height: NODE_H, rx: 8 }));
      g.appendChild(svgEl("text", { x: n.x + 14, y: n.y + 24 }, n.label));
      g.appendChild(svgEl("text", { x: n.x + 14, y: n.y + 42, class: "sub" }, n.sub));
      g.addEventListener("click", () => show(n.id));
      g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); show(n.id); } });
      groups[n.id] = g; svg.appendChild(g);
    }
    el.appendChild(h("div", { class: "space-y-4 diagram-host" },
      h("p", { class: "text-sm muted", text: "Click a box for what it is and where it lives." }),
      h("div", { class: "overflow-x-auto" }, h("div", { style: "min-width: 560px" }, svg)),
      detail));
    show("db");
  }

  // ---- bank lifecycle -----------------------------------------------------------
  const LIFECYCLE = [
    { id: "add", label: "Add", title: "Add an entry",
      text: "Curated knowledge is authored by hand in knowledge/curated/*.yaml, validated against the schema and reviewed in a pull request. Experiential knowledge is not hand-written: `knowledge import-experiment` reads an experiment log and writes one entry per outcome.",
      command: "python -m agent knowledge validate\npython -m agent knowledge import-experiment --log outputs/experiments/peak_rectification/log.jsonl" },
    { id: "rebuild", label: "Rebuild on open", title: "Rebuild on open",
      text: "Whenever the store opens, every curated row is deleted and re-read from YAML, so the YAML files are the source of truth and a stale database cannot disagree with them. Experiential rows are left alone and survive the rebuild.",
      command: "python -m agent knowledge rebuild" },
    { id: "retrieve", label: "Retrieve", title: "Retrieve for a context",
      text: "A query names phase, model, metric (and optionally season_week). An entry matches when its context lacks the key or lists the value. Matches are ordered by provenance trust, confidence, n_observations and id, and capped at 30. When the cap truncates, up to 6 slots are reserved for experiential entries so the loop's own history is never crowded out by curated rules.",
      command: "python -m agent knowledge query --phase peak --model xgboost_direct" },
    { id: "cite", label: "Cite", title: "Cite in the loop",
      text: "The proposal prompt shows the retrieved entries with their ids as a KNOWN FACTS block. Agent 2 must list the ids it relied on; the loop records cited_entries, computes supported_by (retrieved rules recommending the chosen action) and supported_by_params_mismatch, and logs an advisory override when nothing supports the action. The run report prints a cites: line per iteration.",
      command: "python -m agent improve --cutoff-date 2025-12-06 --auto-apply\npython -m agent report <run_id>" },
    { id: "remove", label: "Remove", title: "Remove an entry",
      text: "Curated: delete it from the YAML file and rebuild (or just reopen the store). Experiential: `knowledge remove --id` deletes the row; `--dry-run` shows what would go first. Removal is deliberate and leaves no tombstone, so the git history of the YAML and the experiment log remain the audit trail.",
      command: "python -m agent knowledge remove --id exp-window-12-2025 --dry-run\npython -m agent knowledge remove --id exp-window-12-2025" },
  ];
  function bankLifecycle(el) {
    const buttons = {};
    const panel = h("div", { class: "card" });
    function show(id) {
      const s = LIFECYCLE.find(x => x.id === id);
      for (const [k, b] of Object.entries(buttons)) b.setAttribute("aria-pressed", k === id ? "true" : "false");
      panel.textContent = "";
      panel.appendChild(h("div", { class: "card-header !p-4 !pb-2" }, h("h4", { class: "card-title-sm", text: s.title })));
      panel.appendChild(h("div", { class: "card-content !p-4 !pt-0 space-y-3" }, h("p", { class: "text-sm", text: s.text }), pre(s.command)));
    }
    const flow = h("div", { class: "flow" });
    LIFECYCLE.forEach((s, i) => {
      if (i) flow.appendChild(h("span", { class: "flow-arrow", "aria-hidden": "true", text: "->" }));
      // Remove is the one destructive stage; the active stage otherwise takes primary via aria-pressed.
      const b = h("button", { type: "button", class: "btn btn-sm " + (s.id === "remove" ? "btn-destructive-soft" : "btn-outline"), "aria-pressed": "false", onClick: () => show(s.id) }, s.label);
      buttons[s.id] = b; flow.appendChild(b);
    });
    el.appendChild(h("div", { class: "space-y-4" }, h("p", { class: "text-sm muted", text: "The five things that can happen to an entry. Click a stage for the rule and the command." }), flow, panel));
    show("add");
  }

  // ---- schema reference -----------------------------------------------------------
  const SCHEMA_FIELDS = [
    ["id", "slug", "Unique, stable, lower-case with dashes. Experiential ids are generated from the experiment config and season (exp-lambda-3-2025)."],
    ["provenance", "curated | derived | experiential", "Who produced it: a person, a scheduled derivation job, or the loop/experiment harness. Drives the trust rank 3 / 2 / 1."],
    ["category", "model_characteristics | input_data | forecasts | domain_dynamics", "What the statement is about."],
    ["statement", "single line", "The fact as the LLM reads it. One sentence, no newline."],
    ["entities", "keys: model, phase, data_source, season, metric", "What the statement is about, as single values. Descriptive, not used for matching."],
    ["context", "keys: phase, model, metric, season_week", "When the entry applies. phase/model/metric are lists; season_week is a [start, end] range. An absent key means 'always'. Phases: " + KNOWN_PHASES.join(", ") + "."],
    ["payload", "free-form object", "Structured detail. Convention for actionable entries: recommendation = {action, params, note} where action is an adapter action or not_yet_available. Experiential entries carry state, action, reward, bias/coverage before and after, and a guard metric."],
    ["evidence", "{source, n_observations, reward_delta}", "Where it came from (person and date, or log path and run_at), how many observations back it, and the measured delta on the reward metric when there is one."],
    ["confidence", "low | medium | high", "Second sort key after provenance. Imported experiment outcomes start low with one observation."],
    ["created_at", "YYYY-MM-DD", "When the entry was written. Not a sort key."],
  ];
  const SCHEMA_CONSTANTS = [
    ["DEFAULT_QUERY_LIMIT", String(DEFAULT_QUERY_LIMIT), "agent/knowledge/store.py", "most entries one query returns"],
    ["DEFAULT_PROPOSAL_FACTS_LIMIT", String(DEFAULT_PROPOSAL_FACTS_LIMIT), "agent/orchestrator.py", "entries shown per proposal; pinned to the query limit"],
    ["DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS", String(DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS), "agent/orchestrator.py", "experiential entries always shown when the limit truncates"],
    ["MIN_SEASON_WEEKS", String(MIN_SEASON_WEEKS), "src/direct_forecast.py", "a season needs this many weeks before its peak counts for approaching-peak weights"],
    ["DEFAULT_APPROACHING_PEAK_WEEKS", String(DEFAULT_APPROACHING_PEAK_WEEKS), "src/direct_forecast.py", "K: weeks before each eligible season's peak labelled approaching-peak"],
  ];
  // "a | b | c" in the type column is an enumeration: render each value as an outline badge.
  function enumOrText(cell) {
    if (typeof cell !== "string" || !/ \| /.test(cell)) return cell;
    return h("span", { class: "flex flex-wrap gap-1" }, cell.split(" | ").map(v => h("span", { class: "badge badge-outline mono", text: v })));
  }
  function simpleTable(headers, rows) {
    return h("div", { class: "table-wrap rounded-md border" }, h("table", { class: "table table-compact" },
      h("thead", {}, h("tr", {}, headers.map(t => h("th", { text: t })))),
      h("tbody", {}, rows.map(r => h("tr", {}, r.map((cell, i) => h("td", {}, i === 0 ? code(cell) : i === 1 ? enumOrText(cell) : cell)))))));
  }
  const PROVENANCE_BADGE = { curated: "badge", derived: "badge badge-info", experiential: "badge badge-warning" };
  function schemaReference(el, entries, options) {
    const opts = options || {};
    const exampleIds = opts.examples || ["rectify-peak-loss-weight-approaching-peak", "exp-lambda-3-2025"];
    const byId = Object.fromEntries((entries || []).map(e => [e.id, e]));
    const examples = exampleIds.map(id => {
      const e = byId[id];
      if (!e) return card(id, [h("p", { class: "text-sm muted", text: "not present in the embedded data" })], { compact: true });
      const header = h("div", { class: "card-header !p-4 !pb-2" },
        h("div", { class: "flex flex-wrap items-center gap-2" }, h("h4", { class: "card-title-sm break-all", text: e.id }),
          h("span", { class: PROVENANCE_BADGE[e.provenance] || "badge badge-secondary", text: e.provenance }),
          h("span", { class: "badge badge-outline", text: e.confidence })),
        h("p", { class: "card-description", text: e.statement }));
      return h("div", { class: "card" }, header, h("div", { class: "card-content !p-4 !pt-0" }, pre(JSON.stringify(e, null, 2))));
    });
    el.appendChild(h("div", { class: "space-y-6" },
      h("div", { class: "space-y-2" }, h("h4", { class: "text-sm font-semibold", text: "Fields" }), simpleTable(["field", "type or values", "notes"], SCHEMA_FIELDS)),
      h("div", { class: "space-y-2" }, h("h4", { class: "text-sm font-semibold", text: "Ordering rule" }),
        h("p", { class: "text-sm", text: "Matches are sorted by provenance trust (curated 3, derived 2, experiential 1) descending, then confidence (high 3, medium 2, low 1) descending, then evidence.n_observations descending, then id ascending. The order is deterministic, so two runs with the same bank see the same KNOWN FACTS block." })),
      h("div", { class: "space-y-2" }, h("h4", { class: "text-sm font-semibold", text: "Constants" }), simpleTable(["constant", "value", "where", "meaning"], SCHEMA_CONSTANTS)),
      h("div", { class: "space-y-2" }, h("h4", { class: "text-sm font-semibold", text: "Two entries in full" }),
        h("p", { class: "text-sm muted", text: "A curated rule and the experiential entry the experiment produced for the same action, as stored." }),
        h("div", { class: "grid gap-4 xl:grid-cols-2" }, examples))));
  }

  // ---- tabs -----------------------------------------------------------------------
  function initTabs(root) {
    root.querySelectorAll("[data-tabs]").forEach(tabs => {
      const triggers = [...tabs.querySelectorAll(":scope > .tabs-list > .tabs-trigger")];
      const panels = [...tabs.querySelectorAll(":scope > .tabs-content")];
      function activate(name, focus) {
        triggers.forEach(t => { const on = t.dataset.tab === name; t.dataset.state = on ? "active" : "inactive"; t.setAttribute("aria-selected", on ? "true" : "false"); t.tabIndex = on ? 0 : -1; if (on && focus) t.focus(); });
        panels.forEach(p => { const on = p.dataset.tab === name; p.dataset.state = on ? "active" : "inactive"; p.hidden = !on; });
      }
      triggers.forEach((t, i) => {
        t.addEventListener("click", () => activate(t.dataset.tab, false));
        t.addEventListener("keydown", ev => {
          if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
          ev.preventDefault();
          const j = (i + (ev.key === "ArrowRight" ? 1 : triggers.length - 1)) % triggers.length;
          activate(triggers[j].dataset.tab, true);
        });
      });
      const initial = triggers.find(t => t.dataset.state === "active") || triggers[0];
      if (initial) activate(initial.dataset.tab, false);
    });
  }

  // ---- mount ------------------------------------------------------------------------
  const WIDGETS = { bankExplorer, runStepper, sweepCharts, sweepTable, bankLifecycle, schemaReference, architectureDiagram };
  function init(data) {
    document.querySelectorAll("[data-widget]").forEach(el => {
      const name = el.dataset.widget; const fn = WIDGETS[name];
      let options = {};
      try { options = el.dataset.options ? JSON.parse(el.dataset.options) : {}; } catch (e) { options = {}; }
      if (!fn) { el.appendChild(h("div", { class: "alert" }, h("div", { class: "alert-title", text: "Unknown widget" }), h("div", { class: "alert-description", text: name }))); return; }
      try {
        fn(el, el.dataset.source ? data[el.dataset.source] : undefined, options, data);
      } catch (err) {
        el.appendChild(h("div", { class: "alert" }, h("div", { class: "alert-title", text: "Widget failed: " + name }), h("div", { class: "alert-description", text: String(err && err.message || err) })));
      }
    });
    initTabs(document);
  }

  global.ReportWidgets = Object.assign({ init, initTabs, matches, orderKey, lineFor, applyProposalLimit }, WIDGETS);
})(window);
