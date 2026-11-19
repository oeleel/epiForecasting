# RepairAgent - paper summary and what the project can reuse

> Bouzenia, Devanbu, Pradel. *RepairAgent: An Autonomous, LLM-Based Agent for Program Repair.* arXiv 2403.17134v2,
> 28 Oct 2024 (v1: 25 Mar 2024). University of Stuttgart / UC Davis. ICSE 2025 (venue per the task brief; the arXiv
> PDF does not name it). https://arxiv.org/abs/2403.17134
>
> Read in full 2026-11-19 from the v2 PDF (14 pages: 11 main, 3 references). Page numbers are PDF pages. Statements
> marked **(inference)** are mine, not the paper's. The tool-to-state mapping is **(read from figure)**. The paper
> prints no prompt text, so the wording of the prompt sections is unknown. I did not read the released code.

## 1. Summary

RepairAgent fixes Java bugs with one LLM agent (GPT-3.5-0125, built on AutoGPT; §IV, p7). Earlier LLM repair tools
use a hard-coded feedback loop over a fixed code context; RepairAgent lets the model interleave gathering
information, searching for repair ingredients and validating fixes (§I, p1-2). Each cycle is one LLM query that
returns one tool call; the run ends when the agent declares success or after 40 cycles (§III-B, §III-C3, p3). Three
components make this work: a prompt of eight sections, some static and some rebuilt every cycle (Table I, p3); 14
tools (Table II, p6); and a middleware that offers only the tools of the current state of a four-state machine,
repairs malformed output, rejects repeated commands and stores tool output (§III-E, p6-7). On all 835 Defects4J bugs
it fixes 164 correctly, against 162 for ChatRepair, at about 270,000 tokens or 14 cents per bug (p1; Table III, p8).

## 2. System design

**States and tools (§III-C4, Fig 2, p4; Table II, p6).** The stated motivation: without guidance the authors
"observed the LLM agent to frequently get lost in aimless exploration" (p4). The machine "constrains which tools are
available at a given point in time" but "does not enforce a strict order of tool invocations", because the agent
changes state by calling control tools (p4).

| State | Tools offered **(read from figure)** | How the agent leaves |
|---|---|---|
| Understand the bug (start) | `run_tests`, `extract_tests`, `run_fault_localization` | `express_hypothesis` -> Collect (automatic) |
| Collect information to fix the bug | `search_code_base`, `find_similar_api_calls`, `generate_method_body`, `read_range`, `get_classes_and_methods`, `extract_method` | `write_fix` -> Try to fix; `discard_hypothesis` -> Understand |
| Try to fix the bug | `write_fix`, `read_range` | `collect_more_information` -> Collect; `discard_hypothesis` -> Understand; `goal_accomplished` -> Done |
| Done | none | terminal |

- The hypothesis is explicit state: `express_hypothesis` and `discard_hypothesis` "enforce a structured approach to
  hypothesis formulation" (§III-D4, p6). `goal_accomplished` is allowed once one fix passes all tests (p6).
- `write_fix` takes a patch in a fixed JSON shape (per file: `insertions`, `deletions`, `modifications`, by line
  number; Fig 5, p6), applies it, runs the tests and **reverts automatically when tests fail**. It also samples up
  to 30 variants of the fix, removes duplicates and tests each (§III-D3, p6).
- Tools shape their output for the prompt: `run_tests` removes stack-trace entries outside the project (p5).

**Prompt sections (Table I, p3; §III-C, p3-5).** The prompt is rebuilt every cycle from these sections (Definition
2, p3); it is not a growing chat transcript **(inference from Definition 2 and §III-E3)**.

| # | Section | Nature | Content |
|---|---|---|---|
| 1 | Role | static | Java bug-fixing expert; decisions are autonomous, no user help |
| 2 | Goals | static | Five goals: locate, gather information, simple fixes, complex fixes, iterate |
| 3 | Guidelines | static | Recurring fix patterns with examples; comment the changed code; end the reasoning with one next step that maps to a tool call; the cycle budget |
| 4 | State description | dynamic | The current state of the machine |
| 5 | Available tools | dynamic | Only the tools of the current state: name, description, typed arguments |
| 6 | Gathered information | dynamic | The memory (below) |
| 7 | Output format | static | The JSON response format (below) |
| 8 | Last executed command and result | dynamic | Last tool name, arguments and output, including any problem; cycles used and cycles left |

**Response format (Fig 3-4, p4-5).** One JSON object per cycle with two fields: `thoughts` (string) and `command`
(`name`: string, `args`: map). `thoughts` is free text, kept for transparency, debugging and better reasoning (p4).

**Middleware and guards (§III-E, p6-7).**
- Repair of malformed output, in three steps. (1) Tool name: accept an available tool when one name is a substring
  of the other, else when the Levenshtein distance is below a threshold (0.1 by default; a normalized distance,
  **(inference)**). (2) Argument names by the same logic, for example `path` to `file_path`. (3) Invalid argument
  values are mapped or replaced, for example an invalid file path.
- When repair fails or gives several candidates, the middleware does not guess. It "informs the LLM about the issue
  via the 'Last executed command and result' prompt section and enters a new cycle".
- Repeated actions: if the command equals a command of a previous cycle (same tool, same arguments), the middleware
  "informs the agent about the repetition and enters a new cycle". Both cases spend one cycle of the budget
  **(inference from "enters a new cycle")**.
- Commands run in Docker (§IV, p7). After each call the middleware updates sections 4, 5, 6 and 8 (§III-E3, p7).

**Memory (§III-C6, p4).** Gathered information is "structured into different subsections, where each subsection
contains the outputs produced by a specific tool". It accumulates over all cycles. Fig 1 (p3) says "store and
summarize tool output", but the text describes no summarization step and no cap.

## 3. Evaluation and results

**Setup (§V-A, p7).** Defects4J, all 835 bugs from 17 Java projects (395 in v1.2, 440 added in v2). GitBug-Java: 100
sampled of 199 bugs, all from 2023, after the model cut-off. Baselines: ChatRepair, ITER, SelfAPR, through the
patches their authors published. Plausible = passes all tests. Correct = matches the developer fix exactly, or the
authors judge it semantically consistent by hand. Perfect fault localization is the default.

**Effectiveness (§V-B, Tables III-V, p8-9).**
- Defects4J: 186 plausible, **164 correct** (116 exact, 48 by manual judgement). ChatRepair 162, SelfAPR 110, ITER
  57. On v2: 90 against 48 for ChatRepair. On v1.2 it is behind: 74 against 114.
- 39 bugs are fixed by no baseline. Against ChatRepair: fewer single-line fixes (115 to 133), more multi-line (46 to 29).
- GitBug-Java: 19 plausible, 13 correct of 100 (9 of 19 single-line, 4 of 64 multi-line, 0 of 17 multi-file).

**Cost (§V-C, §V-E, p9-10).** Median 920 seconds per bug; 99% of the time is tool execution, mostly tests. Median
about 270,000 tokens, about 14 cents at March 2024 prices (the abstract calls this an "average"). Fixed bugs use
about 21,000 tokens, unfixed bugs about 315,000. Average 35 tool calls per bug; `write_fix` is the most used.

**Ablation (§V-D, Table VI, p9-10).** Same 100 random Defects4J bugs for every row.

| Configuration | Plausible | Correct | Cost ($) |
|---|---|---|---|
| RepairAgent (default) | 23 | 21 | 16 |
| No state machine | 18 | 14 | 31 |
| No search tools | 14 | 11 | 28 |
| Single-cycle memory | 9 | 6 | 8 |
| Realistic fault localization (GZoltar) | 16 | 16 | 29 |

Without the state machine the agent "directly starts with suggesting a fix (often wrong) without collecting any
information". With single-cycle memory it "repeats the same commands after a few cycles" and uses wrong file and
function names (p9-10).

## 4. Limits the authors state (§VI, p10-11)

- Data leakage: GPT-3.5 may have seen the Defects4J projects. GitBug-Java is the stated mitigation.
- Every bug comes with a failing test; real use may have none. Two runs can differ (non-deterministic LLM).
- Realistic fault localization costs 25% of the fixes and 81% more money (§V-D, p10).
- The agent sometimes proposes complex fixes for simple bugs, and fixes only part of a multi-location bug (§VI-A).

My reading **(inference)**: "generalizes well" rests on 13 of 100 on GitBug-Java against about 20% on Defects4J. The
ablation is 100 bugs with no repeated runs reported, so a gap of a few bugs is weak evidence. The paper does not
report how often the output repair or the repetition guard fired.

## 5. What we can reuse

1. **Restrict actions by state, in code.** `agent-restructure-note.md` (the note) proposes five roles. The paper
   gets its structure from one agent plus a state machine, and the ablation credits the machine with 21 against 14
   fixes at half the cost. For us: states such as `diagnose`, `propose`, `review`, each with its own slice of the
   action catalog. This fits the note's rule "do not add agents that only relay text". The mapping is **(inference)**.
2. **Give the hypothesis a lifecycle.** `express_hypothesis` and `discard_hypothesis` gate the state changes.
   `DIAGNOSIS_SCHEMA` has `hypotheses`, but Agent 1 writes a new diagnosis every iteration and nothing records that
   a hypothesis was tested and rejected. Add a hypothesis id and status to the note's `Handoff` envelope, so the
   report states which hypothesis each action tested.
3. **Add a code-side repetition guard.** Today the proposal prompt only asks Agent 2 not to repeat itself, and the
   orchestrator catches a repeat only when the config does not change (`no_op`). An action that was applied and
   reverted can be proposed again and costs a full retrain. Compare `(name, params)` with the iteration history
   before `apply_action` and return the repetition as feedback.
4. **Repair small format errors, never choose silently.** The three-step mapping accepts a near miss and sends
   everything ambiguous back to the LLM. This sits between TSci's silent default (the note's criticism) and our
   single retry with the validation error. Use it for action names and parameter keys from the 8B model and log
   every repair in the hand-off `status`. Do not repair numeric values: for us a value changes a model **(inference)**.
5. **Build prompts from named static and dynamic sections.** Role, goals and guidelines are static; diagnosis,
   catalog, KNOWN FACTS and history are dynamic. The Agent 2 prompt lacks two of their sections: a "last command and
   result" section (our history shows action and metric delta, not the status or the reason for a revert; in the
   restructure this is the reviewers' verdict), and the remaining iteration budget.

## 6. What does not transfer

- **The oracle.** A test suite gives a binary result in seconds. Our signal is a noisy change in WIS after a
  retrain, which is why the note proposes a significance reviewer.
- **Sampling 30 variants per fix and a 40-cycle budget.** Test runs are cheap. Each of our iterations is a retrain.
- **Unbounded memory.** It saturates their prompt (§V-C, p9). A local 8B model needs a cap per section.
- **Read and search tools.** They exist because a code base does not fit in a prompt. Our evaluator already gives
  structured metrics; the knowledge-bank retrieval is our only search step **(inference)**.
- **A single agent.** The paper has no reviewer, so it gives no evidence for or against parallel reviewers with weights.
