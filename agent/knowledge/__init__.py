"""Knowledge bank v1: schema, SQLite store, curated YAML intake, prompt rendering.

Implements `documentation/knowledge-bank-design.md` sections 3, 4 and 6 (the
curated write path and the KNOWN FACTS block). The derived refresh jobs, the
end-of-run bank-writer (section 5) and the loop injection point are not part
of v1 and do not live here yet.

Reading order for a newcomer: `knowledge/README.md` (what the bank is and how
to author an entry), then `schema.py` (what an entry is), `curated.py` (how
YAML becomes entries), `store.py` (how they are stored and queried), and
`render.py` (how they reach the LLM).

Typical use:

    from agent.knowledge import KnowledgeBank, RetrievalContext, render_known_facts

    bank = KnowledgeBank.open()                       # rebuilds curated rows from YAML
    facts = bank.query(RetrievalContext(phase="peak", model="xgboost_direct"))
    prompt_block = render_known_facts(facts)
"""

from __future__ import annotations

from agent.knowledge.curated import (
    DEFAULT_CURATED_DIR,
    REPO_ROOT,
    list_curated_files,
    load_curated_dir,
)
from agent.knowledge.render import (
    DEFAULT_HEADER,
    EMPTY_BLOCK,
    render_entry_line,
    render_known_facts,
)
from agent.knowledge.schema import (
    CATEGORIES,
    CONFIDENCES,
    CONTEXT_KEYS,
    ENTITY_KEYS,
    KNOWN_PHASES,
    PROVENANCES,
    RECOMMENDATION_NOT_YET_AVAILABLE,
    KnowledgeEntry,
    RetrievalContext,
)
from agent.knowledge.store import (
    CONFIDENCE_RANK,
    DEFAULT_DB_PATH,
    DEFAULT_QUERY_LIMIT,
    TRUST_RANK,
    KnowledgeBank,
)

__all__ = [
    "CATEGORIES",
    "CONFIDENCES",
    "CONFIDENCE_RANK",
    "CONTEXT_KEYS",
    "DEFAULT_CURATED_DIR",
    "DEFAULT_DB_PATH",
    "DEFAULT_HEADER",
    "DEFAULT_QUERY_LIMIT",
    "EMPTY_BLOCK",
    "ENTITY_KEYS",
    "KNOWN_PHASES",
    "PROVENANCES",
    "RECOMMENDATION_NOT_YET_AVAILABLE",
    "REPO_ROOT",
    "TRUST_RANK",
    "KnowledgeBank",
    "KnowledgeEntry",
    "RetrievalContext",
    "list_curated_files",
    "load_curated_dir",
    "render_entry_line",
    "render_known_facts",
]
