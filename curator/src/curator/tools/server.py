"""The MCP server: the only way any agent touches the outside world.

Why MCP and not plain Python functions? Three reasons that are worth saying out loud
in an interview:

1. The tool surface is a *contract* with a JSON schema, discovered at runtime. The
   agent code never imports the retriever; it lists tools and calls them by name.
2. The same server is reusable outside this repo — point Claude Desktop or any MCP
   client at `python -m curator.tools.server` and the corpus is browsable by hand.
3. It forces a real failure boundary. Tool errors arrive as data, get counted in the
   trace, and are handled by the graph rather than raising through the agent.

Run standalone:  python -m curator.tools.server            # stdio
                 python -m curator.tools.server --http     # streamable-http
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import load_config
from ..retrieval.store import Filters, HybridIndex, load_corpus

server = MCPServer(
    name="curator-tools",
    instructions=(
        "Search and inspect a corpus of research papers, and read/write a user's "
        "taste profile. Always cite paper_id values returned by these tools."
    ),
)

_CFG = load_config(os.getenv("CURATOR_CONFIG"))
_INDEX: HybridIndex | None = None


def get_index() -> HybridIndex:
    """Built once per process. Tests inject their own via `set_index`."""
    global _INDEX
    if _INDEX is None:
        index_dir = Path(_CFG.retrieval.index_dir)
        if (index_dir / "papers.jsonl").exists():
            _INDEX = HybridIndex.load(_CFG.retrieval)
        else:
            _INDEX = HybridIndex(_CFG.retrieval).build(load_corpus(_CFG.retrieval.corpus_path))
    return _INDEX


def set_index(index: HybridIndex) -> None:
    global _INDEX
    _INDEX = index


# ------------------------------------------------------------------ tools
@server.tool()
def search_papers(
    query: str,
    k: int = 5,
    year_min: int | None = None,
    year_max: int | None = None,
    exclude_surveys: bool = False,
) -> list[dict[str, Any]]:
    """Search the paper corpus. Returns ranked stubs with paper_id, title, year, score.

    Use a descriptive natural-language query, not keywords. Call `get_paper` afterwards
    for the abstract of anything you intend to recommend.
    """
    hits = get_index().search(
        query,
        k=k,
        filters=Filters(year_min=year_min, year_max=year_max, exclude_surveys=exclude_surveys),
    )
    return [
        {
            "paper_id": h.paper.paper_id,
            "title": h.paper.title,
            "year": h.paper.year,
            "categories": h.paper.categories,
            "score": round(h.score, 5),
            "source": h.source,
        }
        for h in hits
    ]


@server.tool()
def get_paper(paper_id: str) -> dict[str, Any]:
    """Fetch a full paper record, including the abstract. Errors if the id is unknown —
    which is deliberate: a hallucinated id must fail loudly, not return an empty dict."""
    paper = get_index().by_id.get(paper_id)
    if paper is None:
        raise ValueError(f"unknown paper_id: {paper_id!r}")
    return paper.model_dump()


@server.tool()
def similar_papers(paper_id: str, k: int = 5) -> list[dict[str, Any]]:
    """Find papers similar to a given one. Use for 'more like this' style refinements."""
    return [
        {"paper_id": e.paper.paper_id, "title": e.paper.title, "year": e.paper.year,
         "score": round(e.score, 5)}
        for e in get_index().similar_to(paper_id, k=k)
    ]


def _profiles_path() -> Path:
    p = Path(_CFG.agents.profile_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_profiles() -> dict[str, Any]:
    p = _profiles_path()
    return json.loads(p.read_text()) if p.exists() else {}


@server.tool()
def get_user_profile(user_id: str) -> dict[str, Any]:
    """Read a user's stored taste profile: liked ids, disliked ids, free-text notes."""
    return _read_profiles().get(
        user_id, {"user_id": user_id, "liked": [], "disliked": [], "notes": []}
    )


@server.tool()
def update_user_profile(
    user_id: str,
    liked: list[str] | None = None,
    disliked: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Record feedback about a user so later turns can personalise. Additive only."""
    profiles = _read_profiles()
    prof = profiles.get(user_id, {"user_id": user_id, "liked": [], "disliked": [], "notes": []})
    prof["liked"] = sorted(set(prof["liked"]) | set(liked or []))
    prof["disliked"] = sorted(set(prof["disliked"]) | set(disliked or []))
    if note:
        prof["notes"] = [*prof["notes"], note][-10:]
    profiles[user_id] = prof
    _profiles_path().write_text(json.dumps(profiles, indent=2))
    return prof


if __name__ == "__main__":  # pragma: no cover
    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    server.run(transport=transport)
