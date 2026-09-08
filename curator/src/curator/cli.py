"""Command line: `curator index`, `curator chat`, `curator ask`, `curator tools`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .graph import build_curator
from .retrieval.store import HybridIndex, load_corpus
from .tools.client import Toolbox
from .tools.server import server as mcp_server

app = typer.Typer(add_completion=False, help="Curator — grounded conversational paper recommender")
console = Console()


@app.command()
def index(config: str = typer.Option(None, "--config", "-c")) -> None:
    """Build the hybrid index from the corpus and persist it to data/index."""
    cfg = load_config(config)
    papers = load_corpus(cfg.retrieval.corpus_path)
    console.print(f"embedding {len(papers)} papers with {cfg.retrieval.embed_model} …")
    idx = HybridIndex(cfg.retrieval).build(papers)
    idx.save(cfg.retrieval.index_dir)
    console.print(f"[green]indexed[/] {len(papers)} papers -> {cfg.retrieval.index_dir}")


@app.command()
def tools(config: str = typer.Option(None, "--config", "-c")) -> None:
    """List what the MCP server advertises — the agent sees exactly this."""
    cfg = load_config(config)
    with Toolbox(mcp_server) as tb:
        table = Table("tool", "arguments", "description")
        for t in tb.schemas():
            props = (t["parameters"] or {}).get("properties", {})
            table.add_row(t["name"], ", ".join(props), t["description"].splitlines()[0])
        console.print(table)
    console.print(f"[dim]config fingerprint {cfg.fingerprint()}[/]")


@app.command()
def ask(
    query: str,
    config: str = typer.Option(None, "--config", "-c"),
    show_trace: bool = typer.Option(False, "--trace"),
) -> None:
    """Answer a single question."""
    cfg = load_config(config)
    with Toolbox(mcp_server) as tb:
        curator = build_curator(cfg, tb)
        answer, trace = curator.answer(query)
    _render(answer, trace, show_trace)


@app.command()
def chat(config: str = typer.Option(None, "--config", "-c")) -> None:
    """Multi-turn session. History is what makes 'refine' intents meaningful."""
    cfg = load_config(config)
    history: list[dict[str, str]] = []
    with Toolbox(mcp_server) as tb:
        curator = build_curator(cfg, tb)
        console.print("[dim]ctrl-c to exit[/]")
        while True:
            try:
                q = console.input("[bold cyan]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            answer, trace = curator.answer(q, history=history)
            _render(answer, trace, False)
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer.reply})


def _render(answer, trace, show_trace: bool) -> None:
    console.print(Panel(answer.reply, title="curator", border_style="cyan"))
    for rec in answer.recommendations:
        console.print(f"  [bold]{rec.paper_id}[/] — {rec.why}  [dim]{rec.citations}[/]")
    console.print(
        f"[dim]{trace.latency_ms:.0f} ms · {trace.llm_calls} llm calls · "
        f"{len(trace.tool_calls)} tool calls · {trace.critic_rounds} critic rounds[/]"
    )
    if show_trace:
        console.print_json(json.dumps(trace.model_dump(mode="json"), default=str))


@app.command()
def corpus_stats(config: str = typer.Option(None, "--config", "-c")) -> None:
    """Sanity-check the corpus before you trust any metric computed on it."""
    cfg = load_config(config)
    papers = load_corpus(cfg.retrieval.corpus_path)
    years = [p.year for p in papers if p.year]
    console.print(f"{len(papers)} papers · years {min(years)}–{max(years)}")
    cats: dict[str, int] = {}
    for p in papers:
        for c in p.categories:
            cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items(), key=lambda kv: -kv[1])[:10]:
        console.print(f"  {c:<12} {n}")
    console.print(f"[dim]{Path(cfg.retrieval.corpus_path).resolve()}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
