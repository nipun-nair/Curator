"""The MCP tool surface is a contract. These tests are what stops it drifting."""

from __future__ import annotations

import pytest

from curator.retrieval.store import Filters


def test_tools_are_discoverable_with_schemas(toolbox):
    names = toolbox.names()
    assert {"search_papers", "get_paper", "similar_papers", "get_user_profile"} <= names
    schema = next(t for t in toolbox.schemas() if t["name"] == "search_papers")
    props = schema["parameters"]["properties"]
    assert "query" in props and "k" in props
    assert schema["description"], "every tool needs a docstring; the agent reads it"


def test_prompt_block_is_generated_from_discovery(toolbox):
    block = toolbox.prompt_block()
    assert "search_papers(" in block
    # adding a tool to the server must not require a prompt edit
    assert all(name in block for name in toolbox.names())


def test_search_returns_ranked_hits(toolbox):
    res = toolbox.call("search_papers", {"query": "retrieval augmented generation", "k": 3})
    assert res.ok
    assert len(res.content) == 3
    assert all("paper_id" in h for h in res.content)
    scores = [h["score"] for h in res.content]
    assert scores == sorted(scores, reverse=True)


def test_unknown_tool_is_data_not_exception(toolbox):
    res = toolbox.call("delete_everything", {})
    assert res.ok is False and "no such tool" in res.error


def test_unknown_paper_id_fails_loudly(toolbox):
    res = toolbox.call("get_paper", {"paper_id": "NOT-A-REAL-ID"})
    assert res.ok is False


def test_profile_round_trip(toolbox, tmp_path, monkeypatch):
    from curator.tools import server as srv

    monkeypatch.setattr(srv._CFG.agents, "profile_path", str(tmp_path / "profiles.json"))
    toolbox.call("update_user_profile", {"user_id": "u1", "liked": ["FIX-0001"], "note": "likes RAG"})
    res = toolbox.call("get_user_profile", {"user_id": "u1"})
    assert res.ok and "FIX-0001" in res.content["liked"]


def test_every_call_is_recorded(toolbox):
    before = len(toolbox.history)
    toolbox.call("search_papers", {"query": "vector database", "k": 2})
    toolbox.call("nope", {})
    assert len(toolbox.history) == before + 2
    assert [r.ok for r in toolbox.history[-2:]] == [True, False]


@pytest.mark.parametrize(
    "filters,expect_all",
    [
        (Filters(year_min=2025), lambda p: p.year >= 2025),
        (Filters(exclude_surveys=True), lambda p: "survey" not in p.title.lower()),
    ],
)
def test_index_filters(index, filters, expect_all):
    hits = index.search("retrieval augmented generation", k=10, filters=filters)
    assert hits
    assert all(expect_all(h.paper) for h in hits)
