"""Hybrid retrieval: BM25 + dense embeddings fused with Reciprocal Rank Fusion.

Two backends behind one interface:
  numpy  — brute-force cosine over an in-memory matrix. Zero infrastructure, which is
           what makes `make eval` runnable on a laptop and in CI.
  qdrant — a real vector database with payload filtering, for the deployed build.

Both are exercised by the same tests, so the ablation "does the vector DB change the
numbers" is answerable rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..schemas import Evidence, Paper


# --------------------------------------------------------------------------- corpus
def load_corpus(path: str | Path) -> list[Paper]:
    papers: list[Paper] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                papers.append(Paper.model_validate_json(line))
    return papers


def paper_text(p: Paper) -> str:
    return f"{p.title}\n\n{p.abstract}"


# --------------------------------------------------------------------------- embeddings
class Embedder:
    """Lazy wrapper so importing this module never pulls in torch."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # bge models want an instruction prefix on the query side only.
        if is_query and "bge" in self.model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------- filters
@dataclass
class Filters:
    year_min: int | None = None
    year_max: int | None = None
    exclude_surveys: bool = False

    def keep(self, p: Paper) -> bool:
        if self.year_min and (p.year or 0) < self.year_min:
            return False
        if self.year_max and (p.year or 9999) > self.year_max:
            return False
        if self.exclude_surveys and any(
            w in p.title.lower() for w in ("survey", "a review of", "systematic review")
        ):
            return False
        return True


# --------------------------------------------------------------------------- index
class HybridIndex:
    """Owns the corpus, the sparse index, the dense index, and the fusion."""

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.papers: list[Paper] = []
        self.by_id: dict[str, Paper] = {}
        self._bm25: Any = None
        self._matrix: np.ndarray | None = None
        self._embedder = Embedder(cfg.embed_model)
        self._qdrant: Any = None

    # ---- build / load ----
    def build(self, papers: list[Paper]) -> HybridIndex:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        self.papers = papers
        self.by_id = {p.paper_id: p for p in papers}
        self._bm25 = BM25Okapi([_tokenize(paper_text(p)) for p in papers])
        if self.cfg.mode in ("hybrid", "dense"):
            self._matrix = self._embedder.encode([paper_text(p) for p in papers])
            if self.cfg.store == "qdrant":
                self._push_to_qdrant()
        return self

    def save(self, index_dir: str | Path) -> None:
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "papers.jsonl", "w", encoding="utf-8") as fh:
            for p in self.papers:
                fh.write(p.model_dump_json() + "\n")
        if self._matrix is not None:
            np.save(d / "dense.npy", self._matrix)
        (d / "meta.json").write_text(
            json.dumps({"embed_model": self.cfg.embed_model, "n": len(self.papers)}, indent=2)
        )

    @classmethod
    def load(cls, cfg: RetrievalConfig) -> HybridIndex:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        d = Path(cfg.index_dir)
        idx = cls(cfg)
        idx.papers = load_corpus(d / "papers.jsonl")
        idx.by_id = {p.paper_id: p for p in idx.papers}
        idx._bm25 = BM25Okapi([_tokenize(paper_text(p)) for p in idx.papers])
        dense = d / "dense.npy"
        if dense.exists() and cfg.mode in ("hybrid", "dense"):
            idx._matrix = np.load(dense)
        return idx

    # ---- qdrant ----
    def _client(self) -> Any:
        if self._qdrant is None:
            from qdrant_client import QdrantClient  # noqa: PLC0415

            self._qdrant = QdrantClient(url=self.cfg.qdrant_url)
        return self._qdrant

    def _push_to_qdrant(self) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: PLC0415

        assert self._matrix is not None
        client = self._client()
        client.recreate_collection(
            collection_name=self.cfg.collection,
            vectors_config=VectorParams(size=self._matrix.shape[1], distance=Distance.COSINE),
        )
        client.upsert(
            collection_name=self.cfg.collection,
            points=[
                PointStruct(id=i, vector=self._matrix[i].tolist(), payload=p.model_dump())
                for i, p in enumerate(self.papers)
            ],
        )

    # ---- search ----
    def _bm25_rank(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(self.papers[i].paper_id, float(scores[i])) for i in order]

    def _dense_rank(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.cfg.store == "qdrant":
            qv = self._embedder.encode([query], is_query=True)[0]
            hits = self._client().search(
                collection_name=self.cfg.collection, query_vector=qv.tolist(), limit=k
            )
            return [(h.payload["paper_id"], float(h.score)) for h in hits]
        if self._matrix is None:
            return []
        qv = self._embedder.encode([query], is_query=True)[0]
        sims = self._matrix @ qv
        order = np.argsort(sims)[::-1][:k]
        return [(self.papers[i].paper_id, float(sims[i])) for i in order]

    def search(self, query: str, k: int | None = None, filters: Filters | None = None) -> list[Evidence]:
        k = k or self.cfg.top_k
        filters = filters or Filters()
        ck = self.cfg.candidate_k

        sparse = self._bm25_rank(query, ck) if self.cfg.mode in ("hybrid", "bm25") else []
        dense = self._dense_rank(query, ck) if self.cfg.mode in ("hybrid", "dense") else []

        if self.cfg.mode == "bm25":
            fused = [(pid, s, "bm25") for pid, s in sparse]
        elif self.cfg.mode == "dense":
            fused = [(pid, s, "dense") for pid, s in dense]
        else:
            fused = _rrf(sparse, dense, self.cfg.rrf_k)

        out: list[Evidence] = []
        for pid, score, source in fused:
            paper = self.by_id.get(pid)
            if paper is None or not filters.keep(paper):
                continue
            out.append(Evidence(paper=paper, score=score, source=source))  # type: ignore[arg-type]
            if len(out) >= k:
                break
        return out

    def similar_to(self, paper_id: str, k: int = 5) -> list[Evidence]:
        paper = self.by_id.get(paper_id)
        if paper is None:
            return []
        return [e for e in self.search(paper_text(paper), k=k + 1) if e.paper.paper_id != paper_id][:k]


def _rrf(
    a: list[tuple[str, float]], b: list[tuple[str, float]], k: int
) -> list[tuple[str, float, str]]:
    """Reciprocal rank fusion. Rank-based, so BM25's unbounded scores and cosine's
    [-1,1] never need calibrating against each other."""
    scores: dict[str, float] = {}
    for ranking in (a, b):
        for rank, (pid, _) in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(pid, s, "rrf") for pid, s in ordered]


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 1]


__all__ = ["Embedder", "Filters", "HybridIndex", "load_corpus", "paper_text"]
