"""Pull a real corpus from the arXiv API into data/corpus.jsonl.

Politeness matters here — arXiv asks for ~3s between requests and will throttle you
otherwise. The script is resumable: it appends and skips ids already on disk, so a
dropped connection costs one page, not the whole fetch.

    python scripts/fetch_corpus.py --categories cs.CL cs.IR cs.AI --max 4000
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
ROOT = Path(__file__).resolve().parents[1]


def parse_entry(entry: ET.Element) -> dict | None:
    raw_id = entry.findtext("a:id", default="", namespaces=NS)
    m = re.search(r"abs/([^v]+)", raw_id)
    if not m:
        return None
    published = entry.findtext("a:published", default="", namespaces=NS)
    return {
        "paper_id": m.group(1),
        "title": " ".join(entry.findtext("a:title", "", NS).split()),
        "abstract": " ".join(entry.findtext("a:summary", "", NS).split()),
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "authors": [
            a.findtext("a:name", "", NS) for a in entry.findall("a:author", NS)
        ][:12],
        "categories": [
            c.attrib.get("term", "") for c in entry.findall("a:category", NS)
        ],
        "url": raw_id,
    }


def fetch(categories: list[str], max_results: int, out: Path, page: int = 100) -> None:
    seen: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            seen = {json.loads(line)["paper_id"] for line in fh if line.strip()}
        print(f"resuming: {len(seen)} papers already on disk")

    query = "+OR+".join(f"cat:{c}" for c in categories)
    written = 0
    with open(out, "a", encoding="utf-8") as fh:
        for start in range(0, max_results, page):
            url = (
                f"{API}?search_query={query}&start={start}&max_results={page}"
                "&sortBy=submittedDate&sortOrder=descending"
            )
            try:
                r = httpx.get(url, timeout=60.0)
                r.raise_for_status()
            except Exception as exc:
                print(f"  page {start} failed ({exc}); stopping cleanly")
                break
            entries = ET.fromstring(r.text).findall("a:entry", NS)
            if not entries:
                break
            for e in entries:
                rec = parse_entry(e)
                if rec and rec["paper_id"] not in seen and rec["abstract"]:
                    seen.add(rec["paper_id"])
                    fh.write(json.dumps(rec) + "\n")
                    written += 1
            fh.flush()
            print(f"  {start + len(entries):>6} fetched · {written} new")
            time.sleep(3.0)  # arXiv API etiquette
    print(f"done: {written} new papers -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["cs.CL", "cs.IR"])
    ap.add_argument("--max", type=int, default=2000, dest="max_results")
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fetch(args.categories, args.max_results, out)


if __name__ == "__main__":
    main()
