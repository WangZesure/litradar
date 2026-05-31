import os
import time
import requests
from typing import List, Dict, Any, Optional
from tqdm import tqdm


S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,abstract,authors,year,venue,citationCount,referenceCount,tldr,externalIds,citations,references"


class SemanticScholarFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key", "")
        self.delay = config.get("delay_seconds", 3)
        self.max_citation_depth = config.get("max_citation_depth", 1)
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.semanticscholar.org"
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",api.semanticscholar.org"

    def fetch(self, queries: List[str]) -> List[Dict[str, Any]]:
        papers = {}
        for query in tqdm(queries, desc="Fetching from Semantic Scholar"):
            results = self._search(query)
            for paper in results:
                pid = paper["s2_id"]
                if pid and pid not in papers:
                    papers[pid] = paper
            time.sleep(self.delay)
        print(f"[S2] Fetched {len(papers)} unique papers")
        return list(papers.values())

    def fetch_citations(self, paper_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        citation_map = {}
        for pid in tqdm(paper_ids, desc="Fetching citation details"):
            detail = self._get_paper(pid)
            if detail:
                citation_map[pid] = detail
            time.sleep(self.delay)
        return citation_map

    def _search(self, query: str) -> List[Dict[str, Any]]:
        url = f"{S2_BASE_URL}/paper/search"
        params = {"query": query, "limit": 50, "fields": S2_FIELDS}
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            if resp.status_code == 429:
                print("[S2] Rate limited, waiting 60s...")
                time.sleep(60)
                resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [self._parse(p) for p in data.get("data", [])]
        except requests.exceptions.Timeout:
            print(f"[S2] Timeout for query: {query}")
            return []
        except Exception as e:
            print(f"[S2] Search error for '{query}': {e}")
            return []

    def _get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        url = f"{S2_BASE_URL}/paper/{paper_id}"
        params = {"fields": S2_FIELDS}
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            if resp.status_code == 429:
                time.sleep(60)
                resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[S2] Get paper error {paper_id}: {e}")
        return None

    def _parse(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        ext_ids = paper.get("externalIds") or {}
        tldr = paper.get("tldr")
        return {
            "s2_id": paper.get("paperId"),
            "arxiv_id": ext_ids.get("ArXiv", ""),
            "doi": ext_ids.get("DOI", ""),
            "title": (paper.get("title") or "").strip(),
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
            "abstract": (paper.get("abstract") or "").strip(),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "citation_count": paper.get("citationCount", 0),
            "reference_count": paper.get("referenceCount", 0),
            "tldr": tldr.get("text", "") if tldr else "",
            "citations": [c.get("paperId") for c in (paper.get("citations") or []) if c.get("paperId")],
            "references": [r.get("paperId") for r in (paper.get("references") or []) if r.get("paperId")],
            "source": "semantic_scholar",
        }
