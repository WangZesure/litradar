import os
import time
import arxiv
from typing import List, Dict, Any
from tqdm import tqdm


class ArxivFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.max_results = config.get("max_results", 500)
        self.delay = config.get("delay_seconds", 3)
        os.environ["NO_PROXY"] = "export.arxiv.org,arxiv.org"
        os.environ["no_proxy"] = "export.arxiv.org,arxiv.org"

    def fetch(self, queries: List[str]) -> List[Dict[str, Any]]:
        papers = {}
        for query in tqdm(queries, desc="Fetching from arXiv"):
            results = self._search_with_retry(query)
            for paper in results:
                arxiv_id = paper["arxiv_id"]
                if arxiv_id not in papers:
                    papers[arxiv_id] = paper
            time.sleep(self.delay)
        print(f"[arXiv] Fetched {len(papers)} unique papers")
        return list(papers.values())

    def _search_with_retry(self, query: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        for attempt in range(max_retries):
            try:
                return self._search(query)
            except Exception as e:
                wait_time = (attempt + 1) * 5
                print(f"[arXiv] Attempt {attempt + 1} failed: {type(e).__name__}. Waiting {wait_time}s...")
                time.sleep(wait_time)
        print(f"[arXiv] Failed to fetch query: {query}")
        return []

    def _search(self, query: str) -> List[Dict[str, Any]]:
        client = arxiv.Client(
            page_size=100,
            delay_seconds=self.delay,
            num_retries=3,
        )
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for result in client.results(search):
            results.append(self._parse(result))
        return results

    def _parse(self, result: arxiv.Result) -> Dict[str, Any]:
        return {
            "arxiv_id": result.entry_id.split("/abs/")[-1],
            "title": result.title.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip().replace("\n", " "),
            "published": result.published.strftime("%Y-%m-%d"),
            "year": result.published.year,
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "source": "arxiv",
        }
