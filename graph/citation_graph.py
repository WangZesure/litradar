import json
from typing import List, Dict, Any, Tuple
import networkx as nx


class CitationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.paper_map: Dict[str, Dict[str, Any]] = {}

    def build(self, papers: List[Dict[str, Any]]) -> None:
        for paper in papers:
            pid = paper.get("s2_id") or paper.get("arxiv_id")
            if not pid:
                continue
            self.paper_map[pid] = paper
            self.graph.add_node(pid, title=paper.get("title", ""), year=paper.get("year"))
            for cited_id in paper.get("references", []):
                self.graph.add_edge(pid, cited_id)
            for citing_id in paper.get("citations", []):
                self.graph.add_edge(citing_id, pid)

    def get_core_papers(self, top_k: int = 10) -> List[Dict[str, Any]]:
        if len(self.graph) == 0:
            return []
        try:
            pagerank = nx.pagerank(self.graph)
        except Exception:
            pagerank = {}
        scored = []
        for pid, score in pagerank.items():
            paper = self.paper_map.get(pid, {})
            if paper:
                paper["pagerank_score"] = round(score, 6)
                scored.append(paper)
        scored.sort(key=lambda x: x.get("pagerank_score", 0), reverse=True)
        return scored[:top_k]

    def get_most_cited(self, top_k: int = 10) -> List[Dict[str, Any]]:
        scored = []
        for paper in self.paper_map.values():
            cc = paper.get("citation_count", 0) or 0
            scored.append((cc, paper))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def get_timeline(self) -> Dict[int, List[Dict[str, Any]]]:
        timeline: Dict[int, List[Dict[str, Any]]] = {}
        for paper in self.paper_map.values():
            year = paper.get("year")
            if year:
                timeline.setdefault(year, []).append(paper)
        for year in timeline:
            timeline[year].sort(key=lambda x: x.get("citation_count", 0) or 0, reverse=True)
        return dict(sorted(timeline.items()))

    def get_clusters(self) -> List[List[Dict[str, Any]]]:
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        components.sort(key=len, reverse=True)
        clusters = []
        for comp in components[:5]:
            cluster = [self.paper_map[pid] for pid in comp if pid in self.paper_map]
            if cluster:
                clusters.append(cluster)
        return clusters

    def export_json(self, path: str) -> None:
        data = {
            "nodes": [
                {"id": n, "title": self.graph.nodes[n].get("title", ""), "year": self.graph.nodes[n].get("year")}
                for n in self.graph.nodes
            ],
            "edges": [{"source": u, "target": v} for u, v in self.graph.edges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
