import csv
from typing import List, Dict, Any


class CSVExporter:
    def export(self, papers: List[Dict[str, Any]], path: str) -> None:
        if not papers:
            return
        rows = [self._to_row(p) for p in papers]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[CSV] Exported {len(rows)} papers to {path}")

    def _to_row(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        a = paper.get("analysis", {})
        metrics = a.get("performance_metrics", {})
        return {
            "Title": paper.get("title", ""),
            "Authors": "; ".join(paper.get("authors", [])[:5]),
            "Year": paper.get("year", ""),
            "Venue": paper.get("venue", ""),
            "Citations": paper.get("citation_count", ""),
            "Published": paper.get("published", ""),
            "arXiv_ID": paper.get("arxiv_id", ""),
            "DOI": paper.get("doi", ""),
            "Method_Category": a.get("method_category", ""),
            "Diffusion_Type": a.get("diffusion_type", ""),
            "Application_Domain": a.get("application_domain", ""),
            "Key_Contribution": a.get("key_contribution", ""),
            "Datasets": "; ".join(a.get("datasets_used", [])),
            "AUROC": metrics.get("AUROC", ""),
            "AUPR": metrics.get("AUPR", ""),
            "F1": metrics.get("F1", ""),
            "Novel_Techniques": "; ".join(a.get("novel_techniques", [])),
            "Limitations": a.get("limitations", ""),
            "Relevance_Score": a.get("relevance_score", ""),
            "TLDR": paper.get("tldr", ""),
            "PDF_URL": paper.get("pdf_url", ""),
        }
