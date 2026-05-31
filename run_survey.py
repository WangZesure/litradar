import os
import sys
import yaml
from pathlib import Path
from fetcher import ArxivFetcher, SemanticScholarFetcher
from analyzer import LLMClient, PaperAnalyzer
from graph import CitationGraph
from reporter import CSVExporter, SurveyWriter


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_papers(arxiv_papers, s2_papers):
    merged = {}
    for p in s2_papers:
        key = p.get("arxiv_id") or p.get("s2_id")
        if key:
            merged[key] = p
    for p in arxiv_papers:
        key = p.get("arxiv_id")
        if key and key in merged:
            merged[key]["pdf_url"] = p.get("pdf_url", "")
            merged[key]["published"] = p.get("published", "")
            merged[key]["categories"] = p.get("categories", [])
            if not merged[key].get("abstract") and p.get("abstract"):
                merged[key]["abstract"] = p["abstract"]
        elif key and key not in merged:
            merged[key] = p
    print(f"[Merge] Total unique papers: {len(merged)}")
    return list(merged.values())


def main():
    config = load_config()
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(exist_ok=True)
    queries = config["queries"]

    print("=" * 60)
    print("  Diffusion Model x Anomaly Detection - Literature Survey")
    print("=" * 60)

    print("\n[Stage 1/4] Fetching papers...")
    arxiv_fetcher = ArxivFetcher(config["arxiv"])
    arxiv_papers = arxiv_fetcher.fetch(queries)

    s2_fetcher = SemanticScholarFetcher(config["semantic_scholar"])
    s2_papers = s2_fetcher.fetch(queries)

    papers = merge_papers(arxiv_papers, s2_papers)

    print(f"\n[Stage 2/4] Building citation graph...")
    graph = CitationGraph()
    graph.build(papers)

    top_k = config["output"].get("top_k_core_papers", 10)
    core_papers = graph.get_core_papers(top_k)
    most_cited = graph.get_most_cited(top_k)
    timeline = graph.get_timeline()

    graph.export_json(str(output_dir / "citation_graph.json"))
    print(f"  Core papers identified: {len(core_papers)}")
    print(f"  Timeline spans: {min(timeline.keys()) if timeline else 'N/A'} - {max(timeline.keys()) if timeline else 'N/A'}")

    print(f"\n[Stage 3/4] Analyzing papers with LLM...")
    llm = LLMClient(config["deepseek"])
    analyzer = PaperAnalyzer(llm)

    papers_with_abstract = [p for p in papers if p.get("abstract")]
    papers_with_abstract.sort(key=lambda x: x.get("year", 0), reverse=True)
    max_analyze = min(80, len(papers_with_abstract))
    papers_to_analyze = papers_with_abstract[:max_analyze]
    print(f"  Analyzing top {max_analyze} papers (sorted by recency)...")
    papers = analyzer.analyze_papers(papers_to_analyze)

    print("  Generating survey sections...")
    llm_survey = analyzer.generate_survey(papers)

    print(f"\n[Stage 4/4] Generating reports...")
    csv_exporter = CSVExporter()
    csv_exporter.export(papers, str(output_dir / "papers.csv"))

    survey_writer = SurveyWriter()
    survey_writer.write(
        papers=papers,
        core_papers=core_papers,
        timeline=timeline,
        llm_survey=llm_survey,
        output_path=str(output_dir / "survey_report.md"),
    )

    print("\n" + "=" * 60)
    print("  Survey complete!")
    print(f"  Report: {output_dir / 'survey_report.md'}")
    print(f"  CSV:    {output_dir / 'papers.csv'}")
    print(f"  Graph:  {output_dir / 'citation_graph.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
