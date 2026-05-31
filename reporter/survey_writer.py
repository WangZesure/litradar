from typing import List, Dict, Any
from datetime import datetime


class SurveyWriter:
    def write(
        self,
        papers: List[Dict[str, Any]],
        core_papers: List[Dict[str, Any]],
        timeline: Dict[int, List[Dict[str, Any]]],
        llm_survey: str,
        output_path: str,
    ) -> None:
        sections = []
        sections.append(self._header(papers))
        sections.append(self._timeline_section(timeline))
        sections.append(self._method_overview(papers))
        if llm_survey:
            sections.append(f"## Detailed Survey\n\n{llm_survey}")
        sections.append(self._core_papers_section(core_papers))
        sections.append(self._benchmark_table(papers))
        sections.append(self._trends_section(papers))
        report = "\n\n".join(sections)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] Written to {output_path}")

    def _header(self, papers: List[Dict[str, Any]]) -> str:
        years = [p.get("year") for p in papers if p.get("year")]
        min_y, max_y = min(years), max(years) if years else ("N/A", "N/A")
        return (
            f"# Literature Survey: Diffusion Models for Anomaly Detection\n\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Total papers analyzed: **{len(papers)}**\n\n"
            f"Year range: **{min_y} - {max_y}**\n\n"
            f"---"
        )

    def _timeline_section(self, timeline: Dict[int, List[Dict[str, Any]]]) -> str:
        lines = ["## Timeline\n\n| Year | # Papers | Notable Papers |", "|------|----------|----------------|"]
        for year, papers in sorted(timeline.items()):
            notable = "; ".join([p.get("title", "")[:50] for p in papers[:3]])
            lines.append(f"| {year} | {len(papers)} | {notable} |")
        return "\n".join(lines)

    def _method_overview(self, papers: List[Dict[str, Any]]) -> str:
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for p in papers:
            cat = p.get("analysis", {}).get("method_category", "other")
            categories.setdefault(cat, []).append(p)
        lines = ["## Method Categories\n"]
        for cat, cat_papers in sorted(categories.items(), key=lambda x: -len(x[1])):
            lines.append(f"### {cat.replace('_', ' ').title()} ({len(cat_papers)} papers)\n")
            for p in cat_papers[:5]:
                a = p.get("analysis", {})
                lines.append(f"- **{p.get('title', '')}** ({p.get('year', '')}) — {a.get('key_contribution', '')}")
            lines.append("")
        return "\n".join(lines)

    def _core_papers_section(self, core_papers: List[Dict[str, Any]]) -> str:
        lines = ["## Core Papers (Top by Influence)\n"]
        for i, p in enumerate(core_papers, 1):
            a = p.get("analysis", {})
            lines.append(
                f"**{i}. {p.get('title', '')}** ({p.get('year', '')})\n\n"
                f"- Authors: {', '.join(p.get('authors', [])[:5])}\n"
                f"- Citations: {p.get('citation_count', 'N/A')}\n"
                f"- Method: {a.get('method_category', 'N/A')} | Diffusion: {a.get('diffusion_type', 'N/A')}\n"
                f"- Contribution: {a.get('key_contribution', '')}\n"
                f"- Techniques: {', '.join(a.get('novel_techniques', []))}\n"
            )
        return "\n".join(lines)

    def _benchmark_table(self, papers: List[Dict[str, Any]]) -> str:
        lines = ["## Benchmark Comparison\n\n| Paper | Year | Dataset | AUROC | AUPR | F1 |", "|-------|------|---------|-------|------|----|"]
        for p in papers:
            a = p.get("analysis", {})
            metrics = a.get("performance_metrics", {})
            if not metrics:
                continue
            datasets = ", ".join(a.get("datasets_used", []))
            lines.append(
                f"| {p.get('title', '')[:40]} | {p.get('year', '')} | {datasets} | "
                f"{metrics.get('AUROC', '-')} | {metrics.get('AUPR', '-')} | {metrics.get('F1', '-')} |"
            )
        if len(lines) <= 2:
            return "## Benchmark Comparison\n\nNo benchmark data extracted."
        return "\n".join(lines)

    def _trends_section(self, papers: List[Dict[str, Any]]) -> str:
        diffusion_types: Dict[str, int] = {}
        domains: Dict[str, int] = {}
        for p in papers:
            a = p.get("analysis", {})
            dt = a.get("diffusion_type", "other")
            dom = a.get("application_domain", "other")
            diffusion_types[dt] = diffusion_types.get(dt, 0) + 1
            domains[dom] = domains.get(dom, 0) + 1
        lines = ["## Trends\n\n### Diffusion Model Types\n\n| Type | Count |", "|------|-------|"]
        for dt, count in sorted(diffusion_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {dt} | {count} |")
        lines.append("\n### Application Domains\n\n| Domain | Count |")
        lines.append("|--------|-------|")
        for dom, count in sorted(domains.items(), key=lambda x: -x[1]):
            lines.append(f"| {dom} | {count} |")
        return "\n".join(lines)
