from typing import List, Dict, Any
from tqdm import tqdm
from .llm_client import LLMClient


SYSTEM_PROMPT = """You are an expert research analyst specializing in diffusion models and anomaly detection.
Analyze the given paper and extract structured information."""

ANALYZE_PROMPT = """Analyze the following paper and return a JSON object with these fields:

- "method_category": one of ["reconstruction_based", "likelihood_based", "score_based", "hybrid", "other"]
- "diffusion_type": one of ["DDPM", "DDIM", "Score_SDE", "Latent_Diffusion", "Flow_Matching", "other"]
- "application_domain": one of ["industrial_defect", "medical_imaging", "video_anomaly", "general_image", "tabular", "other"]
- "key_contribution": one sentence describing the main contribution
- "datasets_used": list of dataset names mentioned
- "performance_metrics": dict of metric_name to value (e.g. {{"AUROC": "95.2%"}})
- "novel_techniques": list of key techniques proposed
- "limitations": one sentence on limitations if mentioned, else ""
- "relevance_score": integer 1-10, how relevant to "diffusion model for anomaly detection"

Paper:
Title: {title}
Abstract: {abstract}
Year: {year}
Authors: {authors}

Return ONLY a valid JSON object, no other text."""


SUMMARIZE_BATCH_PROMPT = """You are writing a literature survey on "Diffusion Models for Anomaly Detection".
Given the following list of papers with their analyses, write a coherent survey section.

Papers:
{papers_json}

Write a survey section in Markdown that:
1. Groups papers by method category
2. Discusses the evolution of approaches
3. Compares performance on common benchmarks
4. Identifies trends and open challenges
5. Uses citation format: (Author et al., Year)

Write in academic English. Be specific about methods and results."""


class PaperAnalyzer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def analyze_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for paper in tqdm(papers, desc="Analyzing papers"):
            if paper.get("analysis"):
                continue
            analysis = self._analyze_single(paper)
            if analysis:
                paper["analysis"] = analysis
            else:
                paper["analysis"] = self._default_analysis()
        return papers

    def _analyze_single(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ANALYZE_PROMPT.format(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", "")[:3000],
            year=paper.get("year", "unknown"),
            authors=", ".join(paper.get("authors", [])[:5]),
        )
        return self.llm.chat_json(SYSTEM_PROMPT, prompt)

    def generate_survey(self, papers: List[Dict[str, Any]]) -> str:
        analyzed = [p for p in papers if p.get("analysis")]
        batch_size = 15
        sections = []
        for i in range(0, len(analyzed), batch_size):
            batch = analyzed[i:i + batch_size]
            batch_data = []
            for p in batch:
                batch_data.append({
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "authors": p.get("authors", [])[:3],
                    "analysis": p.get("analysis", {}),
                })
            prompt = SUMMARIZE_BATCH_PROMPT.format(
                papers_json=self._format_papers(batch_data)
            )
            section = self.llm.chat(SYSTEM_PROMPT, prompt)
            if section:
                sections.append(section)
        return "\n\n".join(sections)

    def _format_papers(self, batch: List[Dict[str, Any]]) -> str:
        lines = []
        for p in batch:
            a = p.get("analysis", {})
            lines.append(
                f"- [{p['year']}] {p['title']} | "
                f"Method: {a.get('method_category', 'N/A')} | "
                f"Diffusion: {a.get('diffusion_type', 'N/A')} | "
                f"Domain: {a.get('application_domain', 'N/A')} | "
                f"Contribution: {a.get('key_contribution', 'N/A')}"
            )
        return "\n".join(lines)

    def _default_analysis(self) -> Dict[str, Any]:
        return {
            "method_category": "other",
            "diffusion_type": "other",
            "application_domain": "other",
            "key_contribution": "",
            "datasets_used": [],
            "performance_metrics": {},
            "novel_techniques": [],
            "limitations": "",
            "relevance_score": 5,
        }
