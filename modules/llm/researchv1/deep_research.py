import datetime as dt
import ollama
import logging
import json
import requests
import time
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass, asdict
from enum import Enum


from modules.llm.enchanced_ollama import EnhancedOllamaModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResearchStage(Enum):
    """Different stages of the research process"""

    INITIAL_QUERY = "initial_query"
    WEB_SEARCH = "web_search"
    CONTENT_EXTRACTION = "content_extraction"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    FINAL_REPORT = "final_report"


@dataclass
class SearchResult:
    """Structure for search results"""

    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    content: str = ""
    extracted_at: str = ""


@dataclass
class ResearchFinding:
    """Structure for research findings"""

    topic: str
    finding: str
    sources: List[str]
    confidence_level: float
    stage: ResearchStage
    timestamp: str


class WebSearcher:
    """Web searching capabilities with multiple search engines"""

    def __init__(self, search_apis: Dict[str, str] = None):
        """Initialize with API keys for different search services"""
        self.search_apis = search_apis or {}
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_duckduckgo(
        self, query: str, max_results: int = 10
    ) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key required)"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            async with self.session.get(url, params=params) as response:
                data = await response.json()

                results = []
                # Get instant answer if available
                if data.get("AbstractText"):
                    results.append(
                        SearchResult(
                            title=data.get("Heading", "DuckDuckGo Instant Answer"),
                            url=data.get("AbstractURL", ""),
                            snippet=data.get("AbstractText", ""),
                            source="duckduckgo_instant",
                            extracted_at=dt.datetime.now().isoformat(),
                        )
                    )

                # Get related topics
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(
                            SearchResult(
                                title=topic.get("Text", "").split(" - ")[0],
                                url=topic.get("FirstURL", ""),
                                snippet=topic.get("Text", ""),
                                source="duckduckgo_related",
                                extracted_at=dt.datetime.now().isoformat(),
                            )
                        )

                return results[:max_results]

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def search_serp_api(
        self, query: str, api_key: str, max_results: int = 10
    ) -> List[SearchResult]:
        """Search using SERP API (requires API key)"""
        if not api_key:
            logger.warning("SERP API key not provided")
            return []

        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "num": max_results,
            }

            response = requests.get(url, params=params)
            data = response.json()

            results = []
            for result in data.get("organic_results", []):
                results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source="serp_api",
                        extracted_at=dt.datetime.now().isoformat(),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"SERP API search error: {e}")
            return []

    async def extract_content(self, url: str, max_length: int = 5000) -> str:
        """Extract content from a webpage"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Remove unwanted elements
                    for element in soup(
                        ["script", "style", "nav", "header", "footer", "ads"]
                    ):
                        element.decompose()

                    # Extract main content
                    content = soup.get_text(separator=" ", strip=True)

                    # Clean up whitespace
                    content = re.sub(r"\s+", " ", content)

                    return (
                        content[:max_length] if len(content) > max_length else content
                    )

        except Exception as e:
            logger.error(f"Content extraction error for {url}: {e}")
            return ""


class DeepResearchFramework:
    """Main research framework combining Ollama and web search capabilities"""

    def __init__(
        self,
        model_name: str = "llama2",
        system_prompt: str = None,
        search_apis: Dict[str, str] = None,
        max_search_results: int = 10,
        content_extraction_limit: int = 5,
    ):

        self.model_name = model_name
        self.system_prompt = system_prompt or self._default_research_prompt()
        self.search_apis = search_apis or {}
        self.max_search_results = max_search_results
        self.content_extraction_limit = content_extraction_limit

        # Initialize Ollama model
        self.ollama_model = EnhancedOllamaModel(
            model_name=model_name, system_prompt=self.system_prompt
        )

        # Research state
        self.research_findings: List[ResearchFinding] = []
        self.search_results: List[SearchResult] = []
        self.current_stage = ResearchStage.INITIAL_QUERY
        self.research_id = dt.datetime.now().isoformat()

    def _default_research_prompt(self) -> str:
        """Default system prompt for research tasks"""
        return """You are an advanced AI research assistant designed to conduct thorough, multi-stage research.

Your capabilities include:
- Analyzing complex research questions
- Formulating targeted search queries
- Evaluating source credibility and relevance
- Synthesizing information from multiple sources
- Identifying knowledge gaps and areas needing further investigation
- Providing evidence-based conclusions with confidence levels

Guidelines:
1. Always think critically and question assumptions
2. Provide confidence levels for your findings (0.0 to 1.0)
3. Clearly distinguish between facts, opinions, and speculation
4. Identify when more research is needed
5. Cite sources appropriately
6. Look for contradictory information and address it
7. Consider multiple perspectives on controversial topics

When analyzing sources, consider:
- Credibility and authority of the source
- Recency and relevance of information
- Potential bias or agenda
- Supporting evidence provided
- Consistency with other sources"""

    async def conduct_research(
        self,
        research_question: str,
        depth_level: int = 3,
        follow_up_questions: List[str] = None,
    ) -> Dict[str, Any]:
        """Conduct comprehensive research on a given question"""

        logger.info(f"Starting research on: {research_question}")
        self.current_stage = ResearchStage.INITIAL_QUERY

        research_results = {
            "research_id": self.research_id,
            "question": research_question,
            "started_at": dt.datetime.now().isoformat(),
            "stages": {},
            "findings": [],
            "sources": [],
            "confidence_level": 0.0,
            "recommendations": [],
        }

        try:
            # Stage 1: Initial Analysis and Query Formulation
            await self._stage_initial_analysis(research_question, research_results)

            # Stage 2: Web Search
            await self._stage_web_search(research_question, research_results)

            # Stage 3: Content Extraction and Analysis
            await self._stage_content_analysis(research_results)

            # Stage 4: Synthesis
            await self._stage_synthesis(research_results)

            # Stage 5: Follow-up Research (if needed)
            if follow_up_questions and depth_level > 1:
                await self._stage_follow_up(
                    follow_up_questions, research_results, depth_level - 1
                )

            # Stage 6: Final Report Generation
            await self._stage_final_report(research_results)

        except Exception as e:
            logger.error(f"Research error: {e}")
            research_results["error"] = str(e)

        research_results["completed_at"] = dt.datetime.now().isoformat()
        return research_results

    async def _stage_initial_analysis(self, question: str, results: Dict) -> None:
        """Stage 1: Analyze the research question and formulate search strategies"""
        self.current_stage = ResearchStage.INITIAL_QUERY
        logger.info("Stage 1: Initial analysis")

        analysis_prompt = f"""
        Analyze this research question and provide:
        1. Key concepts and topics to investigate
        2. 5-7 specific search queries to gather comprehensive information
        3. Potential challenges or limitations in researching this topic
        4. Expected types of sources that would be most valuable
        
        Research question: {question}
        
        Format your response as JSON with keys: concepts, search_queries, challenges, valuable_sources
        """

        analysis_response = self.ollama_model.get_response(analysis_prompt)

        try:
            # Try to parse JSON response
            analysis_data = json.loads(analysis_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response
            analysis_data = {
                "concepts": [question],
                "search_queries": [question],
                "challenges": ["Response parsing error"],
                "valuable_sources": [
                    "Academic papers",
                    "News articles",
                    "Official reports",
                ],
            }

        results["stages"]["initial_analysis"] = {
            "stage": ResearchStage.INITIAL_QUERY.value,
            "timestamp": dt.datetime.now().isoformat(),
            "analysis": analysis_data,
            "raw_response": analysis_response,
        }

        # Store search queries for next stage
        self.search_queries = analysis_data.get("search_queries", [question])

    async def _stage_web_search(self, question: str, results: Dict) -> None:
        """Stage 2: Perform web searches"""
        self.current_stage = ResearchStage.WEB_SEARCH
        logger.info("Stage 2: Web searching")

        all_search_results = []

        async with WebSearcher(self.search_apis) as searcher:
            # Search with multiple queries
            for query in self.search_queries[:5]:  # Limit to 5 queries
                logger.info(f"Searching for: {query}")

                # Try DuckDuckGo first (no API key needed)
                ddg_results = await searcher.search_duckduckgo(
                    query, self.max_search_results
                )
                all_search_results.extend(ddg_results)

                # Try SERP API if available
                if "serp_api" in self.search_apis:
                    serp_results = searcher.search_serp_api(
                        query, self.search_apis["serp_api"], self.max_search_results
                    )
                    all_search_results.extend(serp_results)

                # Rate limiting
                await asyncio.sleep(1)

        # Remove duplicates and rank by relevance
        unique_results = self._deduplicate_results(all_search_results)
        ranked_results = await self._rank_search_results(unique_results, question)

        self.search_results = ranked_results[: self.max_search_results]

        results["stages"]["web_search"] = {
            "stage": ResearchStage.WEB_SEARCH.value,
            "timestamp": dt.datetime.now().isoformat(),
            "queries_used": self.search_queries,
            "total_results": len(all_search_results),
            "unique_results": len(unique_results),
            "final_results": len(self.search_results),
        }

        results["sources"] = [asdict(result) for result in self.search_results]

    async def _stage_content_analysis(self, results: Dict) -> None:
        """Stage 3: Extract and analyze content from top sources"""
        self.current_stage = ResearchStage.CONTENT_EXTRACTION
        logger.info("Stage 3: Content extraction and analysis")

        async with WebSearcher() as searcher:
            # Extract content from top sources
            extraction_tasks = []
            for result in self.search_results[: self.content_extraction_limit]:
                if result.url:
                    content = await searcher.extract_content(result.url)
                    result.content = content
                    result.extracted_at = dt.datetime.now().isoformat()

        # Analyze each piece of content
        content_analyses = []
        for result in self.search_results[: self.content_extraction_limit]:
            if result.content:
                analysis_prompt = f"""
                Analyze this content for research purposes:
                
                Source: {result.title} ({result.url})
                Content: {result.content[:2000]}...
                
                Provide:
                1. Key findings relevant to the research question
                2. Credibility assessment (0.0 to 1.0)
                3. Potential bias or limitations
                4. Important quotes or data points
                
                Format as JSON with keys: key_findings, credibility, bias_assessment, important_quotes
                """

                analysis = self.ollama_model.get_response(analysis_prompt)
                content_analyses.append({"source": result.url, "analysis": analysis})

        results["stages"]["content_analysis"] = {
            "stage": ResearchStage.CONTENT_EXTRACTION.value,
            "timestamp": dt.datetime.now().isoformat(),
            "sources_analyzed": len(content_analyses),
            "analyses": content_analyses,
        }

    async def _stage_synthesis(self, results: Dict) -> None:
        """Stage 4: Synthesize findings from all sources"""
        self.current_stage = ResearchStage.SYNTHESIS
        logger.info("Stage 4: Synthesis")

        # Prepare synthesis prompt with all findings
        sources_summary = "\n\n".join(
            [
                f"Source {i+1}: {result.title}\nURL: {result.url}\nContent: {result.content[:1000]}..."
                for i, result in enumerate(
                    self.search_results[: self.content_extraction_limit]
                )
                if result.content
            ]
        )

        synthesis_prompt = f"""
        Based on the research conducted, synthesize the findings to answer the original question.
        
        Original Question: {results['question']}
        
        Sources analyzed:
        {sources_summary}
        
        Provide a comprehensive synthesis including:
        1. Direct answer to the research question
        2. Supporting evidence from sources
        3. Contradictions or uncertainties found
        4. Overall confidence level (0.0 to 1.0)
        5. Areas needing further research
        6. Actionable recommendations
        
        Format as JSON with keys: answer, evidence, contradictions, confidence, further_research, recommendations
        """

        synthesis_response = self.ollama_model.get_response(
            synthesis_prompt, max_tokens=3000
        )

        results["stages"]["synthesis"] = {
            "stage": ResearchStage.SYNTHESIS.value,
            "timestamp": dt.datetime.now().isoformat(),
            "synthesis": synthesis_response,
        }

        # Try to extract confidence level
        try:
            synthesis_data = json.loads(synthesis_response)
            results["confidence_level"] = synthesis_data.get("confidence", 0.5)
            results["recommendations"] = synthesis_data.get("recommendations", [])
        except json.JSONDecodeError:
            results["confidence_level"] = 0.5

    async def _stage_follow_up(
        self, follow_up_questions: List[str], results: Dict, depth_level: int
    ) -> None:
        """Stage 5: Conduct follow-up research on specific questions"""
        logger.info("Stage 5: Follow-up research")

        follow_up_results = []
        for question in follow_up_questions:
            sub_research = await self.conduct_research(question, depth_level, None)
            follow_up_results.append(sub_research)

        results["stages"]["follow_up"] = {
            "stage": "follow_up",
            "timestamp": dt.datetime.now().isoformat(),
            "follow_up_research": follow_up_results,
        }

    async def _stage_final_report(self, results: Dict) -> None:
        """Stage 6: Generate final research report"""
        self.current_stage = ResearchStage.FINAL_REPORT
        logger.info("Stage 6: Final report generation")

        report_prompt = f"""
        Generate a comprehensive research report based on all findings.
        
        Research Question: {results['question']}
        Confidence Level: {results.get('confidence_level', 'Unknown')}
        
        Include:
        1. Executive Summary
        2. Methodology
        3. Key Findings
        4. Sources and Evidence
        5. Limitations and Uncertainties
        6. Conclusions
        7. Recommendations for Further Research
        
        Make the report professional, well-structured, and actionable.
        """

        final_report = self.ollama_model.get_response(report_prompt, max_tokens=4000)

        results["final_report"] = final_report
        results["stages"]["final_report"] = {
            "stage": ResearchStage.FINAL_REPORT.value,
            "timestamp": dt.datetime.now().isoformat(),
            "report_generated": True,
        }

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results based on URL"""
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    async def _rank_search_results(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Rank search results by relevance to the research question"""

        ranking_prompt = f"""
        Rank the following search results by relevance to this research question: "{query}"
        
        Results:
        {json.dumps([{'title': r.title, 'snippet': r.snippet, 'url': r.url} for r in results[:10]], indent=2)}
        
        Return only a JSON list of URLs in order of relevance (most relevant first).
        """

        try:
            ranking_response = self.ollama_model.get_response(ranking_prompt)
            ranked_urls = json.loads(ranking_response)

            # Reorder results based on ranking
            ranked_results = []
            url_to_result = {r.url: r for r in results}

            for url in ranked_urls:
                if url in url_to_result:
                    ranked_results.append(url_to_result[url])

            # Add any remaining results
            for result in results:
                if result not in ranked_results:
                    ranked_results.append(result)

            return ranked_results

        except Exception as e:
            logger.error(f"Ranking error: {e}")
            return results

    def save_research(self, filepath: str, results: Dict) -> None:
        """Save research results to file"""
        research_data = {
            "framework_version": "1.0",
            "model_used": self.model_name,
            "saved_at": dt.datetime.now().isoformat(),
            "results": results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Research saved to {filepath}")

    def load_research(self, filepath: str) -> Dict:
        """Load research results from file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("results", {})
