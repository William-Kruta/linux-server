from dataclasses import dataclass, field
from typing import List
import datetime as dt
import re

from modules.llm.enchanced_ollama import EnhancedOllamaModel


@dataclass
class ResearchQuery:
    """Represents a research query with metadata"""

    query: str
    context: str = ""
    priority: int = 1  # 1-5, where 5 is highest priority
    tags: List[str] = field(default_factory=list)
    expected_depth: str = "medium"  # shallow, medium, deep
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    query_id: str = field(
        default_factory=lambda: f"query_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


@dataclass
class ResearchResult:
    """Represents research results with metadata"""

    query_id: str
    query: str
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    model_used: str = ""
    processing_time: float = 0.0


class ResearchAgent:
    """Specialized research agent with different research strategies"""

    def __init__(self, model: EnhancedOllamaModel, agent_type: str = "general"):
        self.model = model
        self.agent_type = agent_type
        self.specialization = self._get_specialization()

    def _get_specialization(self) -> str:
        """Get agent specialization prompt"""
        specializations = {
            "analyst": "You are a research analyst focused on data analysis, pattern recognition, and evidence-based conclusions.",
            "synthesizer": "You are a synthesis expert who excels at combining information from multiple sources into coherent insights.",
            "critic": "You are a critical thinker who questions assumptions, identifies biases, and points out potential flaws in reasoning.",
            "explorer": "You are an exploratory researcher who identifies new research directions and asks probing questions.",
            "general": "You are a general research assistant capable of handling diverse research tasks.",
        }
        return specializations.get(self.agent_type, specializations["general"])

    def research(self, query: ResearchQuery, context: str = "") -> ResearchResult:
        """Conduct research on a given query"""
        start_time = dt.datetime.now()

        # Construct research prompt
        prompt = self._build_research_prompt(query, context)

        # Get response
        response = self.model.get_response(prompt)

        # Parse response
        result = self._parse_research_response(query, response)
        result.model_used = self.model.model_name
        result.processing_time = (dt.datetime.now() - start_time).total_seconds()

        return result

    def _build_research_prompt(self, query: ResearchQuery, context: str = "") -> str:
        """Build a comprehensive research prompt"""
        prompt = f"""
{self.specialization}

Research Task: {query.query}

Context: {query.context + " " + context if query.context or context else "No additional context provided"}

Expected Depth: {query.expected_depth}

Please provide a comprehensive research response that includes:

1. **Main Answer**: A clear, well-reasoned answer to the research question
2. **Reasoning Steps**: Step-by-step explanation of your analysis
3. **Key Insights**: Important findings or patterns identified
4. **Confidence Assessment**: Rate your confidence in the answer (0-1 scale) and explain why
5. **Related Questions**: 3-5 follow-up questions that could deepen understanding
6. **Potential Sources**: Suggest types of sources or evidence that would strengthen this research

Format your response clearly with headers for each section.
"""
        return prompt

    def _parse_research_response(
        self, query: ResearchQuery, response: str
    ) -> ResearchResult:
        """Parse the research response into structured result"""
        # Extract confidence score
        confidence_match = re.search(
            r"confidence.*?(\d+\.?\d*)", response, re.IGNORECASE
        )
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        # Extract reasoning steps
        reasoning_pattern = r"(?:reasoning|steps|analysis):(.*?)(?=\n\n|\n#|\nKey|\nConfidence|\nRelated|\nPotential|$)"
        reasoning_match = re.search(
            reasoning_pattern, response, re.IGNORECASE | re.DOTALL
        )
        reasoning_steps = (
            [
                step.strip()
                for step in reasoning_match.group(1).split("\n")
                if step.strip()
            ]
            if reasoning_match
            else []
        )

        # Extract related questions
        related_pattern = (
            r"(?:related questions|follow-up):(.*?)(?=\n\n|\n#|\nPotential|$)"
        )
        related_match = re.search(related_pattern, response, re.IGNORECASE | re.DOTALL)
        related_queries = (
            [
                q.strip().lstrip("- ").lstrip("1234567890. ")
                for q in related_match.group(1).split("\n")
                if q.strip()
            ]
            if related_match
            else []
        )

        return ResearchResult(
            query_id=query.query_id,
            query=query.query,
            answer=response,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            related_queries=related_queries[:5],  # Limit to 5
        )
