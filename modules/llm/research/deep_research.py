from pathlib import Path
from typing import Any, Dict
import logging
import yaml
import json
import datetime as dt

from modules.llm.research.periphery import ResearchAgent, ResearchResult, ResearchQuery
from modules.llm.enchanced_ollama import EnhancedOllamaModel


class DeepResearchFramework:
    """Main research framework orchestrating multiple agents and research strategies"""

    def __init__(
        self, model_configs: Dict[str, Dict], output_dir: str = "research_output"
    ):
        self.models = {}
        self.agents = {}
        self.research_history = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize models and agents
        self._initialize_models(model_configs)
        self._initialize_agents()

        # Setup logging
        self._setup_logging()

    def _initialize_models(self, model_configs: Dict[str, Dict]):
        """Initialize different models for different purposes"""
        for model_name, config in model_configs.items():
            model = EnhancedOllamaModel(
                model_name=config["model"],
                system_prompt=config.get(
                    "system_prompt", "You are a helpful research assistant."
                ),
                history=config.get("history", []),
            )
            self.models[model_name] = model

    def _initialize_agents(self):
        """Initialize specialized research agents"""
        if "primary" in self.models:
            primary_model = self.models["primary"]
        else:
            primary_model = list(self.models.values())[0]

        agent_types = ["analyst", "synthesizer", "critic", "explorer"]

        for agent_type in agent_types:
            if agent_type in self.models:
                self.agents[agent_type] = ResearchAgent(
                    self.models[agent_type], agent_type
                )
            else:
                self.agents[agent_type] = ResearchAgent(primary_model, agent_type)

    def _setup_logging(self):
        """Setup logging for the research framework"""
        log_file = self.output_dir / "research.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def conduct_research(
        self, query: ResearchQuery, strategy: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Conduct research using specified strategy"""
        logging.info(f"Starting research for query: {query.query}")

        research_session = {
            "query": query,
            "strategy": strategy,
            "timestamp": dt.datetime.now().isoformat(),
            "results": {},
            "synthesis": "",
            "final_insights": [],
        }

        if strategy == "comprehensive":
            research_session = self._comprehensive_research(query, research_session)
        elif strategy == "parallel":
            research_session = self._parallel_research(query, research_session)
        elif strategy == "iterative":
            research_session = self._iterative_research(query, research_session)
        else:
            # Single agent research
            agent = self.agents.get("analyst", list(self.agents.values())[0])
            result = agent.research(query)
            research_session["results"]["single"] = result

        # Save research session
        self._save_research_session(research_session)
        self.research_history.append(research_session)

        logging.info(f"Research completed for query: {query.query}")
        return research_session

    def _comprehensive_research(self, query: ResearchQuery, session: Dict) -> Dict:
        """Comprehensive research using multiple agents sequentially"""
        # Phase 1: Initial analysis
        analyst_result = self.agents["analyst"].research(query)
        session["results"]["analysis"] = analyst_result

        # Phase 2: Exploration of related questions
        if analyst_result.related_queries:
            exploration_results = []
            for related_q in analyst_result.related_queries[:3]:  # Limit to 3
                related_query = ResearchQuery(
                    query=related_q,
                    context=f"Related to main query: {query.query}",
                    expected_depth="shallow",
                )
                result = self.agents["explorer"].research(related_query)
                exploration_results.append(result)
            session["results"]["exploration"] = exploration_results

        # Phase 3: Critical analysis
        critic_context = f"Original analysis: {analyst_result.answer}"
        critic_query = ResearchQuery(
            query=f"Critically evaluate this research: {query.query}",
            context=critic_context,
            expected_depth="medium",
        )
        critic_result = self.agents["critic"].research(critic_query)
        session["results"]["critique"] = critic_result

        # Phase 4: Synthesis
        synthesis_context = self._build_synthesis_context(session["results"])
        synthesis_query = ResearchQuery(
            query=f"Synthesize all research findings for: {query.query}",
            context=synthesis_context,
            expected_depth="deep",
        )
        synthesis_result = self.agents["synthesizer"].research(synthesis_query)
        session["results"]["synthesis"] = synthesis_result
        session["synthesis"] = synthesis_result.answer

        return session

    def _parallel_research(self, query: ResearchQuery, session: Dict) -> Dict:
        """Parallel research using multiple agents simultaneously"""
        # Note: This is a simplified parallel approach
        # In a real implementation, you might use asyncio or threading

        results = {}
        for agent_name, agent in self.agents.items():
            if agent_name != "synthesizer":  # Save synthesizer for final step
                specialized_query = ResearchQuery(
                    query=query.query,
                    context=f"{query.context} (Focus: {agent_name} perspective)",
                    expected_depth=query.expected_depth,
                )
                results[agent_name] = agent.research(specialized_query)

        session["results"]["parallel"] = results

        # Synthesize results
        synthesis_context = self._build_synthesis_context(results)
        synthesis_query = ResearchQuery(
            query=f"Synthesize multiple perspectives on: {query.query}",
            context=synthesis_context,
            expected_depth="deep",
        )
        synthesis_result = self.agents["synthesizer"].research(synthesis_query)
        session["results"]["synthesis"] = synthesis_result
        session["synthesis"] = synthesis_result.answer

        return session

    def _iterative_research(self, query: ResearchQuery, session: Dict) -> Dict:
        """Iterative research with progressive refinement"""
        current_query = query
        iterations = []

        for i in range(3):  # 3 iterations
            logging.info(f"Research iteration {i+1}")

            # Primary research
            result = self.agents["analyst"].research(current_query)

            # Critical review
            if i > 0:  # Skip critique on first iteration
                critic_query = ResearchQuery(
                    query=f"What gaps or improvements are needed in this research on: {query.query}",
                    context=result.answer,
                    expected_depth="medium",
                )
                critique = self.agents["critic"].research(critic_query)
                result.reasoning_steps.extend(
                    ["=== CRITIQUE ==="] + critique.reasoning_steps
                )

            iterations.append(result)

            # Prepare next iteration if needed
            if i < 2 and result.related_queries:
                # Create refined query for next iteration
                refined_question = (
                    result.related_queries[0] if result.related_queries else query.query
                )
                current_query = ResearchQuery(
                    query=refined_question,
                    context=f"Building on previous research: {result.answer}",
                    expected_depth=query.expected_depth,
                )

        session["results"]["iterations"] = iterations

        # Final synthesis
        synthesis_context = "\n\n".join(
            [f"Iteration {i+1}: {result.answer}" for i, result in enumerate(iterations)]
        )
        synthesis_query = ResearchQuery(
            query=f"Provide final synthesis of iterative research on: {query.query}",
            context=synthesis_context,
            expected_depth="deep",
        )
        synthesis_result = self.agents["synthesizer"].research(synthesis_query)
        session["results"]["synthesis"] = synthesis_result
        session["synthesis"] = synthesis_result.answer

        return session

    def _build_synthesis_context(self, results: Dict) -> str:
        """Build context for synthesis from multiple results"""
        context_parts = []
        for key, result in results.items():
            if isinstance(result, ResearchResult):
                context_parts.append(f"=== {key.upper()} ===\n{result.answer}")
            elif isinstance(result, list):
                for i, res in enumerate(result):
                    if isinstance(res, ResearchResult):
                        context_parts.append(
                            f"=== {key.upper()} {i+1} ===\n{res.answer}"
                        )

        return "\n\n".join(context_parts)

    def _save_research_session(self, session: Dict):
        """Save research session to file"""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_session_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert session to JSON-serializable format
        serializable_session = self._make_serializable(session)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_session, f, indent=2, ensure_ascii=False)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (ResearchQuery, ResearchResult)):
            return obj.__dict__
        else:
            return obj

    def generate_research_report(self, session: Dict, format: str = "markdown") -> str:
        """Generate a formatted research report"""
        if format == "markdown":
            return self._generate_markdown_report(session)
        elif format == "html":
            return self._generate_html_report(session)
        else:
            return str(session)

    def _generate_markdown_report(self, session: Dict) -> str:
        """Generate markdown research report"""
        report = f"""# Research Report

## Query
**Question:** {session['query'].query}
**Strategy:** {session['strategy']}
**Timestamp:** {session['timestamp']}

## Executive Summary
{session.get('synthesis', 'No synthesis available')}

## Detailed Findings

"""

        for key, result in session["results"].items():
            if key == "synthesis":
                continue

            report += f"### {key.title()}\n\n"

            if isinstance(result, list):
                for i, res in enumerate(result):
                    if hasattr(res, "answer"):
                        report += f"#### {key.title()} {i+1}\n{res.answer}\n\n"
            elif hasattr(result, "answer"):
                report += f"{result.answer}\n\n"

        return report

    def _generate_html_report(self, session: Dict) -> str:
        """Generate HTML research report"""
        # Basic HTML template - could be enhanced with CSS
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .summary {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Research Report</h1>
    <p><strong>Query:</strong> {session['query'].query}</p>
    <p><strong>Strategy:</strong> {session['strategy']}</p>
    <p><strong>Timestamp:</strong> {session['timestamp']}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{session.get('synthesis', 'No synthesis available')}</p>
    </div>
    
    <h2>Detailed Findings</h2>
"""

        for key, result in session["results"].items():
            if key == "synthesis":
                continue

            html += f"<h3>{key.title()}</h3>"

            if isinstance(result, list):
                for i, res in enumerate(result):
                    if hasattr(res, "answer"):
                        html += f"<h4>{key.title()} {i+1}</h4><p>{res.answer}</p>"
            elif hasattr(result, "answer"):
                html += f"<p>{result.answer}</p>"

        html += "</body></html>"
        return html


# Example usage and configuration
def create_research_framework_example():
    """Example of how to set up and use the research framework"""

    # Model configurations
    model_configs = {
        "primary": {
            "model": "llama3.1:8b",  # or whatever model you have
            "system_prompt": "You are an expert researcher with deep analytical capabilities.",
        },
        "analyst": {
            "model": "llama3.1:8b",
            "system_prompt": "You are a data analyst focused on evidence-based research and pattern recognition.",
        },
        "synthesizer": {
            "model": "llama3.1:8b",
            "system_prompt": "You are a synthesis expert who combines multiple sources into coherent insights.",
        },
    }

    # Create framework
    framework = DeepResearchFramework(model_configs)

    # Example research query
    query = ResearchQuery(
        query="What are the implications of artificial intelligence on future job markets?",
        context="Focus on both positive and negative impacts, with consideration for different sectors",
        expected_depth="deep",
        tags=["AI", "employment", "economics", "future"],
    )

    # Conduct research
    results = framework.conduct_research(query, strategy="comprehensive")

    # Generate report
    report = framework.generate_research_report(results, format="markdown")

    return framework, results, report


# Example usage and configuration
def create_research_framework_example():
    """Example of how to set up and use the research framework"""

    # Model configurations
    model_configs = {
        "primary": {
            "model": "gemma3:12b",  # or whatever model you have
            "system_prompt": "You are an expert researcher with deep analytical capabilities.",
        },
        "analyst": {
            "model": "gemma3:12b",
            "system_prompt": "You are a data analyst focused on evidence-based research and pattern recognition.",
        },
        "synthesizer": {
            "model": "gemma3:12b",
            "system_prompt": "You are a synthesis expert who combines multiple sources into coherent insights.",
        },
    }

    # Create framework
    framework = DeepResearchFramework(model_configs)

    # Example research query
    query = ResearchQuery(
        query="What are the implications of artificial intelligence on future job markets?",
        context="Focus on both positive and negative impacts, with consideration for different sectors",
        expected_depth="deep",
        tags=["AI", "employment", "economics", "future"],
    )

    # Conduct research
    results = framework.conduct_research(query, strategy="comprehensive")

    # Generate report
    report = framework.generate_research_report(results, format="markdown")

    return framework, results, report
