import logging
import datetime as dt
from enum import Enum

from modules.llm.researchv3.agents import (
    AnalystAgent,
    SynthesizerAgent,
    CriticAgent,
    ExplorerAgent,
)


# Configure logging to display informative messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResearchStage(Enum):
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CRITICISM = "criticism"
    EXPLORATION = "exploration"


class ResearchAgent:
    def __init__(self, research_config: dict = {}):
        if research_config == {}:
            research_config = {
                "analyst": {"model": "gemma3:12b", "context_window": 4096},
                "critic": {"model": "gemma3:12b", "context_window": 4096},
                "explorer": {"model": "gemma3:12b", "context_window": 4096},
                "synthesizer": {"model": "gemma3:12b", "context_window": 4096},
            }

        # Initialize your agents here, making them available to the instance
        self.analyst = AnalystAgent(research_config["analyst"]["model"])
        self.critic = CriticAgent(research_config["critic"]["model"])
        self.explorer = ExplorerAgent(research_config["explorer"]["model"])
        self.synthesizer = SynthesizerAgent(research_config["synthesizer"]["model"])
        # Assuming other helper methods like _critisize are also in this class

    def run_research(self, topic: str, recursion_depth: int = 3):
        """
        Public-facing method to start the recursive research process.

        Args:
            topic (str): The initial question or topic to start the research.
            recursion_depth (int): The maximum number of recursive steps.

        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  full state of one research step (topic, analysis, critique, etc.).
        """
        print(
            f"--- Starting Recursive Research for topic: '{topic}' with depth: {recursion_depth} ---"
        )
        return self._recursive_step(
            current_topic=topic,
            recursion_depth=recursion_depth,
            current_depth=0,
            research_history=[],
        )

    def _recursive_step(
        self,
        current_topic: str,
        recursion_depth: int,
        current_depth: int,
        research_history: list,
    ):
        """
        Performs a single research step and calls itself with the next question.

        Args:
            current_topic (str): The topic/question for the current step.
            recursion_depth (int): The maximum allowed depth.
            current_depth (int): The current position in the recursion.
            research_history (list): The accumulated results from previous steps.

        Returns:
            list: The final, populated research history.
        """
        # 1. Base Case: Stop recursion if we've reached the desired depth.
        if current_depth >= recursion_depth:
            print("\n--- Reached maximum recursion depth. Finalizing research. ---")
            return research_history

        print(
            f"\n--- [Depth {current_depth + 1}/{recursion_depth}] Researching topic: '{current_topic}' ---"
        )

        # 2. The Workflow: Execute one cycle of research.
        web_query = self._refine_prompt_for_web(current_topic, self.explorer)

        analyzer_response = self.analyst.run(task=current_topic, search_query=web_query)
        critic_response = self._critisize(analyzer_response, self.critic)
        synthesized_response = self._synthesize(
            analyzer_response, critic_response, self.synthesizer
        )

        # This is the crucial step that generates the input for the next recursive call.
        next_question = self._next_step(synthesized_response, self.explorer)

        # 3. Store State: Save the results of the current step.
        research_state = {
            "depth": current_depth,
            "topic": current_topic,
            "web_query": web_query,
            "analysis": analyzer_response,
            "critique": critic_response,
            "synthesis": synthesized_response,
            "next_question": next_question,  # This will be the topic for the next step
        }
        research_history.append(research_state)

        # --- Optional: Print progress for this step ---
        print(f"Analysis: {analyzer_response}")
        print(f"Critique: {critic_response}")
        print(f"Synthesis: {synthesized_response}")
        print(f"Next Question to Investigate: {next_question}")
        # ---------------------------------------------

        # 4. The Recursive Call: Call the function again with the new question.
        return self._recursive_step(
            current_topic=next_question,
            recursion_depth=recursion_depth,
            current_depth=current_depth + 1,
            research_history=research_history,
        )

    def _refine_prompt_for_web(self, prompt: str, agent):
        instructions = f"Based on the following prompt: '{prompt}', refine the users query for a google search. Return only the refined query. If relevant here is today's date: {dt.datetime.now().date()}"
        return agent.run(task=instructions)

    def _explore(
        self,
        topic: str,
        agent,
        context_window: int,
        search_query: str = None,
        instructions: str = "",
    ):
        prompt = f"Generate 3-4 key research questions about the topic: '{topic}'.\n\nIf relevant, here is the current date: {dt.datetime.now().date()}. Return only the questions. No other commentary."
        response = agent.run(
            task=prompt,
            search_query=search_query,
            context_window=context_window,
            max_tokens=context_window,
        )
        response = [q.strip() for q in response.split("\n") if q.strip()]
        return response

    def _critisize(self, prompt: str, agent):
        insturctions = f"""Critically evaluate the following analysis. Identify any potential biases,
unstated assumptions, or logical fallacies. Is the evidence strong enough
to support the conclusions? Your response should be a bullet point list. Only return the bullet point list. 
--- ANALYSIS ---
{prompt}"""
        response = agent.run(task=insturctions)
        response = [q.strip() for q in response.split("\n") if q.strip()]
        return response

    def _synthesize(self, analysis: str, criticism: str, agent):
        prompt = f"""Create a coherent summary that incorporates the initial analysis and the subsequent critique.
Present a balanced view based on both pieces of information.

--- INITIAL ANALYSIS ---
{analysis}

--- CRITIQUE ---
{criticism}"""
        response = agent.run(task=prompt)
        return response

    def _next_step(self, synthesis: str, agent):
        prompt = f"""
Based on the following research summary and critique, what are the most
important unanswered questions or next steps for a deeper investigation?

--- SUMMARY ---
{synthesis}
"""
        response = agent.run(task=prompt)
        return response


class Orchestrator:
    def __init__(
        self,
        research_config={
            "analyst": {"model": "gemma3:12b", "context_window": 4096},
            "critic": {"model": "gemma3:12b", "context_window": 4096},
            "explorer": {"model": "gemma3:12b", "context_window": 4096},
            "synthesizer": {"model": "gemma3:12b", "context_window": 4096},
        },
    ):
        self.analyst = AnalystAgent(research_config["analyst"]["model"])
        self.critic = CriticAgent(research_config["critic"]["model"])
        self.explorer = ExplorerAgent(research_config["explorer"]["model"])
        self.synthesizer = SynthesizerAgent(research_config["synthesizer"]["model"])

    def _create_models(self, flow):
        models = {}
        for fl in flow:
            if fl["stage"] == ResearchStage.ANALYSIS:
                models["analyst"] = AnalystAgent(fl["model"])
            elif fl["stage"] == ResearchStage.SYNTHESIS:
                models["synthesizer"] = SynthesizerAgent(fl["model"])
            elif fl["stage"] == ResearchStage.CRITICISM:
                models["critic"] = CriticAgent(fl["model"])
            elif fl["stage"] == ResearchStage.EXPLORATION:
                models["explorer"] = ExplorerAgent(fl["model"])
        return models

    def research_topic(self, topic: str, research_config: dict = {}):
        if research_config == {}:
            research_config = {
                "analyst": {"model": "gemma3:12b", "context_window": 4096},
                "critic": {"model": "gemma3:12b", "context_window": 4096},
                "explorer": {"model": "gemma3:12b", "context_window": 4096},
                "synthesizer": {"model": "gemma3:12b", "context_window": 4096},
            }

        analyst = AnalystAgent(research_config["analyst"]["model"])
        critic = CriticAgent(research_config["critic"]["model"])
        explorer = ExplorerAgent(research_config["explorer"]["model"])
        synthesizer = SynthesizerAgent(research_config["synthesizer"]["model"])
        web_topic = self._refine_prompt_for_web(topic, explorer)

        topic_exploration = self._explore(
            topic,
            explorer,
            context_window=research_config["explorer"]["context_window"],
            search_query=web_topic,
        )

        research = []

        for t in topic_exploration:

            wt = self._refine_prompt_for_web(t, explorer)

            analyzer_response = analyst.run(task=t, search_query=wt)
            critic_response = self._critisize(analyzer_response, critic)
            synthesized_response = self._synthesize(
                analyzer_response, critic_response, synthesizer
            )
            next_steps = self._next_step(synthesized_response, explorer)

            research_state = {
                "topic": t,
                "web_query": wt,
                "questions": [],
                "initial_analysis": analyzer_response,
                "critique": critic_response,
                "synthesis": synthesized_response,
                "next_steps": next_steps,
            }
            print(f"Analyzer: {analyzer_response}")
            print(f"CRITIC: {critic_response}")
            print(f"Synthesizer: {synthesized_response}")
            print(f"Next Steps: {next_steps}")
            exit()

    def _refine_prompt_for_web(self, prompt: str, agent):
        instructions = f"Based on the following prompt: '{prompt}', refine the users query for a google search. Return only the refined query. If relevant here is today's date: {dt.datetime.now().date()}"
        return agent.run(task=instructions)

    def _explore(
        self,
        topic: str,
        agent,
        context_window: int,
        search_query: str = None,
        instructions: str = "",
    ):
        prompt = f"Generate 3-4 key research questions about the topic: '{topic}'.\n\nIf relevant, here is the current date: {dt.datetime.now().date()}. Return only the questions. No other commentary."
        response = agent.run(
            task=prompt,
            search_query=search_query,
            context_window=context_window,
            max_tokens=context_window,
        )
        response = [q.strip() for q in response.split("\n") if q.strip()]
        return response

    def _critisize(self, prompt: str, agent):
        insturctions = f"""Critically evaluate the following analysis. Identify any potential biases,
unstated assumptions, or logical fallacies. Is the evidence strong enough
to support the conclusions? Your response should be a bullet point list. Only return the bullet point list. 
--- ANALYSIS ---
{prompt}"""
        response = agent.run(task=insturctions)
        response = [q.strip() for q in response.split("\n") if q.strip()]
        return response

    def _synthesize(self, analysis: str, criticism: str, agent):
        prompt = f"""Create a coherent summary that incorporates the initial analysis and the subsequent critique.
Present a balanced view based on both pieces of information.

--- INITIAL ANALYSIS ---
{analysis}

--- CRITIQUE ---
{criticism}"""
        response = agent.run(task=prompt)
        return response

    def _next_step(self, synthesis: str, agent):
        prompt = f"""
Based on the following research summary and critique, what are the most
important unanswered questions or next steps for a deeper investigation?

--- SUMMARY ---
{synthesis}
"""
        response = agent.run(task=prompt)
        return response


# Example usage and configuration
def create_research_framework_example(topic):
    # Define the starting point of the research.
    initial_topic = "The impact of AI on creative industries"
    recursion_depth = 3
    researcher = ResearchAgent()
    # Run the research process.
    final_research_data = researcher.run_research(
        topic=initial_topic, recursion_depth=recursion_depth
    )

    # """Example of how to set up and use the research framework"""

    # # Model configurations
    # model_configs = {
    #     "primary": {
    #         "model": "gemma3:12b",  # or whatever model you have
    #         "system_prompt": "You are an expert researcher with deep analytical capabilities.",
    #     },
    #     "analyst": {
    #         "model": "gemma3:12b",
    #         "system_prompt": "You are a data analyst focused on evidence-based research and pattern recognition.",
    #     },
    #     "synthesizer": {
    #         "model": "gemma3:12b",
    #         "system_prompt": "You are a synthesis expert who combines multiple sources into coherent insights.",
    #     },
    # }

    # # Create framework
    # framework = DeepResearchFramework(model_configs)

    # # Example research query
    # query = ResearchQuery(
    #     query="What are the implications of artificial intelligence on future job markets?",
    #     context="Focus on both positive and negative impacts, with consideration for different sectors",
    #     expected_depth="deep",
    #     tags=["AI", "employment", "economics", "future"],
    # )

    # # Conduct research
    # results = framework.conduct_research(query, strategy="comprehensive")

    # # Generate report
    # report = framework.generate_research_report(results, format="markdown")

    # return framework, results, report
