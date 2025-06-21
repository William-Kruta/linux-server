# orchestrator.py
import logging
from modules.llm.researchv2.agents import (
    AnalystAgent,
    SynthesizerAgent,
    CriticAgent,
    ExplorerAgent,
)
from modules.llm.researchv2.tools import search_web, get_page_content

# --- Configuration ---
# Configure logging to display informative messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_research_task(
    topic: str,
    analyst_model: str = "gemma3:12b",
    synthesizer_model: str = "gemma3:12b",
    critic_model: str = "gemma3:12b",
    explorer_model: str = "gemma3:12b",
):
    """
    Orchestrates a team of agents to perform a research task on a given topic.

    The process involves:
    1.  Explorer: Brainstorms initial research questions.
    2.  Analyst: Gathers and analyzes information on the first question.
    3.  Critic: Evaluates the analyst's findings for biases or gaps.
    4.  Synthesizer: Creates a summary report based on the analysis and critique.
    5.  Explorer: Suggests next steps for deeper research.

    Args:
        topic: The high-level research topic.
    """
    print(f"\n--- Starting Research on: '{topic}' ---")

    # --- Initialize Agents ---
    analyst = AnalystAgent(analyst_model)
    synthesizer = SynthesizerAgent(synthesizer_model)
    critic = CriticAgent(critic_model)
    explorer = ExplorerAgent(explorer_model)

    # --- Research State ---
    # This dictionary will hold the findings of each agent
    research_state = {
        "topic": topic,
        "questions": [],
        "initial_analysis": "",
        "critique": "",
        "synthesis": "",
        "next_steps": "",
    }

    # --- Step 1: Explorer Agent - Brainstorm Questions ---
    print("\n[Phase 1/5] 🧭 Explorer Agent is brainstorming research questions...")
    explorer_task = f"Generate 3-4 key research questions about the topic: {topic}"
    questions_str = explorer.run(task=explorer_task)
    research_state["questions"] = [
        q.strip() for q in questions_str.split("\n") if q.strip()
    ]
    print(f"✅ Generated Questions:\n{questions_str}")

    if not research_state["questions"]:
        logging.error("Explorer failed to generate questions. Aborting.")
        return

    # --- Step 2: Analyst Agent - Analyze First Question ---
    first_question = research_state["questions"][0]
    print(
        f"\n[Phase 2/5] 🔬 Analyst Agent is analyzing the first question: '{first_question}'..."
    )
    analyst_task = f"""
Analyze the web search results to answer the following question: '{first_question}'.
Focus on key findings, methodologies, and evidence.
"""
    # The agent's `run` method handles the search
    analysis = analyst.run(task=analyst_task, search_query=first_question)
    research_state["initial_analysis"] = analysis
    print(f"✅ Analysis Complete:\n{analysis[:500]}...")

    # --- Step 3: Critic Agent - Evaluate Analysis ---
    print("\n[Phase 3/5] 🧐 Critic Agent is evaluating the analysis...")
    critic_task = f"""
Critically evaluate the following analysis. Identify any potential biases,
unstated assumptions, or logical fallacies. Is the evidence strong enough
to support the conclusions?

--- ANALYSIS ---
{research_state['initial_analysis']}
"""
    critique = critic.run(task=critic_task)
    research_state["critique"] = critique
    print(f"✅ Critique Complete:\n{critique[:500]}...")

    # --- Step 4: Synthesizer Agent - Create Summary ---
    print("\n[Phase 4/5] 📝 Synthesizer Agent is creating a summary report...")
    synthesizer_task = f"""
Create a coherent summary that incorporates the initial analysis and the subsequent critique.
Present a balanced view based on both pieces of information.

--- INITIAL ANALYSIS ---
{research_state['initial_analysis']}

--- CRITIQUE ---
{research_state['critique']}
"""
    synthesis = synthesizer.run(task=synthesizer_task)
    research_state["synthesis"] = synthesis
    print(f"✅ Synthesis Complete.")

    # --- Step 5: Explorer Agent - Suggest Next Steps ---
    print("\n[Phase 5/5] 🧭 Explorer Agent is suggesting next research steps...")
    explorer_next_steps_task = f"""
Based on the following research summary and critique, what are the most
important unanswered questions or next steps for a deeper investigation?

--- SUMMARY ---
{research_state['synthesis']}
"""
    next_steps = explorer.run(task=explorer_next_steps_task)
    research_state["next_steps"] = next_steps
    print("✅ Next Steps Identified.")

    # --- Final Report ---
    print("\n\n---  النهائية Research Report ---")
    print("=" * 40)
    print(f"Topic: {research_state['topic']}")
    print("=" * 40)
    print("\n## Research Summary\n")
    print(research_state["synthesis"])
    print("\n## Critical Evaluation\n")
    print(research_state["critique"])
    print("\n## Suggested Next Steps\n")
    print(research_state["next_steps"])
    print("\n--- End of Report ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the high-level research topic
    main_topic = "The impact of generative AI on software development productivity"
    run_research_task(main_topic)
