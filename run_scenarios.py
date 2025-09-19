import os
import json

from src.smacs.memory_agent import MemoryAgent
from src.smacs.research_agent import ResearchAgent
from src.smacs.analysis_agent import AnalysisAgent
from src.smacs.coordinator import Coordinator


def ensure_outputs_dir():
    """Ensure outputs directory exists."""
    os.makedirs("outputs", exist_ok=True)


def save_text_output(filename: str, content: str):
    """Save plain text content to outputs/."""
    with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
        f.write(content)


def run_scenarios(coordinator: Coordinator):
    """Run assessment scenarios and save outputs."""
    ensure_outputs_dir()
    log_file = os.path.join("outputs", "scenario_traces.jsonl")

    scenarios = {
        "simple_query.txt": "What are the main types of neural networks?",
        "complex_query.txt": "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
        "memory_test.txt": "What did we discuss about neural networks earlier?",
        "multi_step.txt": "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
        "collaborative.txt": "Compare two machine-learning approaches and recommend which is better for our use case."
    }

    for filename, query in scenarios.items():
        print(f"\n=== Running scenario: {filename} ===")
        response = coordinator.handle(query)

        # Print final answer
        print("Assistant:", response["final"]["answer"])

        # Print trace summary
        print("\n--- Agent Trace ---")
        for step in response["steps"]:
            stage = step["stage"]
            trace = step["trace"]
            print(f"[{stage.upper()}] by {trace['agent']} (confidence={trace['confidence']})")

        # Save plain text answer
        save_text_output(filename, response["final"]["answer"])

        # Save JSONL trace
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")


def main():
    # Mock KB to simulate research
    mock_kb = [
        {"title": "Deep Learning Basics", "text": "Covers CNNs, RNNs, and transformers.", "source": "mock_kb"},
        {"title": "AI in Healthcare", "text": "Applications include diagnosis and drug discovery.", "source": "mock_kb"},
        {"title": "Scalable Training", "text": "Discusses parallelism and efficiency trade-offs.", "source": "mock_kb"},
    ]

    # Initialize agents
    memory = MemoryAgent()
    research = ResearchAgent(memory, mock_kb)
    analysis = AnalysisAgent(memory)
    coordinator = Coordinator(memory, research, analysis)

    # Run assessment scenarios
    run_scenarios(coordinator)


if __name__ == "__main__":
    main()

