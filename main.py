import os
import json

from src.smacs.memory_agent import MemoryAgent
from src.smacs.research_agent import ResearchAgent
from src.smacs.analysis_agent import AnalysisAgent
from src.smacs.coordinator import Coordinator


def ensure_outputs_dir():
    """Ensure outputs directory exists."""
    os.makedirs("outputs", exist_ok=True)


def run_cli(coordinator: Coordinator):
    """Interactive CLI loop."""
    print("🔹 Interactive Research Assistant")
    print("Type your query (or 'exit' to quit).")

    ensure_outputs_dir()
    log_file = os.path.join("outputs", "agent_traces.jsonl")

    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("👋 Exiting. Goodbye!")
                break

            response = coordinator.handle(query)

            # Print answer
            print("\nAssistant:", response["final"]["answer"])

            # Print trace summary
            print("\n--- Agent Trace ---")
            for step in response["steps"]:
                stage = step["stage"]
                trace = step["trace"]
                print(f"[{stage.upper()}] by {trace['agent']} (confidence={trace['confidence']})")

            # Save full JSONL trace
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting. Goodbye!")
            break


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

    # Run CLI
    run_cli(coordinator)


if __name__ == "__main__":
    main()