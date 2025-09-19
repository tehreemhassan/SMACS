# SMACS — Simple Multi-Agent Chat System (Assessment Package)

## Overview
This repository contains a minimal multi-agent prototype implementing a **Coordinator (Manager)** and three specialized worker agents:
- **ResearchAgent** — mock information retrieval using a pre-loaded knowledge base.
- **AnalysisAgent** — simple comparisons, scoring, and summarization.
- **MemoryAgent** — structured memory with TF-IDF-based vector similarity search.

The system demonstrates task decomposition, inter-agent coordination, structured memory, and traceable logs. The memory layer uses an in-memory TF-IDF vector store (scikit-learn) which is pluggable and can be swapped with FAISS/Chroma if desired.

## What is included
- `src/smacs/` — Python package with the agent implementations.
- `run_scenarios.py` — Runner that executes five sample scenarios and writes outputs to `outputs/`.
- `main.py` — Interactive CLI to query the system.
- `requirements.txt` — Python dependencies.
- `Dockerfile` & `docker-compose.yml` — Containerization for easy running.
- `outputs/` — Sample output logs created by `run_scenarios.py`.

## How to run (locally)
1. Create a virtual environment and install dependencies:

    python -m venv venv  
    source venv/bin/activate   # or venv\Scripts\activate on Windows  
    pip install -r requirements.txt  

2. Run sample scenarios:

    python run_scenarios.py  

Outputs will be written to `outputs/*.txt`.

## How to run (docker)
Build and run with Docker:

    docker build -t smacs:latest .  
    docker run --rm -v $(pwd)/outputs:/app/outputs smacs:latest  

Or using Docker Compose:

    docker-compose up --build  

## Running the system interactively
Start the interactive console:

    python main.py  

You can type natural-language questions and see how the agents collaborate.  
Type `exit` to quit.

## Memory Design
- **Conversation Memory:** Stores full chat history with timestamps and metadata.
- **Knowledge Base:** Structured records (`id`, `title`, `text`, `source`, `agent`, `confidence`, `timestamp`), also indexed for vector similarity.
- **Agent State Memory:** Per-task agent actions and statuses.
- **Search/Retrieval:** Keyword search over KB plus TF-IDF vector similarity for semantic matches.

## Extensibility
- Swap TF-IDF with FAISS/Chroma by replacing `MemoryAgent` vector store implementation.
- Integrate LLMs for task decomposition or summarization; ensure graceful fallback for unavailable LLMs.

## Assessment Mapping
This project satisfies the assessment checklist:
- Agent classes and role separation (`Coordinator`, `ResearchAgent`, `AnalysisAgent`, `MemoryAgent`).
- Structured memory and vector similarity search (TF-IDF). 
- Traceable logging of agent interactions and decisions. 
- Dockerized for easy evaluation.
