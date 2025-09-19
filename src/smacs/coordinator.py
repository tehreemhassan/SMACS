import time, json
from datetime import datetime
from typing import Dict, Any, List

from src.smacs.research_agent import ResearchAgent
from src.smacs.analysis_agent import AnalysisAgent
from src.smacs.memory_agent import MemoryAgent


def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


class Coordinator:
    def __init__(self, memory: MemoryAgent, research: ResearchAgent, analysis: AnalysisAgent):
        self.memory = memory
        self.research = research
        self.analysis = analysis
        self.trace_log: List[Dict[str, Any]] = []

    # ---------- Utility Methods ----------

    def log_trace(self, entry: dict):
        """Log a structured trace entry (printed + stored)."""
        entry["timestamp"] = now_ts()
        self.trace_log.append(entry)
        print("[TRACE]", json.dumps(entry))

    def complexity_estimate(self, user_query: str) -> str:
        """Heuristic to classify query complexity."""
        q = user_query.lower()
        if any(k in q for k in [
            "compare", "analyze", "trade-off", "efficiency", "recommend",
            "which is better", "summarize", "papers", "methodologies"
        ]):
            return "complex"
        if len(q.split()) > 8:
            return "moderate"
        return "simple"

    def _generate_task_id(self) -> str:
        return f"task_{int(time.time() * 1000)}"

    # ---------- Main Workflow ----------

    def handle(self, user_query: str) -> dict:
        """Handle a user query by coordinating research + analysis."""
        task_id = self._generate_task_id()

        # Store user query in memory
        self.memory.store_conversation("user", user_query, {"task_id": task_id})

        complexity = self.complexity_estimate(user_query)
        self.log_trace({
            "event": "received_query",
            "task_id": task_id,
            "query": user_query,
            "complexity": complexity
        })

        response = {
            "task_id": task_id,
            "query": user_query,
            "steps": [],
            "final": None
        }

        try:
            if complexity == "simple":
                final = self._handle_simple(user_query, task_id, response)
            else:
                final = self._handle_complex(user_query, task_id, response)

            response["final"] = final
            return response

        except Exception as e:
            return self._handle_error(e, task_id, response)

    # ---------- Query Handlers ----------

    def _handle_simple(self, query: str, task_id: str, response: dict) -> dict:
        """Handle simple queries with just research."""
        r = self.research.research(query, task_id, top_n=3)
        self.log_trace({"event": "research_done", "task_id": task_id, "research_trace": r})

        saved_ids = r["saved"]
        kb_items = [next((k for k in self.memory.knowledge_base if k["id"] == sid), None) for sid in saved_ids]
        kb_items = [k for k in kb_items if k]

        synth = "\n".join([f"- {k['title']}: {k['text'][:240]}..." for k in kb_items]) or "No results found."

        final = {"answer": synth, "confidence": r["confidence"]}
        response["steps"].append({"stage": "research", "trace": r})

        self.memory.store_conversation(
            "assistant",
            final["answer"],
            {"task_id": task_id, "confidence": final["confidence"]}
        )
        return final

    def _handle_complex(self, query: str, task_id: str, response: dict) -> dict:
        """Handle moderate/complex queries with research + analysis."""
        r = self.research.research(query, task_id, top_n=6)
        self.log_trace({"event": "research_done", "task_id": task_id, "research_trace": r})

        saved_ids = r["saved"]
        kb_items = [k for k in self.memory.knowledge_base if k["id"] in saved_ids]

        # Add memory-based suggestions
        for m in r["memory_suggestions"]:
            kb_item = next((k for k in self.memory.knowledge_base if k["id"] == m["id"]), None)
            if kb_item and kb_item not in kb_items:
                kb_items.append(kb_item)

        # Fallback if no knowledge items
        if not kb_items:
            kb_items = self.memory.keyword_search_kb(query.split(), top_k=4)

        # Run analysis
        a = self.analysis.analyze(kb_items, directive=query, task_id=task_id)
        self.log_trace({"event": "analysis_done", "task_id": task_id, "analysis_trace": a})

        # Build synthesis text
        final_text_lines = [f"Synthesis for: {query}", "Key findings:"]
        analysis_kb = next((k for k in self.memory.knowledge_base if k["id"] == a["summary_kb_id"]), None)

        if analysis_kb:
            final_text_lines.append(analysis_kb["text"])
        else:
            final_text_lines.append("No analysis summary found.")

        final_text = "\n".join(final_text_lines)
        final = {"answer": final_text, "confidence": min(r["confidence"], a["confidence"])}

        response["steps"].append({"stage": "research", "trace": r})
        response["steps"].append({"stage": "analysis", "trace": a})

        # Store results in memory
        self.memory.store_conversation(
            "assistant",
            final_text,
            {"task_id": task_id, "confidence": final["confidence"]}
        )
        self.memory.store_knowledge(
            title=f"Synthesis:{task_id}",
            text=final_text,
            source="Coordinator",
            agent="Coordinator",
            confidence=final["confidence"]
        )
        return final

    def _handle_error(self, error: Exception, task_id: str, response: dict) -> dict:
        """Handle exceptions gracefully."""
        err_msg = f"Error while processing: {error}"
        self.log_trace({"event": "error", "task_id": task_id, "error": err_msg})

        fallback = {
            "answer": "Sorry, I encountered an error while processing your request.",
            "confidence": 0.2,
            "error": str(error)
        }
        response["final"] = fallback
        self.memory.store_conversation("assistant", fallback["answer"], {"task_id": task_id})
        return response