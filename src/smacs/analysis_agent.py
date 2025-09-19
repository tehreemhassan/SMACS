from typing import List, Dict, Any
from .memory_agent import MemoryAgent
from datetime import datetime

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

class AnalysisAgent:
    def __init__(self, memory: MemoryAgent):
        self.memory = memory

    def analyze(self, items: List[Dict[str,Any]], directive: str, task_id: str) -> Dict[str,Any]:
        start = now_ts()
        results = []
        for rec in items:
            txt = rec["text"].lower()
            keywords = ["efficiency","performance","trade-off","accuracy","scalability","compute","parameters","training"]
            score = sum(txt.count(k) for k in keywords)
            length = len(txt.split())
            results.append({"id": rec.get("id"), "title": rec.get("title"), "score": score, "length": length, "source": rec.get("source")})
        results.sort(key=lambda x: (x["score"], x["length"]), reverse=True)
        summary_lines = [f"Analysis for task {task_id} - directive: {directive}"]
        for i,r in enumerate(results):
            summary_lines.append(f"{i+1}. {r['title']} (score={r['score']}, length={r['length']}, source={r['source']})")
        if not results:
            summary_lines.append("No items to analyze.")
        summary = "\n".join(summary_lines)
        kb_rec = self.memory.store_knowledge(title=f"Analysis:{task_id}", text=summary, source="AnalysisAgent", agent="AnalysisAgent", confidence=0.85)
        self.memory.store_agent_state(task_id, "AnalysisAgent", f"Analyzed {len(results)} items", status="done")
        trace = {
            "task_id": task_id,
            "agent": "AnalysisAgent",
            "directive": directive,
            "ranked": results,
            "summary_kb_id": kb_rec["id"],
            "start": start,
            "end": now_ts(),
            "confidence": 0.85
        }
        return trace
