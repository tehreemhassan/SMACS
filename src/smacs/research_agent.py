from typing import List, Dict, Any
from .memory_agent import MemoryAgent
from datetime import datetime

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

class ResearchAgent:
    def __init__(self, memory: MemoryAgent, mock_kb: List[Dict[str,str]]):
        self.memory = memory
        self.mock_kb = mock_kb

    def research(self, query: str, task_id: str, top_n:int=5) -> Dict[str,Any]:
        start = now_ts()
        results = []
        q = query.lower()
        for entry in self.mock_kb:
            score = 0
            txt = (entry["title"] + " " + entry["text"]).lower()
            for w in q.split():
                if w in txt:
                    score += 1
            if score>0:
                results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        hits = [r for s,r in results[:top_n]]
        mem_hits = self.memory.vector_search_kb(query, top_k=3)
        mem_summaries = [{"id": r[0]["id"], "title": r[0]["title"], "sim": r[1], "source": r[0]["source"]} for r in mem_hits]
        saved = []
        for h in hits:
            rec = self.memory.store_knowledge(title=h["title"], text=h["text"], source=h.get("source","mock_kb"), agent="ResearchAgent", confidence=0.9)
            saved.append(rec)
        self.memory.store_agent_state(task_id, "ResearchAgent", f"Found {len(saved)} entries; mem_hits={len(mem_hits)}", status="done")
        trace = {
            "task_id": task_id,
            "agent": "ResearchAgent",
            "query": query,
            "found": [ {"title":s["title"], "source":s.get("source","mock_kb")} for s in hits ],
            "memory_suggestions": mem_summaries,
            "saved": [r["id"] for r in saved],
            "start": start,
            "end": now_ts(),
            "confidence": 0.9
        }
        return trace
