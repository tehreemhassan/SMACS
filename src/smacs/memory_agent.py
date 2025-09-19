from typing import List, Dict, Any, Tuple
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

class MemoryAgent:
    def __init__(self):
        self.conversation_memory: List[Dict[str,Any]] = []
        self.knowledge_base: List[Dict[str,Any]] = []
        self.agent_state: List[Dict[str,Any]] = []
        self._texts: List[str] = []
        self._ids: List[str] = []
        self._vectorizer = TfidfVectorizer()
        self._matrix = None

    def _rebuild_vectors(self):
        if len(self._texts) == 0:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def store_conversation(self, role: str, message: str, metadata: Dict[str,Any]=None):
        record = {"timestamp": now_ts(), "role": role, "message": message, "metadata": metadata or {}}
        self.conversation_memory.append(record)
        return record

    def store_knowledge(self, title: str, text: str, source: str, agent: str, confidence: float=0.8):
        record = {"id": f"kb_{len(self.knowledge_base)+1}", "timestamp": now_ts(), "title": title, "text": text, "source": source, "agent": agent, "confidence": confidence}
        self.knowledge_base.append(record)
        self._ids.append(record["id"]); self._texts.append(title + " " + text)
        self._rebuild_vectors()
        return record

    def store_agent_state(self, task_id: str, agent: str, note: str, status: str="done"):
        record = {"timestamp": now_ts(), "task_id": task_id, "agent": agent, "note": note, "status": status}
        self.agent_state.append(record)
        return record

    def keyword_search_kb(self, keywords: List[str], top_k: int=5) -> List[Dict[str,Any]]:
        kws = [k.lower() for k in keywords]
        hits = []
        for rec in self.knowledge_base:
            txt = (rec["title"] + " " + rec["text"]).lower()
            score = sum(1 for k in kws if k in txt)
            if score>0:
                hits.append((score, rec))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [r for s,r in hits[:top_k]]

    def vector_search_kb(self, query: str, top_k: int=5) -> List[Tuple[Dict[str,Any], float]]:
        if self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            results.append((self.knowledge_base[i], float(sims[i])))
        return results

    def retrieve_conversation(self, topic_keywords: List[str]) -> List[Dict[str,Any]]:
        kws = [k.lower() for k in topic_keywords]
        return [m for m in self.conversation_memory if any(k in m["message"].lower() for k in kws)]
