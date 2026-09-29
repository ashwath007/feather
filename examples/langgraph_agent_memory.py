"""A LangGraph agent with long-term memory in one Feather file.

Run:
    pip install feather-db langgraph
    python examples/langgraph_agent_memory.py

No API key, no server, no container. The agent's memory is `agent_memory.feather`
sitting next to this script — which is the whole point: every other agent-memory
system in this category needs a database or a service running beside your agent.

The one line that matters is passing `store=` to `compile()`. Everything else is
an ordinary LangGraph graph.
"""
import hashlib
import os
import tempfile

import numpy as np
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from feather_db.integrations.langgraph_store import FeatherStore


def embed(text: str) -> np.ndarray:
    """Stand-in embedder so the example runs offline.

    Swap for a real one in production — Feather ships providers for OpenAI,
    Gemini, Voyage, Cohere and Ollama in feather_db.integrations.embedders.
    """
    seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(0, 1, 256).astype(np.float32)
    return v / np.linalg.norm(v)


class State(TypedDict):
    user_id: str
    message: str
    reply: str


def remember(state: State, *, store: BaseStore) -> dict:
    """Write what this turn taught us.

    The namespace tree is the memory model: (tenant, subject, kind). Because
    search takes a PREFIX, ("acme", user) later reaches preferences and episodes
    together, while ("acme", user, "preferences") narrows to one kind.
    """
    ns = ("acme", state["user_id"], "preferences")
    text = state["message"]
    if "prefer" in text.lower() or "rather" in text.lower():
        # A deterministic key means re-learning the same thing updates in place
        # instead of accumulating near-duplicates.
        key = hashlib.sha1(text.lower().encode()).hexdigest()[:12]
        store.put(ns, key, {"kind": "preference", "text": text, "confidence": 0.8})
    return {}


def recall_and_reply(state: State, *, store: BaseStore) -> dict:
    """Read everything known about this user, then answer."""
    hits = store.search(("acme", state["user_id"]), query=state["message"], limit=3)
    if hits:
        known = "; ".join(h.value["text"] for h in hits)
        reply = f"(recalled: {known}) → answering '{state['message']}'"
    else:
        reply = f"(nothing remembered yet) → answering '{state['message']}'"
    return {"reply": reply}


def build(store: BaseStore):
    g = StateGraph(State)
    g.add_node("remember", remember)
    g.add_node("reply", recall_and_reply)
    g.add_edge(START, "remember")
    g.add_edge("remember", "reply")
    g.add_edge("reply", END)
    return g.compile(store=store)          # ← the whole integration


def main() -> None:
    path = os.path.join(tempfile.mkdtemp(), "agent_memory.feather")
    store = FeatherStore(path, dim=256, embed=embed)
    agent = build(store)

    print("── session 1 ───────────────────────────────────────────")
    for msg in [
        "I prefer async written updates over calls",
        "I'd rather see numbers than adjectives",
    ]:
        out = agent.invoke({"user_id": "u_842", "message": msg, "reply": ""})
        print(f"  {msg}\n    {out['reply']}\n")

    # The agent process ends. The file remains.
    store.close()

    print("── session 2, a brand new process would start here ──────")
    store2 = FeatherStore(path, dim=256, embed=embed)
    agent2 = build(store2)
    out = agent2.invoke(
        {"user_id": "u_842", "message": "how should I send you the report?", "reply": ""}
    )
    print(f"  how should I send you the report?\n    {out['reply']}\n")

    print("── what is stored ──────────────────────────────────────")
    for ns in store2.list_namespaces():
        for item in store2.search(ns, limit=10):
            print(f"  {'/'.join(item.namespace)}  {item.key}")
            print(f"      {item.value['text']}")
    print(f"\n  one file, {os.path.getsize(path) // 1024} KB, no server")
    store2.close()


if __name__ == "__main__":
    main()
