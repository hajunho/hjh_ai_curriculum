"""
Tool calling and agents: how to hand an LLM a calculator and a document search.
We implement the ReAct loop (Thought -> Action -> Observation, repeated)
with a rule-based mock LLM.
- a calculation question -> the calculator tool
- a policy question      -> the doc_search tool
- a mixed policy + calculation question -> both tools, in order
"""

import pathlib
import re
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import hjh_data
from mock_llm import MockEmbedding, split_sentences


# ---------------------------------------------------------------------------
# Tool 1: the calculator — LLMs often get arithmetic wrong, so we delegate it
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "Error: only digits and the four arithmetic operators are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # computed only after whitelist validation
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool 2: document search — the semantic search from levels 05-07, wrapped
#                           up as a tool
# ---------------------------------------------------------------------------

class DocSearchTool:
    def __init__(self, docs: dict[str, str]):
        self.chunks = []
        for name, doc in docs.items():
            self.chunks += [(name, s) for s in split_sentences(doc)]
        texts = [t for _, t in self.chunks]
        self.emb = MockEmbedding(dim=512).fit(texts)
        self.vectors = self.emb.embed_batch(texts)

    def __call__(self, query: str) -> str:
        sims = self.vectors @ self.emb.embed(query)
        i = int(np.argmax(sims))
        src, text = self.chunks[i]
        return f"{text} (source: {src})"


# ---------------------------------------------------------------------------
# The mock agent brain — in reality the LLM makes these calls from the prompt
# ---------------------------------------------------------------------------

def _numbers(text: str) -> list[int]:
    """Pull integers out of text, tolerating thousands separators
    ('KRW 80,000' -> 80000)."""
    return [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)]


class MockAgentLLM:
    """A mock LLM that decides the 'next action' by rule, looking at the
    observation notes. A real agent is given the tool list and usage examples
    in the prompt, and the model produces this same decision itself (as a
    JSON-shaped tool call)."""

    def decide(self, question: str, notes: dict) -> dict:
        ql = question.lower()
        needs_docs = any(w in ql for w in ("annual leave", "trip", "allowance",
                                           "warranty", "remote", "policy"))
        needs_math = any(w in ql for w in ("how much", "how many", "left",
                                           "total", "calculate")) and \
            re.search(r"\d", question)

        # 1) Policy-related but no document looked up yet -> search first
        if needs_docs and "doc" not in notes:
            key = next(w for w in ("annual leave", "allowance", "trip",
                                   "warranty", "remote", "policy") if w in ql)
            # Choosing a good search query is the agent's job too
            # (a real LLM generates it)
            search_query = {"annual leave": "annual leave days granted",
                            "allowance": "daily allowance for business trips",
                            "trip": "daily allowance for business trips"}.get(key, key + " policy")
            return {"thought": f"I don't know the '{key}' rule. I should look up the documents first.",
                    "action": "doc_search", "input": search_query}

        # 2) Arithmetic needed but not done yet -> gather the numbers, call the calculator
        if needs_math and "calc" not in notes:
            q_nums = _numbers(question)
            if "doc" in notes:
                doc_nums = _numbers(notes["doc"])
                if "left" in ql and doc_nums:                   # remaining = document value - question value
                    expr = f"{doc_nums[-1]} - {q_nums[-1]}"
                elif "KRW" in notes["doc"]:                     # total = document unit price x days in the question
                    per_day = doc_nums[-1]
                    expr = f"{per_day} * {q_nums[-1]}"
                else:
                    expr = " + ".join(map(str, q_nums))
                return {"thought": "I have the baseline number from the document. I'll leave the arithmetic to the calculator.",
                        "action": "calculator", "input": expr}
            expr = re.sub(r"[^0-9+\-*/(). ]", "", question).strip()
            return {"thought": "This is pure arithmetic. I'll use the calculator instead of doing it in my head.",
                    "action": "calculator", "input": expr}

        # 3) No tool left to use -> the final answer
        if not notes:
            return {"thought": "This is small talk; no tool of mine can solve it.",
                    "action": "final",
                    "input": "I cannot answer this question with my tools (calculator, document search)."}
        parts = []
        if "doc" in notes:
            parts.append(f"Evidence: {notes['doc']}")
        if "calc" in notes:
            parts.append(f"Calculated: {notes['calc']}")
        return {"thought": "I have everything I need. Time to assemble the answer.",
                "action": "final", "input": " / ".join(parts)}


# ---------------------------------------------------------------------------
# The ReAct loop — the heart of an agent
# ---------------------------------------------------------------------------

def run_agent(question: str, brain: MockAgentLLM, tools: dict, max_steps: int = 4) -> None:
    print(f"\n  Question: {question}")
    notes: dict[str, str] = {}                 # the notebook of observations (tool results)
    for step in range(1, max_steps + 1):
        decision = brain.decide(question, notes)
        print(f"    [Thought {step}] {decision['thought']}")
        if decision["action"] == "final":
            print(f"    [Answer   ] {decision['input']}")
            return
        result = tools[decision["action"]](decision["input"])
        print(f"    [Action  {step}] {decision['action']}({decision['input']!r})")
        print(f"    [Observe {step}] {result}")
        notes["doc" if decision["action"] == "doc_search" else "calc"] = result
    print("    [Halted] step limit exceeded — the infinite-loop guard fired.")


def main() -> None:
    np.random.seed(42)

    print("=" * 62)
    print("Level 10 | Tool calling and agents — the ReAct loop")
    print("=" * 62)

    print("\n[1] The two tools we hand the agent")
    print("    calculator(expression) : a four-function calculator (no mental arithmetic allowed)")
    print("    doc_search(query)      : semantic search over internal documents (levels 05-07 reused)")

    tools = {"calculator": calculator,
             "doc_search": DocSearchTool(hjh_data.SAMPLE_DOCS)}
    brain = MockAgentLLM()

    print("\n[2] Pure arithmetic — one tool is enough")
    run_agent("What is 127 * 34? Calculate it for me", brain, tools)

    print("\n[3] A policy question plus arithmetic — two tools, in order")
    run_agent("Work out how many days of annual leave I have left. "
              "I used 6 days this year, on the standard grant", brain, tools)

    print("\n[4] Policy lookup plus multiplication — the document's rate x the question's days")
    run_agent("If I take a 4-day overseas business trip, "
              "what is the total daily allowance?", brain, tools)

    print("\n[5] A question outside the tools — saying what it cannot do")
    run_agent("Recommend something for lunch today", brain, tools)

    print("\n[6] Summary: agent = LLM (judgement) + tools (execution) + loop (repetition) + guards")
    print("    - The LLM only decides 'what to do'; the tools do the arithmetic and the search exactly")
    print("    - A Thought/Action/Observation log at every step makes debugging possible")
    print("    - A step limit is the essential guard against infinite loops (runaway cost)")

    # ------------------------------------------------------------------
    # [Reference] This is how tool calling is declared with the real Claude API:
    # tools = [{"name": "calculator",
    #           "description": "A four-function calculator. Takes an expression string and returns the result",
    #           "input_schema": {"type": "object",
    #                            "properties": {"expression": {"type": "string"}},
    #                            "required": ["expression"]}}]
    # response = client.messages.create(model="claude-opus-5", max_tokens=1024,
    #                                   tools=tools, messages=[...])
    # # If response.stop_reason == "tool_use", the model is asking for a tool call.
    # # Run the tool, attach a tool_result, and send it back — the same loop as
    # # today's run_agent.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
