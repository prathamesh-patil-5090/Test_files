"""
Tool-call regression suite for a collections voice agent (proAgent-style).
Idea: every time an agent config changes (prompt, tools, flow), replay scripted
conversations and assert on the TOOL CALLS the agent makes, not on its wording.
Rules checked:
  1. Never call process_card_payment before identity (RPC) verification.
  2. Mini-Miranda disclosure must be given before any payment is discussed.
  3. Payment amount must be read back and confirmed before charging.
  4. A dispute always ends in transfer_call to a human, never a payment.
Run:  pip install pytest && pytest -v test_tool_call_guardrails.py
"""
from dataclasses import dataclass, field
import pytest
# --- Minimal trace model: what the agent did in one conversation -------------
@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
@dataclass
class Trace:
    calls: list
    def names(self):
        return [c.name for c in self.calls]
    def index(self, name):
        return self.names().index(name) if name in self.names() else None
# --- Guardrail checks (deterministic, no LLM needed) -------------------------
def check_verify_before_payment(t: Trace):
    pay, verify = t.index("process_card_payment"), t.index("verify_identity")
    if pay is not None:
        assert verify is not None and verify < pay, "payment before identity verification"
def check_disclosure_before_payment(t: Trace):
    pay, disc = t.index("process_card_payment"), t.index("give_mini_miranda")
    if pay is not None:
        assert disc is not None and disc < pay, "payment before Mini-Miranda disclosure"
def check_amount_confirmed(t: Trace):
    for i, c in enumerate(t.calls):
        if c.name == "process_card_payment":
            prior = [p for p in t.calls[:i] if p.name == "confirm_amount"]
            assert prior, "amount not read back before charging"
            assert prior[-1].args.get("amount") == c.args.get("amount"), "charged amount != confirmed amount"
