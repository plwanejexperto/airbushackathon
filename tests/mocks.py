"""Mock model clients and reply builders for coordinator tests (no network, no scripted policies)."""
import json

from airport_ops.llm import ModelResponse


class MockClient:
    mode = "mock"
    model_name = "mock-model"

    def __init__(self, handler):
        self.handler, self.requests = handler, []

    @property
    def calls(self):
        return len(self.requests)

    def complete(self, req):
        self.requests.append(req)
        out = self.handler(req)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, dict):
            out = json.dumps(out)
        return ModelResponse(text=out, model="mock-model", input_tokens=10, output_tokens=5)


def propose(cand_or_spec, reasons=("ACCEPTABLE",), conditions=(), text="proposal"):
    spec = getattr(cand_or_spec, "spec", cand_or_spec)
    return {"type": "PROPOSE", "reason_codes": list(reasons), "explanation": text, "plan_spec": spec.model_dump(),
            "conditions": list(conditions)}


def counter(cand, ref, reasons=("COST_TRADEOFF",), conditions=(), text="counter", tradeoff=None):
    d = {"type": "COUNTER_PROPOSE", "reason_codes": list(reasons), "explanation": text, "referenced_proposal_id": ref,
         "plan_spec": cand.spec.model_dump(), "conditions": list(conditions)}
    if tradeoff:
        d["tradeoff"] = tradeoff
    return d


def accept(pid, conditions=(), text="ok"):
    return {"type": "ACCEPT", "reason_codes": ["ACCEPTABLE"], "explanation": text, "proposal_id": pid, "conditions": list(conditions)}


def reject(pid, codes=("UNACCEPTABLE",), text="no"):
    return {"type": "REJECT", "reason_codes": list(codes), "explanation": text, "proposal_id": pid}


def frozen_id(req):
    return req.context.frozen.proposal_id


class Script:
    """Handler built from {(turn, ROLE): callable(req) -> reply}. Missing entries raise so tests fail loudly."""

    def __init__(self, table):
        self.table = table

    def __call__(self, req):
        key = (req.turn, req.role)
        if key not in self.table:
            raise AssertionError(f"unexpected call {key}")
        v = self.table[key]
        return v(req) if callable(v) else v


def agree_on(cand):
    """Everyone proposes/accepts one valid candidate through turns 1-2 and accepts the frozen one in turn 3."""
    pid = cand.proposal_id
    return Script({
        (1, "ATC"): propose(cand), (1, "PILOT"): accept(pid), (1, "GROUND"): accept(pid),
        (2, "ATC"): accept(pid), (2, "PILOT"): accept(pid), (2, "GROUND"): accept(pid),
        (3, "ATC"): lambda r: accept(frozen_id(r)), (3, "PILOT"): lambda r: accept(frozen_id(r)),
        (3, "GROUND"): lambda r: accept(frozen_id(r)),
    })
