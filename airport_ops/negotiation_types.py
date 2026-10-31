"""Records produced by a negotiation session (kept free of coordinator/agent imports to avoid cycles)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .fallback import FallbackResult
from .messages import AgentRole, Condition, MessageType, NegotiationMessage, Participant, Tradeoff
from .plan import RecoveryPlan
from .schemas import HistoricalFlight, Scenario
from .validate import ValidationReport


class FinalStatus(str, Enum):
    AGENT_CONSENSUS = "AGENT_CONSENSUS"        # all three agents accepted one frozen, validated proposal
    FALLBACK_SELECTED = "FALLBACK_SELECTED"    # deterministic fallback chose a plan (NOT agent consensus)
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"      # fallback found nothing valid (see search_status)
    SESSION_INVALIDATED = "SESSION_INVALIDATED"  # scenario version changed mid-session; nothing combined


class FallbackReason(str, Enum):
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    API_ERROR = "API_ERROR"
    RESPONSE_ERROR = "RESPONSE_ERROR"          # unrecoverable: still malformed after the single repair retry
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    NO_UNANIMOUS_AGREEMENT = "NO_UNANIMOUS_AGREEMENT"
    CONSENSUS_REVALIDATION_FAILED = "CONSENSUS_REVALIDATION_FAILED"


class ProposalRecord(BaseModel):
    """Immutable registry entry. `plan_json` is the canonical text; the digest is recomputed from it at consensus."""
    model_config = ConfigDict(frozen=True)
    proposal_id: str
    version_no: int
    proposer: AgentRole
    proposer_id: str = ""
    turn: int
    seq: int
    referenced_proposal_id: Optional[str]
    plan_json: str
    digest: str
    origin: str
    validation: ValidationReport
    metrics: dict

    @property
    def plan(self) -> RecoveryPlan:
        return RecoveryPlan.model_validate_json(self.plan_json)


class ModelAttempt(BaseModel):
    """One model call. Logged separately from messages; a repair retry is a second attempt for the SAME message slot."""
    session_id: str
    turn: int
    role: AgentRole
    agent_id: str = ""
    attempt: int
    kind: str                       # initial | schema_repair
    outcome: str                    # ok | schema_error | timeout | api_error
    latency_ms: int
    mode: str
    model: Optional[str] = None
    credential_mode: Optional[str] = None   # per_agent | shared (which key slot class; never the key)
    error: Optional[str] = None     # sanitized, truncated
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class DecisionRecord(BaseModel):
    """Concise, public record of why an agent's position changed (explanations only, never private reasoning)."""
    session_id: str
    seq: int
    sender: AgentRole
    agent_id: str = ""
    type: MessageType
    proposal_id: str
    previous_proposal_id: Optional[str]
    changed_preference: bool
    reason_codes: list[str]
    explanation: str
    declared_tradeoff: Optional[Tradeoff] = None
    observed_delta: dict = Field(default_factory=dict)     # recomputed by the coordinator: after - before


class ToolCallRecord(BaseModel):
    """A deterministic tool invocation (advisory option menu, plan-spec materialisation). Logged apart from model attempts."""
    session_id: str
    turn: int
    seq: int
    role: AgentRole
    agent_id: str = ""
    tool: str
    summary: dict = Field(default_factory=dict)


class ValidationEvent(BaseModel):
    """One independent validator run (reference / registration / freeze / final_revalidation). Logged apart from messages."""
    session_id: str
    stage: str
    proposal_id: str
    passed: bool
    n_checks: int
    failed_constraints: list[str] = Field(default_factory=list)


class Vote(BaseModel):
    role: AgentRole
    agent_id: str = ""
    seq: Optional[int] = None
    type: Optional[MessageType] = None
    proposal_id: Optional[str] = None
    counted_as_accept: bool = False
    note: str = ""


class NegotiationResult(BaseModel):
    session_id: str
    mode: str
    model_name: Optional[str]
    scenario_id: str
    scenario_version: str
    status: FinalStatus
    participants: list[Participant] = Field(default_factory=list)
    turns_completed: int
    messages: list[NegotiationMessage]
    proposals: list[ProposalRecord]
    attempts: list[ModelAttempt]
    decisions: list[DecisionRecord]
    reference_plan: Optional[RecoveryPlan] = None
    reference_validation: Optional[ValidationReport] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    validations: list[ValidationEvent] = Field(default_factory=list)
    frozen_proposal_id: Optional[str] = None
    freeze_ranking: list[dict] = Field(default_factory=list)
    votes: list[Vote] = Field(default_factory=list)
    final_plan: Optional[RecoveryPlan] = None
    final_validation: Optional[ValidationReport] = None
    fallback: Optional[FallbackResult] = None
    fallback_reason: Optional[FallbackReason] = None
    fallback_detail: Optional[str] = None
    invalidation_detail: Optional[str] = None
    elapsed_s: float = 0.0
    deadline_s: float = 0.0

    @property
    def consensus(self) -> bool:
        return self.status == FinalStatus.AGENT_CONSENSUS

    def proposal(self, pid: str) -> Optional[ProposalRecord]:
        return next((p for p in self.proposals if p.proposal_id == pid), None)


@dataclass
class AgentContext:
    """Everything one agent call may see. Built fresh by the coordinator for every call."""
    role: AgentRole
    turn: int
    seq: int
    session_id: str
    scenario: Scenario
    historical: HistoricalFlight
    messages: list[NegotiationMessage]
    proposals: list[ProposalRecord]
    standing: dict                                 # agent_id -> that agent's most recently stated conditions
    allowed_types: list[MessageType]
    reference: Optional[dict] = None               # initial reference plan view (plan summary + validation evidence)
    frozen: Optional[ProposalRecord] = None        # turn 3 only
    repair_error: Optional[str] = None
    agent: Optional[Participant] = None            # who is speaking (None = legacy: the bare role)
    participants: list = field(default_factory=list)
    require_report: bool = False

    @property
    def who(self) -> Participant:
        return self.agent or Participant(agent_id=self.role.value, role=self.role)
