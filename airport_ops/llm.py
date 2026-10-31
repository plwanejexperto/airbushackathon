"""Model-client layer. Three interchangeable clients behind one tiny interface:

  AnthropicClient - LIVE: real Claude API calls (Anthropic SDK). Needs credentials from the environment.
  ScriptedClient  - SCRIPTED DEMO: deterministic policies, no network, no model. Always labelled as scripted.
  (tests supply their own mock clients implementing the same `complete` method)

Environment (set in the shell or in a project-root .env file, see load_env_file; nothing is ever written to SQLite or logs):
  ANTHROPIC_API_KEY_ATC / _PILOT / _GROUND   one key per agent (all three must be set; each agent's calls use only its key)
  ANTHROPIC_API_KEY | ANTHROPIC_AUTH_TOKEN   ONE shared key for all agents (used only when no per-agent key is set)
  ANTHROPIC_BASE_URL                         optional, honoured by the SDK
  NEGOTIATION_MODEL                          default claude-opus-5-5
  NEGOTIATION_CALL_TIMEOUT_S                 per-call timeout, default 90
  NEGOTIATION_DEADLINE_S                     total negotiation deadline, default 600
  NEGOTIATION_MAX_TOKENS                     default 8000
  NEGOTIATION_EFFORT                         optional output_config.effort (low|medium|high|xhigh|max)

A live run NEVER silently degrades to scripted mode: missing credentials raise MissingCredentials.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

ROLE_KEY_VARS = {"ATC": "ANTHROPIC_API_KEY_ATC", "PILOT": "ANTHROPIC_API_KEY_PILOT", "GROUND": "ANTHROPIC_API_KEY_GROUND"}
CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") + tuple(ROLE_KEY_VARS.values())
DEFAULT_MODEL = "claude-opus-5-5"


ENV_FILE_VAR = "NEGOTIATION_ENV_FILE"   # optional override of the .env location (tests point it at a missing file)
_ALLOWED_PREFIXES = ("ANTHROPIC_", "NEGOTIATION_")
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
LOADED_FROM_FILE: list[str] = []   # names (never values) this process took from the .env file; survives Streamlit reruns


def load_env_file(path: Optional[Path] = None, environ: Optional[dict] = None) -> list[str]:
    """Load KEY=VALUE lines from the project's .env into the process environment.

    Only ANTHROPIC_* / NEGOTIATION_* names are read; anything already set in the real environment wins (so an exported
    variable overrides the file); the file is never written and values are never logged. Returns the NAMES loaded.
    Accepts `KEY=value`, `export KEY=value`, single/double quotes, `#` comments and blank lines.
    """
    environ = os.environ if environ is None else environ
    path = Path(path or os.environ.get(ENV_FILE_VAR) or DEFAULT_ENV_FILE)
    if not path.is_file():
        return []
    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if name.startswith(_ALLOWED_PREFIXES) and value and name not in environ:
            environ[name] = value
            loaded.append(name)
            if environ is os.environ and name not in LOADED_FROM_FILE:
                LOADED_FROM_FILE.append(name)
    return loaded


class ModelTimeout(Exception):
    """The model call exceeded its timeout."""


class ModelAPIError(Exception):
    """Any non-timeout API failure (connection, HTTP status, refusal)."""


class MissingCredentials(RuntimeError):
    """Live mode requested but credentials / SDK are unavailable. Never converted into scripted mode."""


def sanitize(text: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Remove any credential value from text before it is logged or displayed."""
    env = os.environ if env is None else env
    for name in CREDENTIAL_VARS:
        v = env.get(name)
        if v and len(v) >= 8:
            text = text.replace(v, "[REDACTED]")
    return text


@dataclass
class ModelRequest:
    role: str
    system: str
    prompt: str                  # JSON text sent to a live model
    context: Any                 # AgentContext: used by scripted/mock clients, never sent to a live model
    turn: int
    timeout_s: float
    attempt: int = 1             # 1 = first try, 2 = the single schema-repair retry
    repair_error: Optional[str] = None


@dataclass
class ModelResponse:
    text: str
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class AgentClient(Protocol):
    mode: str                    # "live" | "scripted" | "mock"
    model_name: Optional[str]

    def complete(self, req: ModelRequest) -> ModelResponse: ...


@dataclass
class LiveConfig:
    model: str = DEFAULT_MODEL
    call_timeout_s: float = 90.0
    deadline_s: float = 600.0
    max_tokens: int = 8000
    effort: Optional[str] = None
    credentials_present: bool = False
    credential_mode: str = "missing"            # per_agent | shared | incomplete | missing
    credentials_by_role: dict = field(default_factory=dict)  # role -> bool (presence only, never values)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "LiveConfig":
        env = os.environ if env is None else env
        by_role = {r: bool(env.get(v)) for r, v in ROLE_KEY_VARS.items()}
        shared = bool(env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"))
        if all(by_role.values()):
            mode = "per_agent"
        elif any(by_role.values()):
            mode = "incomplete"   # mixing per-agent and shared keys is refused: it would hide which agent uses which key
        else:
            mode = "shared" if shared else "missing"
            by_role = {r: shared for r in ROLE_KEY_VARS}
        return cls(model=env.get("NEGOTIATION_MODEL") or DEFAULT_MODEL,
                   call_timeout_s=float(env.get("NEGOTIATION_CALL_TIMEOUT_S") or 90),
                   deadline_s=float(env.get("NEGOTIATION_DEADLINE_S") or 600),
                   max_tokens=int(env.get("NEGOTIATION_MAX_TOKENS") or 8000),
                   effort=env.get("NEGOTIATION_EFFORT") or None,
                   credentials_present=mode in ("per_agent", "shared"), credential_mode=mode, credentials_by_role=by_role)


class AnthropicClient:
    mode = "live"
    credential_mode = "shared"

    def __init__(self, cfg: LiveConfig, env: Optional[Mapping[str, str]] = None):
        env = os.environ if env is None else env
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - environment dependent
            raise MissingCredentials("the 'anthropic' package is not installed (pip install anthropic)") from e
        self._sdk = anthropic
        # credentials/base_url come from the environment only. per_agent: one SDK client (one key) per role.
        self._by_role = ({r: anthropic.Anthropic(api_key=env[v]) for r, v in ROLE_KEY_VARS.items()}
                         if cfg.credential_mode == "per_agent" else {})
        self._client = None if self._by_role else anthropic.Anthropic()
        self.cfg, self.model_name = cfg, cfg.model
        self.credential_mode = cfg.credential_mode

    def complete(self, req: ModelRequest) -> ModelResponse:
        a = self._sdk
        kwargs: dict = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens, system=req.system,
                            messages=[{"role": "user", "content": req.prompt}])
        if self.cfg.effort:
            kwargs["extra_body"] = {"output_config": {"effort": self.cfg.effort}}
        try:
            # max_retries=0: the coordinator owns retry policy (at most one schema-repair retry) and the deadline
            resp = self._by_role.get(req.role, self._client).with_options(timeout=req.timeout_s, max_retries=0).messages.create(**kwargs)
        except a.APITimeoutError as e:
            raise ModelTimeout(sanitize(str(e))) from e
        except a.APIStatusError as e:
            raise ModelAPIError(sanitize(f"HTTP {e.status_code}: {getattr(e, 'message', e)}")) from e
        except a.APIConnectionError as e:
            raise ModelAPIError(sanitize(f"connection error: {e}")) from e
        if resp.stop_reason == "refusal":
            raise ModelAPIError("model refused the request (stop_reason=refusal)")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = getattr(resp, "usage", None)
        return ModelResponse(text=text, model=getattr(resp, "model", self.cfg.model),
                             input_tokens=getattr(u, "input_tokens", None), output_tokens=getattr(u, "output_tokens", None))


class ScriptedClient:
    """Deterministic stand-in that runs each role's scripted policy. NOT a language model."""
    mode = "scripted"
    model_name = "scripted-policy"

    def __init__(self):
        self.calls = 0

    def complete(self, req: ModelRequest) -> ModelResponse:
        from .agents import scripted_response
        self.calls += 1
        return ModelResponse(text=scripted_response(req.context), model=self.model_name)


def make_client(mode: str, env: Optional[Mapping[str, str]] = None) -> AgentClient:
    """Factory used by the app. `live` raises MissingCredentials instead of falling back to scripted."""
    if mode == "scripted":
        return ScriptedClient()
    if mode != "live":
        raise ValueError(f"unknown mode {mode!r}")
    cfg = LiveConfig.from_env(env)
    if cfg.credential_mode == "incomplete":
        missing = [v for r, v in ROLE_KEY_VARS.items() if not cfg.credentials_by_role[r]]
        raise MissingCredentials(f"per-agent keys are partly set; also set {', '.join(missing)} (or unset the others "
                                 "and use one shared ANTHROPIC_API_KEY). No scripted fallback is used")
    if not cfg.credentials_present:
        raise MissingCredentials("live mode needs ANTHROPIC_API_KEY_ATC/_PILOT/_GROUND (one per agent) or one shared "
                                 "ANTHROPIC_API_KEY in the environment; no scripted fallback is used")
    return AnthropicClient(cfg, env)
