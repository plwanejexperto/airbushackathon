"""Live-client mapping (stubbed SDK, NO network), credential handling, and Streamlit rerun behaviour."""
import json
import os
import sqlite3
from pathlib import Path

import anthropic
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from mocks import MockClient, agree_on
from airport_ops import eventlog, llm
from airport_ops.llm import ModelAPIError, ModelRequest, ModelTimeout, MissingCredentials
from airport_ops.negotiation import run_negotiation
from airport_ops.tools import candidate_pool

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SECRET = "sk-ant-test-SECRETSECRET1234567890"


class _Stub:
    """Stands in for anthropic.Anthropic: records kwargs, returns or raises what the test sets."""
    def __init__(self, result):
        self.result, self.kwargs, self.options = result, None, None
        self.messages = self

    def with_options(self, **kw):
        self.options = kw
        return self

    def create(self, **kw):
        self.kwargs = kw
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _live(monkeypatch, result):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    client = llm.make_client("live")
    client._client = _Stub(result)
    return client


def _req():
    return ModelRequest(role="ATC", system="sys", prompt="{}", context=None, turn=1, timeout_s=12.5)


class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _Resp:
    stop_reason = "end_turn"
    model = "claude-opus-5-5"
    usage = type("U", (), {"input_tokens": 11, "output_tokens": 7})()
    content = [_Block("thinking"), _Block("text", '{"type": "ACCEPT"}')]


def test_live_mode_without_credentials_never_falls_back_to_scripted(monkeypatch):
    for v in llm.CREDENTIAL_VARS:
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(MissingCredentials, match="no scripted fallback"):
        llm.make_client("live")
    assert llm.make_client("scripted").mode == "scripted"      # scripted only when explicitly requested


def test_live_client_call_shape_and_response_mapping(monkeypatch):
    c = _live(monkeypatch, _Resp())
    out = c.complete(_req())
    assert c.mode == "live" and c.model_name == "claude-opus-5-5"          # default model
    assert out.text == '{"type": "ACCEPT"}' and out.input_tokens == 11 and out.output_tokens == 7   # thinking blocks ignored
    s = c._client
    assert s.options == {"timeout": 12.5, "max_retries": 0}                # per-call timeout; coordinator owns retries
    assert s.kwargs["model"] == "claude-opus-5-5" and s.kwargs["system"] == "sys"
    assert "tool_choice" not in s.kwargs and "prefill" not in json.dumps(s.kwargs)   # no forced tool use / prefill
    assert SECRET not in json.dumps(s.kwargs, default=str)


def test_live_client_env_configuration(monkeypatch):
    monkeypatch.setenv("NEGOTIATION_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("NEGOTIATION_CALL_TIMEOUT_S", "45")
    monkeypatch.setenv("NEGOTIATION_DEADLINE_S", "200")
    monkeypatch.setenv("NEGOTIATION_EFFORT", "low")
    c = _live(monkeypatch, _Resp())
    c.complete(_req())
    cfg = llm.LiveConfig.from_env()
    assert (cfg.model, cfg.call_timeout_s, cfg.deadline_s, cfg.effort, cfg.credentials_present) == ("claude-sonnet-5-5", 45, 200, "low", True)
    assert c._client.kwargs["extra_body"] == {"output_config": {"effort": "low"}}


def test_live_client_error_mapping_and_secret_redaction(monkeypatch):
    rq = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    c = _live(monkeypatch, anthropic.APITimeoutError(request=rq))
    with pytest.raises(ModelTimeout):
        c.complete(_req())
    c = _live(monkeypatch, anthropic.RateLimitError(f"slow down key={SECRET}", response=httpx.Response(429, request=rq), body=None))
    with pytest.raises(ModelAPIError) as e:
        c.complete(_req())
    assert "HTTP 429" in str(e.value) and SECRET not in str(e.value)
    c = _live(monkeypatch, anthropic.APIConnectionError(request=rq))
    with pytest.raises(ModelAPIError, match="connection"):
        c.complete(_req())
    refusal = _Resp(); refusal.stop_reason = "refusal"
    c = _live(monkeypatch, refusal)
    with pytest.raises(ModelAPIError, match="refus"):
        c.complete(_req())


def test_credentials_never_reach_the_event_log(scenario, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    db = tmp_path / "e.db"
    res = run_negotiation(scenario, MockClient(lambda r: ModelAPIError(f"HTTP 401: bad key {SECRET}")), log_path=db)
    dump = json.dumps([e for e in eventlog.read_events(None, 1000, path=db)]) + (res.fallback_detail or "") + res.model_dump_json()
    assert SECRET not in dump and "[REDACTED]" in dump
    raw = sqlite3.connect(db).execute("select group_concat(payload) from events").fetchone()[0]
    assert SECRET not in raw


# ------------------------------------------------------------------ Streamlit reruns
def _start(at):
    btn = next(b for b in at.button if b.label == "Start new negotiation")
    btn.click()
    return at.run()


@pytest.fixture()
def app_env(monkeypatch, tmp_path, scenario):
    monkeypatch.setattr(eventlog, "DB_PATH", tmp_path / "app.db")
    cand = candidate_pool(scenario)[0]
    client = MockClient(agree_on(cand))
    monkeypatch.setattr(llm, "make_client", lambda mode, env=None: client)
    return client, tmp_path / "app.db"


def _count(db):
    return sqlite3.connect(db).execute("select count(*) from events").fetchone()[0]


def test_reruns_do_not_repeat_api_calls_or_duplicate_events_and_new_runs_get_new_ids(app_env, scenario):
    client, db = app_env
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert client.calls == 0                                   # loading the page never calls a model
    at = _start(at)
    assert not at.exception and client.calls == 9
    assert any("AGENT_CONSENSUS" in s.value for s in at.success)
    rows = _count(db)
    sid1 = list(at.session_state["neg_runs"])[0]
    for _ in range(3):                                         # several unrelated reruns
        at.checkbox[0].check().run() if not at.checkbox[0].value else at.checkbox[0].uncheck().run()
        at.run()
    assert client.calls == 9 and _count(db) == rows            # no new API calls, no duplicate events
    assert list(at.session_state["neg_runs"]) == [sid1]        # completed state survived the reruns
    at = _start(at)                                            # an intentional new run
    assert client.calls == 18 and len(at.session_state["neg_runs"]) == 2
    assert sid1 in at.session_state["neg_runs"] and at.session_state["neg_latest"][scenario.scenario_version] != sid1
    assert _count(db) > rows


def test_live_mode_without_credentials_blocks_the_run_in_the_ui(app_env, monkeypatch):
    client, _ = app_env
    for v in llm.CREDENTIAL_VARS:
        monkeypatch.delenv(v, raising=False)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="neg_mode").set_value("Live LLM (Claude API)").run()
    start = next(b for b in at.button if b.label == "Start new negotiation")
    assert start.disabled
    assert any("NOT silently fall back" in e.value for e in at.error)
    assert client.calls == 0 and not at.session_state["neg_runs"]


# ------------------------------------------------------------------ per-agent keys
KEYS = {"ATC": "sk-ant-ATC-key-0000000000", "PILOT": "sk-ant-PILOT-key-111111111", "GROUND": "sk-ant-GROUND-key-22222222"}


def _per_agent_env(monkeypatch, only=None):
    for v in llm.CREDENTIAL_VARS:
        monkeypatch.delenv(v, raising=False)
    for r, v in llm.ROLE_KEY_VARS.items():
        if only is None or r in only:
            monkeypatch.setenv(v, KEYS[r])


def test_each_agent_uses_only_its_own_key(monkeypatch):
    _per_agent_env(monkeypatch)
    c = llm.make_client("live")
    assert c.credential_mode == "per_agent" and set(c._by_role) == {"ATC", "PILOT", "GROUND"}
    assert len({id(x) for x in c._by_role.values()}) == 3 and c._client is None
    assert {r: x.api_key for r, x in c._by_role.items()} == KEYS     # ATC client holds the ATC key, etc.
    stubs = {r: _Stub(_Resp()) for r in KEYS}
    c._by_role = stubs
    for r in KEYS:
        c.complete(ModelRequest(role=r, system="s", prompt="{}", context=None, turn=1, timeout_s=5))
    assert all(s.kwargs is not None for s in stubs.values())         # each stub was called by exactly its own role


def test_partial_per_agent_keys_are_refused_not_mixed_or_scripted(monkeypatch):
    _per_agent_env(monkeypatch, only={"ATC", "PILOT"})
    cfg = llm.LiveConfig.from_env()
    assert cfg.credential_mode == "incomplete" and not cfg.credentials_present
    assert cfg.credentials_by_role == {"ATC": True, "PILOT": True, "GROUND": False}
    with pytest.raises(MissingCredentials, match="ANTHROPIC_API_KEY_GROUND"):
        llm.make_client("live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-shared-key-123456789")   # a shared key does not paper over a partial set
    with pytest.raises(MissingCredentials):
        llm.make_client("live")


def test_shared_key_still_works_and_all_keys_are_redacted(monkeypatch, scenario, tmp_path):
    for v in llm.CREDENTIAL_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-shared-key-123456789")
    assert llm.LiveConfig.from_env().credential_mode == "shared" and llm.make_client("live").credential_mode == "shared"
    _per_agent_env(monkeypatch)
    text = " ".join(KEYS.values())
    assert all(k not in llm.sanitize(text) for k in KEYS.values())
    db = tmp_path / "e.db"
    res = run_negotiation(scenario, MockClient(lambda r: ModelAPIError(f"401 for {KEYS[r.role]}")), log_path=db)
    dump = json.dumps(eventlog.read_events(None, 500, path=db)) + res.model_dump_json()
    assert not any(k in dump for k in KEYS.values())


def test_ui_shows_which_agent_keys_are_set_without_values(app_env, monkeypatch):
    _per_agent_env(monkeypatch, only={"ATC"})
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="neg_mode").set_value("Live LLM (Claude API)").run()
    page = " ".join([e.value for e in at.error] + [i.value for i in at.info])
    assert "incomplete" in page and "ATC: set" in page and "GROUND: missing" in page
    assert KEYS["ATC"] not in page
    assert next(b for b in at.button if b.label == "Start new negotiation").disabled


# ------------------------------------------------------------------ .env loading
def test_dotenv_loads_only_allowed_names_and_never_overrides_the_shell(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\n\nANTHROPIC_API_KEY_ATC=\"" + KEYS["ATC"] + "\"\nexport ANTHROPIC_API_KEY_PILOT='" + KEYS["PILOT"] +
                "'\nANTHROPIC_API_KEY_GROUND=" + KEYS["GROUND"] + "  # trailing comment\nNEGOTIATION_MODEL=claude-sonnet-5-5\n"
                "PATH=/evil\nHOME=/evil\nANTHROPIC_EMPTY=\nnot a line\n")
    env = {"NEGOTIATION_MODEL": "from-shell"}
    names = llm.load_env_file(f, env)
    assert sorted(names) == sorted(llm.ROLE_KEY_VARS.values())                 # model was already set -> shell wins
    assert env["NEGOTIATION_MODEL"] == "from-shell" and "PATH" not in env and "HOME" not in env and "ANTHROPIC_EMPTY" not in env
    assert {r: env[v] for r, v in llm.ROLE_KEY_VARS.items()} == KEYS            # quotes, export and comments handled
    cfg = llm.LiveConfig.from_env(env)
    assert cfg.credential_mode == "per_agent" and all(cfg.credentials_by_role.values())
    assert llm.load_env_file(tmp_path / "missing.env", {}) == []


def test_dotenv_keys_are_redacted_too(tmp_path, monkeypatch):
    for v in llm.CREDENTIAL_VARS:
        monkeypatch.delenv(v, raising=False)
    f = tmp_path / ".env"
    f.write_text(f"ANTHROPIC_API_KEY_ATC={KEYS['ATC']}\n")
    monkeypatch.setenv(llm.ENV_FILE_VAR, str(f))
    llm.load_env_file()
    try:
        assert KEYS["ATC"] not in llm.sanitize(f"failed with {KEYS['ATC']}")
    finally:
        monkeypatch.delenv("ANTHROPIC_API_KEY_ATC", raising=False)


def test_app_reads_dotenv_and_enables_live_mode_without_showing_values(app_env, monkeypatch, tmp_path):
    for v in llm.CREDENTIAL_VARS:
        monkeypatch.delenv(v, raising=False)
    f = tmp_path / ".env"
    f.write_text("".join(f"{v}={KEYS[r]}\n" for r, v in llm.ROLE_KEY_VARS.items()))
    monkeypatch.setenv(llm.ENV_FILE_VAR, str(f))
    try:
        at = AppTest.from_file(APP, default_timeout=180).run()
        at.radio(key="neg_mode").set_value("Live LLM (Claude API)").run()
        page = " ".join(c.value for c in at.caption) + " ".join(i.value for i in at.info)
        assert "**per_agent** (ATC: set, PILOT: set, GROUND: set)" in page and "3 variable(s) read from it" in page
        assert not any(k in page for k in KEYS.values())
        assert not next(b for b in at.button if b.label == "Start new negotiation").disabled
    finally:
        for v in llm.ROLE_KEY_VARS.values():
            os.environ.pop(v, None)
