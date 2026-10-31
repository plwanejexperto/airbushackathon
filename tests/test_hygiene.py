"""Repository hygiene + credential safety. The key-scan test uses the developer's real .env when present (values are never printed)."""
import json
import re
from pathlib import Path

import pytest

from airport_ops import llm
from airport_ops.export import export_session
from airport_ops.llm import ModelAPIError, ScriptedClient
from airport_ops.negotiation import run_negotiation
from mocks import MockClient

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["airport_ops", "demo_output"]
SCAN_FILES = ["app.py", "README.md", ".env.example", "requirements.txt", ".gitignore"]


def test_gitignore_protects_secrets_databases_and_caches():
    lines = {l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()}
    for must in (".env", "*.db", "__pycache__/", ".pytest_cache/", "events.db"):
        assert must in lines, must
    assert "!.env.example" in lines
    assert (ROOT / ".env.example").exists()
    for line in (ROOT / ".env.example").read_text().splitlines():            # the example must contain no real values
        if "=" in line and not line.lstrip().startswith("#"):
            assert line.split("=", 1)[1].strip() == "", line


def test_no_database_files_outside_the_evidence_folder():
    dbs = [p for p in ROOT.rglob("*.db") if ".pytest_cache" not in p.parts]
    assert all("demo_output" in p.parts for p in dbs), dbs                    # e.g. no stray events.db in the project root
    assert not (ROOT / "events.db").exists()


def _files():
    for d in SCAN_DIRS:
        yield from (p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for f in SCAN_FILES:
        if (ROOT / f).exists():
            yield ROOT / f


def test_no_real_credential_value_appears_in_any_project_file_or_evidence_db():
    env = {}
    llm.load_env_file(ROOT / ".env", env)
    secrets = [v for v in env.values() if len(v) >= 12]
    if not secrets:
        pytest.skip("no .env with credentials on this machine")
    hits = [str(p.relative_to(ROOT)) for p in _files() if any(s.encode() in p.read_bytes() for s in secrets)]
    assert not hits, f"credential value found in: {hits}"                    # file names only, never the value


def test_no_anthropic_style_key_in_exports_or_evidence():
    pat = re.compile(rb"sk-ant-[A-Za-z0-9_\-]{8,}")
    hits = [str(p.relative_to(ROOT)) for p in _files() if p.suffix in (".json", ".db", ".md", ".py", ".example", "") and pat.search(p.read_bytes())]
    assert not hits, hits


def test_sanitize_redacts_unknown_keys_and_credential_headers():
    s = llm.sanitize("401 {'x-api-key': 'sk-ant-api03-UNKNOWNKEY123456'} Authorization: Bearer abcDEF123456789 "
                     "api_key=hunter2hunter2 sk-ant-oat01-ANOTHERSECRET99", env={})
    for leaked in ("UNKNOWNKEY", "abcDEF123456789", "hunter2hunter2", "ANOTHERSECRET"):
        assert leaked not in s, leaked
    assert s.count("[REDACTED]") >= 3
    assert llm.sanitize("ordinary text, gate GATE-1 at t+22.0", env={}) == "ordinary text, gate GATE-1 at t+22.0"


def test_export_has_no_credential_or_header_fields_and_scrubs_error_text(scenario, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY_ATC", "sk-ant-ENVSECRET-0123456789")
    err = "HTTP 401 request-headers {'x-api-key': 'sk-ant-LEAKYHEADER0123456789', 'authorization': 'Bearer tokentokentoken1'}"
    res = run_negotiation(scenario, MockClient(lambda r: ModelAPIError(err)), log_path=tmp_path / "e.db")
    d = export_session(scenario, res)
    text = llm.sanitize(json.dumps(d, default=str))

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    bad = [k for k in keys(d) if re.search(r"authoriz|x-api-key|api_key|bearer|headers?$|secret|(^|_)token$|access_token|auth_token", k, re.I)]
    assert not bad, bad                                                       # no credential-ish or header field names
    assert not any(s in text for s in ("LEAKYHEADER", "tokentokentoken1", "ENVSECRET"))
    from airport_ops.export import export_json
    assert not any(s in export_json(scenario, res) for s in ("LEAKYHEADER", "tokentokentoken1", "ENVSECRET"))
    assert not any(s in json.dumps([a.model_dump(mode="json") for a in res.attempts]) for s in ("LEAKYHEADER", "tokentokentoken1"))
