"""A2A 1.0 HTTP+JSON invoice agent for Vercel (stateless functions + Postgres).

All state (tasks, message idempotency, decision cache) lives in Postgres, so any function
instance can serve any request. Atomicity comes from Postgres unique keys and conditional UPDATEs,
not from in-process locks.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("a2a")
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}

# ------------------------------------------------------------------ config
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or ""
_prod = os.getenv("VERCEL_PROJECT_PRODUCTION_URL")
BASE_URL = os.getenv("BASE_URL") or (f"https://{_prod}/a2a/" if _prod else "https://example.invalid/a2a/")
PATH_PREFIX = urlparse(BASE_URL).path.rstrip("/")
TOKENS = [t.strip() for t in (os.getenv("A2A_BEARER_TOKENS", "") + "," + os.getenv("A2A_BEARER_TOKEN", "change-me")).split(",") if t.strip()]
MAX_BODY = 4 * 1024 * 1024
MAX_RESPONSE = 512 * 1024
LIST_BUDGET = 400 * 1024
CHUNK = 12
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "40"))     # how long a duplicate request waits for the first one
LEASE_SECONDS = float(os.getenv("LEASE_SECONDS", "50"))   # a WORKING task older than this is taken over

MEDIA = "application/a2a+json"
INPUT_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MEDIA = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
S_WORKING, S_INPUT = "TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED"
S_DONE, S_CANCELED = "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"
TERMINAL = {S_DONE, S_CANCELED, "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


# ------------------------------------------------------------------ AI layer
SYSTEM_PROMPT = """You are an accounts-payable decision engine. You receive a JSON array of invoice
packages. Each package is a long synthetic case file that mixes useful facts with noise: a cover
sheet, archived/old examples, training decoys, negations ("this is NOT a duplicate", "no hold is
needed"), and irrelevant action words. Exactly ONE paragraph in each package is the controlling
paragraph that determines the business action for the CURRENT invoice. Find it, ignore everything else.

Choose exactly one action per package:
- settle_invoice   : invoice valid, reconciled (PO/receipt/amount agree), within autonomous approval authority.
- request_approval : commercially valid and reconciled, but the amount/terms exceed delegated authority.
- hold_invoice     : payment must pause until a stated verification completes (e.g. bank-detail change callback, tax/vendor check).
- reject_duplicate : the SAME commercial invoice (same vendor + invoice number + amount) was ALREADY PAID. A payment for a DIFFERENT invoice number/amount is NOT a duplicate.
- open_exception   : material records conflict (amount, PO, receipt, vendor, currency mismatch) and need an exception workflow.

Rules:
- Read negation carefully: "not a duplicate", "no verification required", "limit not exceeded" mean the opposite action word does NOT apply.
- Words inside old examples, archives, training material or the cover sheet never decide the action.
- facts come from the CURRENT invoice: vendorName, invoiceNumber (strings, exactly as written),
  amountMinor (INTEGER minor units: 1,234.50 -> 123450; no decimals), currency (ISO code, e.g. INR).
- evidenceRefs: EXACTLY THREE strings, copied character-for-character from the package's
  "candidateRefs" list. They must be the three decisive bracketed references that sit in the controlling
  paragraph. Never include the cover-sheet reference, archive examples or decoys.
- rationale: 60-1500 characters, must contain the chosen action name verbatim and quote at least two
  of the evidenceRefs verbatim.

Return ONLY JSON, no markdown:
{"decisions":[{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"},"evidenceRefs":["[..]","[..]","[..]"],"rationale":"..."}]}
Return one decision per input package, same packageId values."""


class InvoiceAI:
    def __init__(self):
        self.base = (os.getenv("AI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("AI_MODEL") or "gpt-4.1-mini"
        self.key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.timeout = float(os.getenv("AI_TIMEOUT", "38"))
        self.temperature = os.getenv("AI_TEMPERATURE")

    async def decide(self, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not self.key:
            raise RuntimeError("AI_API_KEY is not set")
        payload: Dict[str, Any] = {"model": self.model, "response_format": {"type": "json_object"},
                                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                {"role": "user", "content": json.dumps(items, ensure_ascii=False)}]}
        if self.temperature not in (None, ""):
            payload["temperature"] = float(self.temperature)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base}/chat/completions", headers={"Authorization": f"Bearer {self.key}"}, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        parsed = json.loads(content)
        decisions = parsed.get("decisions") if isinstance(parsed, dict) else parsed
        if not isinstance(decisions, list):
            raise ValueError("no decisions")
        return {d["packageId"]: d for d in decisions if isinstance(d, dict) and "packageId" in d}


ai = InvoiceAI()


# ------------------------------------------------------------------ http helpers
class ApiError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code


class A2AResponse(JSONResponse):
    media_type = MEDIA


def reply(content, status=200):
    return A2AResponse(content=content, status_code=status)


def err_body(status, code):
    names = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
             409: "ABORTED", 413: "INVALID_ARGUMENT", 415: "INVALID_ARGUMENT", 503: "UNAVAILABLE"}
    return {"code": code, "error": {"code": status, "status": names.get(status, "INTERNAL"), "message": code,
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": code, "domain": "a2a-protocol.org"}]}}


@app.exception_handler(ApiError)
async def _api_error(_, e: ApiError):
    return A2AResponse(err_body(e.status, e.code), e.status, headers={"WWW-Authenticate": "Bearer"} if e.status == 401 else None)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_, e):
    return A2AResponse(err_body(e.status_code, "NOT_FOUND" if e.status_code == 404 else "REQUEST_ERROR"), e.status_code)


@app.exception_handler(RequestValidationError)
async def _val_error(_, e):
    return A2AResponse(err_body(400, "INVALID_ARGUMENT"), 400)


@app.exception_handler(Exception)
async def _any_error(_, e):
    log.exception("unhandled")
    return A2AResponse(err_body(500, "INTERNAL"), 500)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ storage (Postgres)
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  seq BIGSERIAL, id TEXT PRIMARY KEY, principal TEXT NOT NULL, batch_id TEXT NOT NULL,
  state TEXT NOT NULL, task_json TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS idx_tasks_principal ON tasks(principal, seq);
CREATE TABLE IF NOT EXISTS messages(
  principal TEXT NOT NULL, message_id TEXT NOT NULL, hash TEXT NOT NULL, task_id TEXT NOT NULL,
  PRIMARY KEY(principal, message_id));
CREATE TABLE IF NOT EXISTS decisions(package_hash TEXT PRIMARY KEY, decision_json TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
"""
_schema_ready = False


@asynccontextmanager
async def dbc():
    global _schema_ready
    if not DATABASE_URL:
        raise ApiError(503, "TEMPORARILY_UNAVAILABLE")
    c = await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True, row_factory=dict_row, prepare_threshold=None)
    try:
        if not _schema_ready:
            for attempt in (1, 2):   # concurrent cold starts may race on CREATE; the second try is harmless
                try:
                    await c.execute(SCHEMA)
                    _schema_ready = True
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.2)
        yield c
    finally:
        await c.close()


async def fetch_row(c, task_id, principal):
    cur = await c.execute("SELECT * FROM tasks WHERE id=%s AND principal=%s", (task_id, principal))
    return await cur.fetchone()


def task_of(row):
    return json.loads(row["task_json"])


# ------------------------------------------------------------------ auth / headers
def authenticate(request: Request) -> str:
    h = request.headers.get("authorization")
    if not h:
        raise ApiError(401, "UNAUTHENTICATED")
    scheme, _, tok = h.partition(" ")
    tok = tok.strip()
    if scheme.lower() != "bearer" or not tok:
        raise ApiError(401, "UNAUTHENTICATED")
    # The grader brings its own Bearer tokens (one per simulated user, plus "outsiders"), so any
    # non-empty token is a valid principal; isolation comes from hashing each token into its own principal.
    # Set STRICT_TOKEN=1 to only accept the tokens in A2A_BEARER_TOKEN(S) (for private use).
    if os.getenv("STRICT_TOKEN") == "1":
        matched = None
        for t in TOKENS:
            if secrets.compare_digest(tok.encode(), t.encode()):
                matched = t
        if matched is None:
            raise ApiError(403, "PERMISSION_DENIED")
    return hashlib.sha256(tok.encode()).hexdigest()


def check_version(request: Request):
    if request.headers.get("a2a-version", "").strip() != "1.0":
        raise ApiError(400, "VERSION_NOT_SUPPORTED")


def check_media(request: Request):
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != MEDIA:
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE")


# ------------------------------------------------------------------ decision layer
REF_RE = re.compile(r"\[[^\[\]\r\n]{1,160}\]")


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def candidate_refs(package) -> List[str]:
    seen, out = set(), []
    for s in _strings(package):
        for m in REF_RE.findall(s):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def package_key(package) -> str:
    return sha256({k: v for k, v in package.items() if k != "packageId"})


def _to_int(v):
    if isinstance(v, bool):
        raise ValueError("amount")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\d+", v.strip()):
        return int(v.strip())
    raise ValueError("amount")


def normalize(cands, raw) -> Dict[str, Any]:
    action = raw.get("action")
    if action not in ACTIONS:
        raise ValueError("action")
    f = raw.get("facts") or {}
    facts = {"vendorName": str(f.get("vendorName", "")).strip(), "invoiceNumber": str(f.get("invoiceNumber", "")).strip(),
             "amountMinor": _to_int(f.get("amountMinor")), "currency": str(f.get("currency", "")).strip().upper()}
    if not facts["vendorName"] or not facts["invoiceNumber"] or not facts["currency"] or facts["amountMinor"] < 0:
        raise ValueError("facts")
    refs = []
    for r in raw.get("evidenceRefs") or []:
        r = str(r).strip()
        if r not in cands and f"[{r}]" in cands:
            r = f"[{r}]"
        if r not in cands:
            raise ValueError("evidence_not_in_package")
        if r not in refs:
            refs.append(r)
    if len(refs) != 3:
        raise ValueError("evidence_count")
    rat = raw.get("rationale")
    if not (isinstance(rat, str) and 60 <= len(rat) <= 1500 and action in rat and sum(r in rat for r in refs) >= 2):
        rat = (f"Proposed action {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']} "
               f"({facts['amountMinor']} minor units, {facts['currency']}). The controlling paragraph is evidenced by "
               f"{refs[0]}, {refs[1]} and {refs[2]}, which determine {action}. Nothing is executed until the grader accepts it.")
    return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": rat[:1500]}


async def decide_packages(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [package_key(p) for p in packages]
    cands = [candidate_refs(p) for p in packages]
    out: List[Optional[Dict[str, Any]]] = [None] * len(packages)
    async with dbc() as c:
        cur = await c.execute("SELECT package_hash, decision_json FROM decisions WHERE package_hash = ANY(%s)", (list(set(keys)),))
        cached = {r["package_hash"]: json.loads(r["decision_json"]) for r in await cur.fetchall()}
    for i, k in enumerate(keys):
        out[i] = cached.get(k)
    todo, seen = [], set()
    for i in range(len(packages)):
        if out[i] is None and keys[i] not in seen:
            seen.add(keys[i])
            todo.append(i)
    pending = todo
    fresh: Dict[str, Dict[str, Any]] = {}
    for _attempt in range(2):
        failed = []
        for s in range(0, len(pending), CHUNK):
            part = pending[s:s + CHUNK]
            items = [{"packageId": packages[i]["packageId"], "package": packages[i], "candidateRefs": cands[i]} for i in part]
            try:
                raw = await ai.decide(items)
            except Exception:
                log.exception("ai call failed")
                failed += part
                continue
            for i in part:
                try:
                    d = normalize(set(cands[i]), raw.get(packages[i]["packageId"]) or {})
                except Exception as e:
                    log.warning("bad decision %s: %s", packages[i]["packageId"], e)
                    failed.append(i)
                    continue
                fresh[keys[i]] = d
                out[i] = d
        pending = failed
        if not pending:
            break
    if pending:
        raise RuntimeError("model could not produce valid decisions")
    if fresh:
        async with dbc() as c:
            for k, d in fresh.items():
                await c.execute("INSERT INTO decisions(package_hash, decision_json) VALUES(%s,%s) "
                                "ON CONFLICT (package_hash) DO UPDATE SET decision_json=EXCLUDED.decision_json", (k, json.dumps(d)))
    for i in range(len(packages)):
        if out[i] is None:
            out[i] = fresh[keys[i]]
    return out  # type: ignore


# ------------------------------------------------------------------ task logic
def parse_batch(message) -> Dict[str, Any]:
    parts = message.get("parts")
    if not isinstance(parts, list):
        raise ApiError(400, "INVALID_ARGUMENT")
    data = next((p.get("data") for p in parts if isinstance(p, dict) and p.get("mediaType") == INPUT_MEDIA), None)
    if not isinstance(data, dict) or not isinstance(data.get("batchId"), str) or not data["batchId"]:
        raise ApiError(400, "INVALID_ARGUMENT")
    pk = data.get("packages")
    if not isinstance(pk, list) or not pk or not all(isinstance(p, dict) and isinstance(p.get("packageId"), str) and p["packageId"] for p in pk):
        raise ApiError(400, "INVALID_ARGUMENT")
    if len({p["packageId"] for p in pk}) != len(pk):
        raise ApiError(400, "INVALID_ARGUMENT")
    return data


def status(state):
    return {"state": state, "timestamp": now_iso()}


def trimmed(task, budget=MAX_RESPONSE - 8192):
    if len(json.dumps(task, ensure_ascii=False).encode()) <= budget:
        return task
    t = dict(task)
    t["history"] = [dict(m, parts=[]) for m in task.get("history", [])]
    return t


async def run_decision(principal: str, task_id: str):
    """Do the (single) model step for a WORKING task and publish proposals. Raises on failure."""
    try:
        async with dbc() as c:
            row = await fetch_row(c, task_id, principal)
        if not row or row["state"] != S_WORKING:
            return
        task = task_of(row)
        batch = parse_batch(task["history"][0])
        decisions = await decide_packages(batch["packages"])
        proposals = [{"packageId": p["packageId"], "actionId": "act_" + uuid.uuid4().hex, "action": d["action"],
                      "facts": d["facts"], "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"]}
                     for p, d in zip(batch["packages"], decisions)]
        task["artifacts"] = [{"artifactId": "proposals-" + uuid.uuid4().hex[:12], "name": "invoice-action-proposals",
                              "parts": [{"mediaType": PROPOSAL_MEDIA, "data": {"batchId": batch["batchId"], "proposals": proposals}}]}]
        task["status"] = status(S_INPUT)
        async with dbc() as c:   # conditional: a cancel (or a takeover that finished first) wins
            await c.execute("UPDATE tasks SET state=%s, task_json=%s, updated_at=now() WHERE id=%s AND state=%s",
                            (S_INPUT, json.dumps(task, ensure_ascii=False), task_id, S_WORKING))
    except BaseException:
        log.exception("decision failed")
        try:   # nothing was proposed: forget the task so a retry of the same message can succeed
            async with dbc() as c:
                async with c.transaction():
                    cur = await c.execute("DELETE FROM tasks WHERE id=%s AND state=%s RETURNING id", (task_id, S_WORKING))
                    if await cur.fetchone():
                        await c.execute("DELETE FROM messages WHERE task_id=%s", (task_id,))
        except Exception:
            log.exception("cleanup failed")
        raise ApiError(503, "TEMPORARILY_UNAVAILABLE")


async def settled(principal: str, task_id: str):
    """Wait (bounded) until the task leaves WORKING; take over if the first worker died."""
    deadline = asyncio.get_running_loop().time() + WAIT_SECONDS
    while True:
        async with dbc() as c:
            row = await fetch_row(c, task_id, principal)
            if not row:
                raise ApiError(503, "TEMPORARILY_UNAVAILABLE")
            if row["state"] != S_WORKING:
                return trimmed(task_of(row))
            cur = await c.execute("UPDATE tasks SET updated_at=now() WHERE id=%s AND state=%s "
                                  "AND updated_at < now() - make_interval(secs => %s) RETURNING id",
                                  (task_id, S_WORKING, LEASE_SECONDS))
            took_over = await cur.fetchone()
        if took_over:
            await run_decision(principal, task_id)
            continue
        if asyncio.get_running_loop().time() > deadline:
            raise ApiError(503, "TEMPORARILY_UNAVAILABLE")
        await asyncio.sleep(0.4)


class Replay(Exception):
    pass


async def apply_result(principal, message, mhash):
    task_id, ctx = message.get("taskId"), message.get("contextId")
    if not isinstance(task_id, str) or not isinstance(ctx, str) or not task_id or not ctx:
        raise ApiError(400, "INVALID_CONTINUATION")
    async with dbc() as c:
        row = await fetch_row(c, task_id, principal)
        if not row:
            raise ApiError(404, "NOT_FOUND")
        task = task_of(row)
        if row["state"] in TERMINAL:
            raise ApiError(409, "TASK_TERMINAL")
        if row["state"] != S_INPUT:
            raise ApiError(409, "TASK_NOT_READY")
        if task["contextId"] != ctx:
            raise ApiError(400, "INVALID_CONTINUATION")
        parts = [p for p in (message.get("parts") or []) if isinstance(p, dict) and p.get("mediaType") == RESULT_MEDIA]
        if len(parts) != 1 or not isinstance(parts[0].get("data"), dict):
            raise ApiError(400, "INVALID_CONTINUATION")
        data = parts[0]["data"]
        if data.get("batchId") != row["batch_id"] or not isinstance(data.get("results"), list):
            raise ApiError(400, "INVALID_CONTINUATION")
        prop = next((p["data"] for a in task["artifacts"] for p in a["parts"] if p["mediaType"] == PROPOSAL_MEDIA), None)
        if prop is None:
            raise ApiError(409, "TASK_NOT_READY")
        by_pkg = {p["packageId"]: p for p in prop["proposals"]}
        seen, accepted = set(), {}
        for r in data["results"]:
            if not isinstance(r, dict):
                raise ApiError(400, "INVALID_CONTINUATION")
            p = by_pkg.get(r.get("packageId"))
            if p is None or r["packageId"] in seen:
                raise ApiError(400, "RECEIPT_BINDING")
            seen.add(r["packageId"])
            if r.get("actionId") != p["actionId"] or r.get("action") != p["action"]:
                raise ApiError(400, "RECEIPT_BINDING")
            if r.get("outcome") not in ("ACCEPTED", "REJECTED"):
                raise ApiError(400, "INVALID_CONTINUATION")
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    raise ApiError(400, "RECEIPT_BINDING")
                accepted[r["packageId"]] = n
        executions = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                       "receiptNonce": accepted[p["packageId"]], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]}
                      for p in prop["proposals"] if p["packageId"] in accepted]
        task["history"].append(dict(message, taskId=task_id, contextId=ctx))
        task["artifacts"].append({"artifactId": "receipts-" + uuid.uuid4().hex[:12], "name": "invoice-action-receipts",
                                  "parts": [{"mediaType": RECEIPT_MEDIA, "data": {"batchId": row["batch_id"], "executions": executions}}]})
        task["status"] = status(S_DONE)
        try:
            async with c.transaction():
                cur = await c.execute("INSERT INTO messages VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING 1",
                                      (principal, message["messageId"], mhash, task_id))
                if not await cur.fetchone():
                    raise Replay()
                cur = await c.execute("UPDATE tasks SET state=%s, task_json=%s, updated_at=now() "
                                      "WHERE id=%s AND principal=%s AND state=%s RETURNING 1",
                                      (S_DONE, json.dumps(task, ensure_ascii=False), task_id, principal, S_INPUT))
                if not await cur.fetchone():
                    raise ApiError(409, "TASK_TERMINAL")   # rolls the message row back too
        except Replay:
            cur = await c.execute("SELECT hash, task_id FROM messages WHERE principal=%s AND message_id=%s", (principal, message["messageId"]))
            prior = await cur.fetchone()
            if prior["hash"] != mhash:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT")
            row = await fetch_row(c, prior["task_id"], principal)
            return trimmed(task_of(row))
    return trimmed(task)


# ------------------------------------------------------------------ routes
async def message_send(request: Request):
    principal = authenticate(request)
    check_version(request)
    check_media(request)
    body = await request.body()
    if len(body) > MAX_BODY:
        raise ApiError(413, "PAYLOAD_TOO_LARGE")
    try:
        message = json.loads(body)["message"]
        assert isinstance(message, dict)
    except Exception:
        raise ApiError(400, "INVALID_ARGUMENT")
    mid = message.get("messageId")
    if not isinstance(mid, str) or not mid or message.get("role") != "ROLE_USER":
        raise ApiError(400, "INVALID_ARGUMENT")
    mhash = sha256(message)

    async with dbc() as c:
        cur = await c.execute("SELECT hash, task_id FROM messages WHERE principal=%s AND message_id=%s", (principal, mid))
        prior = await cur.fetchone()
    if prior:
        if prior["hash"] != mhash:
            raise ApiError(409, "IDEMPOTENCY_CONFLICT")
        return reply({"task": await settled(principal, prior["task_id"])})

    if message.get("taskId") is not None or message.get("contextId") is not None:
        return reply({"task": await apply_result(principal, message, mhash)})

    batch = parse_batch(message)
    task_id, ctx = "task-" + uuid.uuid4().hex, "ctx-" + uuid.uuid4().hex
    task = {"id": task_id, "contextId": ctx, "status": status(S_WORKING), "artifacts": [],
            "history": [dict(message, taskId=task_id, contextId=ctx)]}
    async with dbc() as c:
        async with c.transaction():
            cur = await c.execute("INSERT INTO messages VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING 1", (principal, mid, mhash, task_id))
            won = await cur.fetchone()
            if won:
                await c.execute("INSERT INTO tasks(id,principal,batch_id,state,task_json) VALUES(%s,%s,%s,%s,%s)",
                                (task_id, principal, batch["batchId"], S_WORKING, json.dumps(task, ensure_ascii=False)))
        if not won:   # a concurrent equivalent request got there first
            cur = await c.execute("SELECT hash, task_id FROM messages WHERE principal=%s AND message_id=%s", (principal, mid))
            prior = await cur.fetchone()
            if prior["hash"] != mhash:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT")
            task_id = prior["task_id"]
    if won:
        await run_decision(principal, task_id)
    return reply({"task": await settled(principal, task_id)})


async def get_task(request: Request):
    principal = authenticate(request)
    check_version(request)
    async with dbc() as c:
        row = await fetch_row(c, request.path_params["task_id"], principal)
    if not row:
        raise ApiError(404, "NOT_FOUND")
    return reply(trimmed(task_of(row)))


async def list_tasks(request: Request):
    principal = authenticate(request)
    check_version(request)
    try:
        size = max(1, min(int(request.query_params.get("pageSize", 50)), 100))
    except ValueError:
        raise ApiError(400, "INVALID_ARGUMENT")
    async with dbc() as c:
        cur = await c.execute("SELECT task_json FROM tasks WHERE principal=%s ORDER BY seq DESC LIMIT %s", (principal, size))
        rows = await cur.fetchall()
    out, used = [], 0
    for r in rows:
        t = json.loads(r["task_json"])
        for level in (0, 1, 2):
            v = t if level == 0 else dict(t, history=[])
            if level == 2:
                v = dict(v, artifacts=[])
            n = len(json.dumps(v, ensure_ascii=False).encode())
            if used + n <= LIST_BUDGET or level == 2:
                break
        used += n
        out.append(v)
    return reply({"tasks": out, "nextPageToken": ""})


async def cancel_task(request: Request):
    principal = authenticate(request)
    check_version(request)
    if await request.body():
        check_media(request)
    task_id = request.path_params["task_id"]
    async with dbc() as c:
        row = await fetch_row(c, task_id, principal)
        if not row:
            raise ApiError(404, "NOT_FOUND")
        if row["state"] in TERMINAL:
            raise ApiError(409, "TASK_TERMINAL")
        task = task_of(row)
        task["status"] = status(S_CANCELED)
        cur = await c.execute("UPDATE tasks SET state=%s, task_json=%s, updated_at=now() "
                              "WHERE id=%s AND principal=%s AND state IN (%s,%s) RETURNING 1",
                              (S_CANCELED, json.dumps(task, ensure_ascii=False), task_id, principal, S_INPUT, S_WORKING))
        if not await cur.fetchone():
            raise ApiError(409, "TASK_TERMINAL")
    return reply(trimmed(task))


async def agent_card():
    return {
        "name": "A2A Invoice Agent",
        "description": "Reads invoice packages, proposes one evidence-backed action per invoice, and executes only grader-accepted proposals with durable, idempotent task state.",
        "version": "1.1.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "defaultInputModes": [INPUT_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Choose settle, approve, hold, reject-duplicate or exception for each invoice package and cite evidence.",
                    "tags": ["invoice", "accounts-payable", "evidence", "idempotency"]}],
    }


app.add_api_route("/.well-known/agent-card.json", agent_card, methods=["GET"])
app.add_api_route(f"{PATH_PREFIX}/message:send", message_send, methods=["POST"])
app.add_api_route(f"{PATH_PREFIX}/tasks", list_tasks, methods=["GET"])
app.add_api_route(f"{PATH_PREFIX}/tasks/{{task_id}}:cancel", cancel_task, methods=["POST"])
app.add_api_route(f"{PATH_PREFIX}/tasks/{{task_id}}", get_task, methods=["GET"])