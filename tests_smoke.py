"""Run: python tests_smoke.py   (uses a mock model; no API key needed)"""
import asyncio, json, os, tempfile
os.environ.update(A2A_BEARER_TOKEN="tok-owner", A2A_BEARER_TOKENS="tok-outsider", BASE_URL="https://h.test/a2a/",
                  DATABASE_URL=os.getenv("TEST_DATABASE_URL", "postgresql://t:t@localhost/t"), LEASE_SECONDS="2", WAIT_SECONDS="15")
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
import httpx, psycopg
with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as _c:
    _c.execute("DROP TABLE IF EXISTS tasks, messages, decisions")
import index as A

calls = {"n": 0}
async def fake_decide(items):
    calls["n"] += 1
    await asyncio.sleep(0.3)
    out = {}
    for it in items:
        refs = it["candidateRefs"][:3]
        out[it["packageId"]] = {"packageId": it["packageId"], "action": "request_approval",
            "facts": {"vendorName": "Acme", "invoiceNumber": "INV-1", "amountMinor": "123450", "currency": "inr"},
            "evidenceRefs": refs, "rationale": "short"}   # forces rationale synthesis
    return out
A.ai.decide = fake_decide

OWN = {"A2A-Version": "1.0", "Content-Type": "application/a2a+json", "Authorization": "Bearer tok-owner"}
OUT = {**OWN, "Authorization": "Bearer tok-outsider"}

def batch(bid, n=12, tag="x"):
    pk = [{"packageId": f"{bid}-p{i}", "text": f"cover [COVER-{i}] old example [ARCH-{i}] decisive [E{tag}{i}-A] [E{tag}{i}-B] [E{tag}{i}-C]"} for i in range(n)]
    return {"batchId": bid, "policyRevision": "r1", "packages": pk}

def msg(mid, b, cfg=None):
    m = {"message": {"messageId": mid, "role": "ROLE_USER", "parts": [{"mediaType": A.INPUT_MEDIA, "data": b}]}}
    if cfg is not None: m["configuration"] = cfg
    return m

async def main():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=A.app), base_url="https://h.test") as c:
        r = await c.get("/.well-known/agent-card.json"); assert r.status_code == 200
        card = r.json(); assert card["supportedInterfaces"][0]["url"] == "https://h.test/a2a/"
        # auth / version / media
        assert (await c.get("/a2a/tasks")).status_code == 401
        assert (await c.get("/a2a/tasks", headers={**OWN, "Authorization": "Bearer nope"})).status_code == 403
        assert (await c.get("/a2a/tasks", headers={**OWN, "A2A-Version": "0.3"})).status_code == 400
        assert (await c.post("/a2a/message:send", headers={**OWN, "Content-Type": "application/json"}, json=msg("z", batch("z", 1)))).status_code == 415
        r = await c.get("/a2a/tasks", headers={"A2A-Version": "1.0", "Authorization": "Bearer tok-owner"})   # GET w/o content-type
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/a2a+json"), r.headers
        # concurrent duplicates (+ reordered keys + different returnImmediately) -> one task, complete, one model call
        b = batch("B1"); m1 = msg("m-1", b, {"returnImmediately": False}); m2 = msg("m-1", json.loads(json.dumps(b)), {"returnImmediately": True})
        m2["message"] = dict(reversed(list(m2["message"].items())))
        rs = await asyncio.gather(*[c.post("/a2a/message:send", headers=OWN, json=m) for m in (m1, m2, m1, m2)])
        tasks = [x.json()["task"] for x in rs]
        assert {t["id"] for t in tasks} == {tasks[0]["id"]} and all(len(t["artifacts"]) == 1 for t in tasks), "dedupe"
        assert calls["n"] == 1, calls
        t = tasks[0]; assert t["status"]["state"] == "TASK_STATE_INPUT_REQUIRED" and t["history"][0]["messageId"] == "m-1"
        art = t["artifacts"][0]["parts"]; assert len(art) == 1 and art[0]["mediaType"] == A.PROPOSAL_MEDIA
        props = art[0]["data"]["proposals"]
        assert len(props) == 12 and len({p["actionId"] for p in props}) == 12 and all(len(p["actionId"]) >= 12 for p in props)
        assert props[0]["facts"] == {"vendorName": "Acme", "invoiceNumber": "INV-1", "amountMinor": 123450, "currency": "INR"}
        assert 60 <= len(props[0]["rationale"]) <= 1500 and props[0]["action"] in props[0]["rationale"]
        # conflict
        bad = msg("m-1", batch("B1-other")); r = await c.post("/a2a/message:send", headers=OWN, json=bad)
        assert r.status_code == 409 and "IDEMPOTENCY_CONFLICT" in r.text
        # same package CONTENT in a new batch/message => no model call, new action ids
        b2 = batch("B2"); 
        for p in b2["packages"]: p["packageId"] = p["packageId"].replace("B2", "B1")   # same content+ids -> cache by content
        r = await c.post("/a2a/message:send", headers=OWN, json=msg("m-2", b2)); t2 = r.json()["task"]
        assert calls["n"] == 1, "cache miss"; assert t2["id"] != t["id"]
        # isolation
        assert (await c.get(f"/a2a/tasks/{t['id']}", headers=OUT)).status_code == 404
        assert (await c.post(f"/a2a/tasks/{t['id']}:cancel", headers=OUT)).status_code == 404
        r = await c.get("/a2a/tasks", headers=OUT); assert r.json()["tasks"] == []
        res_msg = lambda mid, **k: {"message": {"messageId": mid, "taskId": t["id"], "contextId": t["contextId"], "role": "ROLE_USER",
            "parts": [{"mediaType": A.RESULT_MEDIA, "data": {"batchId": "B1", "results": [
                {"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                 "outcome": "ACCEPTED" if i % 2 == 0 else "REJECTED", "receiptNonce": f"nonce-{i}-{k.get('salt','')}"} for i, p in enumerate(props)]}}]}}
        assert (await c.post("/a2a/message:send", headers=OUT, json=res_msg("r-x"))).status_code == 404
        # tampered continuations
        bad = res_msg("r-bad"); bad["message"]["parts"][0]["data"]["results"][0]["actionId"] = "tampered-action-id"
        assert (await c.post("/a2a/message:send", headers=OWN, json=bad)).status_code == 400
        bad = res_msg("r-bad2"); bad["message"]["contextId"] = "ctx-wrong"
        assert (await c.post("/a2a/message:send", headers=OWN, json=bad)).status_code == 400
        bad = res_msg("r-bad3"); bad["message"]["parts"][0]["data"]["batchId"] = "other"
        assert (await c.post("/a2a/message:send", headers=OWN, json=bad)).status_code == 400
        assert (await c.get(f"/a2a/tasks/{t['id']}", headers=OWN)).json()["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        # race: result vs cancel -> exactly one 200 / one 409
        ra, rb = await asyncio.gather(c.post("/a2a/message:send", headers=OWN, json=res_msg("r-1")),
                                      c.post(f"/a2a/tasks/{t['id']}:cancel", headers=OWN))
        assert sorted([ra.status_code, rb.status_code]) == [200, 409], (ra.status_code, rb.status_code)
        final = (await c.get(f"/a2a/tasks/{t['id']}", headers=OWN)).json()
        if ra.status_code == 200:
            assert final["status"]["state"] == "TASK_STATE_COMPLETED" and len(final["artifacts"]) == 2
            ex = final["artifacts"][1]["parts"][0]["data"]["executions"]
            assert len(ex) == 6 and all(e["receiptNonce"].startswith("nonce-") for e in ex) and len(final["history"]) == 2
            # terminal replay: identical continuation returns the same completed task, different one -> 409
            rp = await c.post("/a2a/message:send", headers=OWN, json=res_msg("r-1")); assert rp.status_code == 200 and rp.json()["task"] == final
            assert (await c.post("/a2a/message:send", headers=OWN, json=res_msg("r-1", salt="z"))).status_code == 409
            assert (await c.post("/a2a/message:send", headers=OWN, json=res_msg("r-new"))).status_code == 409
            assert (await c.post(f"/a2a/tasks/{t['id']}:cancel", headers=OWN)).status_code == 409
        else:
            assert final["status"]["state"] == "TASK_STATE_CANCELED" and len(final["artifacts"]) == 1
        # list size stays bounded with many big tasks
        big = batch("BIG", 12); 
        for p in big["packages"]: p["text"] += "y" * 20000
        for i in range(8): await c.post("/a2a/message:send", headers=OWN, json=msg(f"big-{i}", {**big, "batchId": f"BIG{i}"}))
        r = await c.get("/a2a/tasks", headers=OWN); assert len(r.content) < 512 * 1024, len(r.content)
        # trailing-slash must not redirect
        assert (await c.get("/a2a/tasks/", headers=OWN)).status_code == 404
    # ---- serverless-specific: AI failure -> retry works; stuck worker -> takeover
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=A.app), base_url="https://h.test") as c:
        good = A.ai.decide
        async def boom(items): raise RuntimeError("provider down")
        A.ai.decide = boom
        fb = batch("FAIL", 2, "f"); r = await c.post("/a2a/message:send", headers=OWN, json=msg("m-fail", fb))
        assert r.status_code == 503 and "FAIL" not in r.text
        A.ai.decide = good
        r = await c.post("/a2a/message:send", headers=OWN, json=msg("m-fail", fb))
        assert r.status_code == 200 and r.json()["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED", r.text
        # simulate a worker that died mid-decision: WORKING row, lease expired
        sb = batch("STUCK", 2, "s"); m = msg("m-stuck", sb)["message"]
        import uuid
        tid = "task-stuck"; task = {"id": tid, "contextId": "ctx-stuck", "status": {"state": A.S_WORKING, "timestamp": "x"}, "artifacts": [],
                                     "history": [dict(m, taskId=tid, contextId="ctx-stuck")]}
        owner = A.hashlib.sha256(b"tok-owner").hexdigest()
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as pc:
            pc.execute("INSERT INTO messages VALUES(%s,%s,%s,%s)", (owner, "m-stuck", A.sha256(m), tid))
            pc.execute("INSERT INTO tasks(id,principal,batch_id,state,task_json,updated_at) VALUES(%s,%s,%s,%s,%s, now()-interval '5 minutes')",
                       (tid, owner, "STUCK", A.S_WORKING, json.dumps(task)))
        r = await c.post("/a2a/message:send", headers=OWN, json=msg("m-stuck", sb))
        assert r.status_code == 200 and r.json()["task"]["id"] == tid and r.json()["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED", r.text
    print("ALL TESTS PASSED")
asyncio.run(main())
