#!/usr/bin/env python3
"""End-to-end self-check against a running service.

    python3 scripts/selfcheck.py [base-url]

Verifies, over real HTTP:
1. health + two distinct models
2. switching the active model
3. version pinning: requests accepted before a switch keep the old version,
   requests accepted after it use the new one, and no batch mixes versions
3b. staggered acceptance: each email waits out its own submit_after_ms
   relative to the batch submission; input order never blocks intake
3c. per-email deadlines: each deadline clock starts at that email's own
   acceptance instant
4. batching: 16 emails -> batches of at most 4
5. deadlines: a slow email times out, its batch-mates still succeed, and the
   late result never flips the timeout back to success
6. early dispatch when the earliest deadline is near
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SPAM = "Congratulations! You are a winner. Claim your free prize now."
HAM = "Hi team, attached is the quarterly report for our meeting tomorrow."
CRYPTO = "Limited offer! Invest in crypto today and double your money fast."

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def wait_job(client: httpx.Client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.02)
    return state


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=30.0)

    print("1. health & models")
    health = client.get("/healthz").json()
    check(health["status"] == "ok", "service healthy")
    check({"v1", "v2"} <= set(health["models"]), "both v1 and v2 shipped")
    models = client.get("/api/models").json()["models"]
    sizes = {m["version"]: m["vocabulary_size"] for m in models}
    check(sizes["v1"] != sizes["v2"], f"models differ (v1={sizes['v1']}, v2={sizes['v2']} terms)")

    print("2. model switch")
    switched = client.post("/api/model", json={"version": "v2"}).json()
    check(switched["active"] == "v2", "active model switched to v2")
    check(client.get("/healthz").json()["active_model"] == "v2", "health reflects the switch")
    check(client.post("/api/model", json={"version": "nope"}).status_code == 404, "unknown version rejected")

    print("3. version pinning across a live switch")
    client.post("/api/model", json={"version": "v1"})
    job = client.post(
        "/api/jobs",
        json={
            "label": "pin",
            "emails": [
                {"text": SPAM, "deadline_ms": 900},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 260},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 420},
            ],
        },
    ).json()
    time.sleep(0.12)
    switch = client.post("/api/model", json={"version": "v2"}).json()
    check(switch["event"]["from"] == "v1" and switch["event"]["to"] == "v2", "switched v1 -> v2 mid-queue")
    pinned = wait_job(client, job["id"])
    versions = [e["model_version"] for e in pinned["entries"]]
    check(versions[0] == "v1", "email accepted before the switch kept v1")
    check(all(v == "v2" for v in versions[1:]), "emails accepted after the switch used v2")
    batches = client.get("/api/batches?limit=50").json()["batches"]
    for entry in pinned["entries"]:
        batch = next((b for b in batches if b["batch_id"] == entry["batch_id"]), None)
        if batch:
            expected = entry["model_version"]
            check(batch["model_version"] == expected, f"batch {batch['batch_id']} is pure {expected}")

    print("3b. staggered acceptance is per-email and independent of input order")
    client.post("/api/model", json={"version": "v1"})
    reverse = client.post(
        "/api/jobs",
        json={
            "label": "reverse-stagger",
            "emails": [
                {"text": SPAM, "deadline_ms": 1000, "submit_after_ms": 500},
                {"text": SPAM, "deadline_ms": 1000, "submit_after_ms": 0},
            ],
        },
    ).json()
    reverse_state = wait_job(client, reverse["id"])
    front, rear = reverse_state["entries"]
    check(front["index"] == 1 and rear["index"] == 2, "result numbering still follows input order")
    gap = (
        datetime.fromisoformat(front["accepted_at"]) -
        datetime.fromisoformat(rear["accepted_at"])
    ).total_seconds()
    check(0.40 <= gap <= 0.65, f"front email accepted ~500ms after the immediate rear one (gap={gap:.3f}s)")
    check(all(e["status"] == "ok" for e in reverse_state["entries"]), "both staggered emails succeeded")
    # Switching after both were accepted must not retarget either email.
    client.post("/api/model", json={"version": "v2"})
    reread = client.get(f"/api/jobs/{reverse['id']}").json()
    check(all(e["model_version"] == "v1" for e in reread["entries"]), "accepted emails keep their pinned model after a later switch")

    print("3c. each deadline starts at that email's own acceptance")
    # The slow first email keeps the single worker busy until ~T0+525ms.  The
    # second email is accepted at T0+500ms with its own 400ms deadline.  With
    # independent intake it is enqueued immediately and served as soon as the
    # worker frees up; a serial runner would have counted its deadline from T0
    # and forced a pre-dispatch timeout.
    own = client.post(
        "/api/jobs",
        json={
            "label": "own-deadline",
            "emails": [
                {"text": SPAM, "deadline_ms": 900, "simulate_latency_ms": 300},
                {"text": HAM, "deadline_ms": 400, "submit_after_ms": 500},
            ],
        },
    ).json()
    own_state = wait_job(client, own["id"])
    blocker, delayed = own_state["entries"]
    check(blocker["status"] == "ok", "blocking email finished")
    check(delayed["status"] == "ok", "delayed email met the deadline measured from its own acceptance")
    check(delayed["latency_ms"] < 400, f"latency {delayed['latency_ms']}ms counted from acceptance, not batch T0")
    delay_gap = (
        datetime.fromisoformat(delayed["accepted_at"]) -
        datetime.fromisoformat(own_state["created_at"])
    ).total_seconds()
    check(0.40 <= delay_gap <= 0.65, f"delayed email accepted ~500ms after batch submission (gap={delay_gap:.3f}s)")

    print("3d. timeout + late-result discard still work with staggered intake")
    late = client.post(
        "/api/classify",
        json={
            "label": "stagger-late",
            "emails": [
                {"text": SPAM, "deadline_ms": 150, "submit_after_ms": 300,
                 "simulate_latency_ms": 500},
                {"text": HAM, "deadline_ms": 1000, "submit_after_ms": 300},
            ],
        },
    ).json()
    slow, fast = late["entries"]
    check(slow["status"] == "timeout", "staggered slow email timed out on its own deadline")
    check(slow["late_result_discarded"] is True, "its late inference result was discarded")
    check(fast["status"] == "ok", "its batch-mate still succeeded")
    check(slow["batch_id"] == fast["batch_id"], "they shared one batch")

    print("4. batching limits (16 emails)")
    client.post("/api/model", json={"version": "v1"})
    bulk = client.post(
        "/api/jobs",
        json={
            "label": "bulk16",
            "emails": [
                {"text": f"{SPAM} ref {i}", "deadline_ms": 1000}
                for i in range(16)
            ],
        },
    ).json()
    bulk_state = wait_job(client, bulk["id"])
    check(bulk_state["ok"] == 16, f"all 16 classified (ok={bulk_state['ok']})")
    bulk_ids = {e["batch_id"] for e in bulk_state["entries"]}
    check(len(bulk_ids) == 4, f"16 emails formed 4 batches (got {len(bulk_ids)})")
    sizes = [b["size"] for b in client.get("/api/batches?limit=50").json()["batches"] if b["batch_id"] in bulk_ids]
    check(max(sizes) <= 4 and sorted(sizes) == [4, 4, 4, 4], f"batch sizes {sorted(sizes)}")

    print("5. deadline timeout + late result cannot flip it")
    client.post("/api/model", json={"version": "v2"})
    timeout_job = client.post(
        "/api/classify",
        json={
            "label": "batch timeout",
            "emails": [
                {"text": SPAM, "deadline_ms": 100, "simulate_latency_ms": 600},
                {"text": CRYPTO, "deadline_ms": 1000},
            ],
        },
    ).json()
    late = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 100)
    fast = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 1000)
    check(late["status"] == "timeout", "slow email timed out")
    check(late["score"] is None, "timed-out email has no score")
    check(late["late_result_discarded"] is True, "late result explicitly discarded")
    check(fast["status"] == "ok", "batch-mate still succeeded")
    check(late["batch_id"] == fast["batch_id"], "they were in the same batch")
    time.sleep(0.7)
    reread = client.get(f"/api/jobs/{timeout_job['id']}").json()
    again = next(e for e in reread["entries"] if e["deadline_ms"] == 100)
    check(again["status"] == "timeout", "timeout still a timeout after the late thread finished")

    print("6. early dispatch when the earliest deadline is near")
    near = client.post(
        "/api/jobs",
        json={"label": "near", "emails": [{"text": SPAM, "deadline_ms": 150}]},
    ).json()
    near_state = wait_job(client, near["id"])
    entry = near_state["entries"][0]
    check(entry["status"] == "ok", "near-deadline request was dispatched early and scored")
    check(entry["latency_ms"] < 200, f"latency {entry['latency_ms']}ms < 200ms window")

    print("7. interleaved demo")
    demo = client.post("/api/demo/interleaved").json()
    demo_state = wait_job(client, demo["id"], timeout=10)
    old = [e for e in demo_state["entries"] if e["index"] <= 2]
    new = [e for e in demo_state["entries"] if e["index"] > 2]
    check({e["model_version"] for e in old} == {"v1"}, "demo old requests pinned to v1")
    check({e["model_version"] for e in new} == {"v2"}, "demo new requests used v2")
    slow = next(e for e in new if e["simulate_latency_ms"] > 0)
    check(slow["status"] == "timeout", "demo slow email timed out")
    check(client.get("/healthz").json()["active_model"] == "v2", "demo left v2 active")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
