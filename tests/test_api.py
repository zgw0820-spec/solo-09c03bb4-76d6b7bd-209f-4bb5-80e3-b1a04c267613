"""Gateway-level tests: model switching while requests are queued."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import main as main_module  # noqa: E402


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _wait_terminal(client, job_id: str, *, ticks: int = 400) -> dict:
    import time

    state = {}
    for _ in range(ticks):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed", "cancelled"):
            return state
        time.sleep(0.01)
    return state


@pytest.fixture()
def client():
    # Reset process-global state per test.
    main_module.jobs.clear()
    main_module.switches.clear()
    main_module.active_version = main_module.registry.default_version
    main_module.worker.metrics.update(
        {key: 0 for key in main_module.worker.metrics}
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_and_models(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert {"v1", "v2"} <= set(health["models"])

    models = client.get("/api/models").json()
    assert models["active"] in ("v1", "v2")
    assert len(models["models"]) == 2
    v1, v2 = {m["version"]: m for m in models["models"]}.values()
    assert v1["vocabulary_size"] != v2["vocabulary_size"]


def test_switch_model_endpoint(client):
    r = client.post("/api/model", json={"version": "v2"})
    assert r.status_code == 200
    assert r.json()["active"] == "v2"
    assert client.get("/healthz").json()["active_model"] == "v2"

    bad = client.post("/api/model", json={"version": "v9"})
    assert bad.status_code == 404


def test_switch_does_not_retarget_queued_requests(client):
    """Old queued emails keep v1; emails accepted after the switch use v2."""
    # Occupy the single worker with a long-deadline job so later submissions
    # wait in the queue.
    client.post("/api/model", json={"version": "v1"})
    entry = {
        "text": "Limited offer! Invest in crypto today and double your money fast.",
        "deadline_ms": 900,
    }
    job = client.post(
        "/api/jobs",
        json={"label": "pinned", "emails": [entry, dict(entry, deadline_ms=800)]},
    ).json()
    # Let intake accept the first email (pinning v1) before switching.
    import time

    watched = None
    for _ in range(100):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["entries"][0]["model_version"]:
            break
        time.sleep(0.01)
    assert watched["entries"][0]["model_version"] == "v1"

    client.post("/api/model", json={"version": "v2"})

    for _ in range(300):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["status"] == "completed":
            break
        time.sleep(0.01)

    assert watched["status"] == "completed"
    assert all(e["model_version"] == "v1" for e in watched["entries"])
    assert all(e["status"] in ("ok", "timeout") for e in watched["entries"])


def test_switch_between_jobs_uses_new_version(client):
    client.post("/api/model", json={"version": "v1"})
    first = client.post(
        "/api/jobs",
        json={"label": "a", "emails": [{"text": "free prize winner", "deadline_ms": 400}]},
    ).json()
    client.post("/api/model", json={"version": "v2"})
    second = client.post(
        "/api/jobs",
        json={"label": "b", "emails": [{"text": "free prize winner", "deadline_ms": 400}]},
    ).json()

    for job in (first, second):
        for _ in range(300):
            state = client.get(f"/api/jobs/{job['id']}").json()
            if state["status"] == "completed":
                break
            import time

            time.sleep(0.01)
        else:
            pytest.fail("job did not complete")
    a = client.get(f"/api/jobs/{first['id']}").json()
    b = client.get(f"/api/jobs/{second['id']}").json()
    assert a["entries"][0]["model_version"] == "v1"
    assert b["entries"][0]["model_version"] == "v2"


def test_submission_limits(client):
    too_many = {
        "label": "big",
        "emails": [{"text": f"mail {i}", "deadline_ms": 300} for i in range(17)],
    }
    assert client.post("/api/jobs", json=too_many).status_code == 422
    bad_deadline = {
        "label": "bad",
        "emails": [{"text": "x", "deadline_ms": 99}],
    }
    assert client.post("/api/jobs", json=bad_deadline).status_code == 422
    ok = {
        "label": "max",
        "emails": [{"text": f"mail {i}", "deadline_ms": 800} for i in range(16)],
    }
    response = client.post("/api/jobs", json=ok)
    assert response.status_code == 202
    assert response.json()["size"] == 16


def test_timeout_never_becomes_success(client):
    result = client.post(
        "/api/classify",
        json={
            "label": "late",
            "emails": [
                {
                    "text": "free spam winner",
                    "deadline_ms": 100,
                    "simulate_latency_ms": 600,
                },
                {"text": "free spam winner", "deadline_ms": 1000},
            ],
        },
    ).json()
    entries = result["entries"]
    late = next(e for e in entries if e["deadline_ms"] == 100)
    fast = next(e for e in entries if e["deadline_ms"] == 1000)
    assert late["status"] == "timeout"
    assert late["score"] is None
    assert late["late_result_discarded"] is True
    assert fast["status"] == "ok"
    assert late["batch_id"] == fast["batch_id"]


def test_batches_never_mix_versions_through_gateway(client):
    client.post("/api/model", json={"version": "v1"})
    client.post(
        "/api/jobs",
        json={
            "label": "v1-batch",
            "emails": [{"text": f"mail {i} free", "deadline_ms": 800} for i in range(4)],
        },
    )
    client.post("/api/model", json={"version": "v2"})
    client.post(
        "/api/jobs",
        json={
            "label": "v2-batch",
            "emails": [{"text": f"mail {i} winner", "deadline_ms": 800} for i in range(4)],
        },
    )
    import time

    time.sleep(0.9)
    batches = client.get("/api/batches").json()["batches"]
    assert batches
    for batch in batches:
        assert batch["size"] <= 4
        assert batch["model_version"] in ("v1", "v2")
    assert {b["model_version"] for b in batches} >= {"v1", "v2"}


def test_demo_interleaved(client):
    result = client.post("/api/demo/interleaved").json()
    state = _wait_terminal(client, result["id"])
    old = [e for e in state["entries"] if e["index"] <= 2]
    new = [e for e in state["entries"] if e["index"] > 2]
    assert {e["model_version"] for e in old} == {"v1"}
    assert {e["model_version"] for e in new} == {"v2"}
    # The artificially slow new-model email must time out but not take down
    # its batch-mate.
    slow = next(e for e in new if e["simulate_latency_ms"] > 0)
    assert slow["status"] == "timeout"
    assert slow["score"] is None
    assert client.get("/healthz").json()["active_model"] == "v2"
    assert any(ev["type"] == "model_switched" for ev in state["events"])


def test_staggered_emails_are_accepted_independently_of_input_order(client):
    """Front email delayed 500 ms must not hold back an immediate rear email."""
    job = client.post(
        "/api/jobs",
        json={
            "label": "stagger-order",
            "emails": [
                {"text": "front email free prize", "deadline_ms": 900,
                 "submit_after_ms": 500},
                {"text": "rear email free prize", "deadline_ms": 900,
                 "submit_after_ms": 0},
            ],
        },
    ).json()
    t0 = _parse(job["created_at"])

    # The rear (index 2) email must be accepted almost immediately even
    # though it sits behind the 500 ms email in input order.
    rear_accepted = None
    for _ in range(200):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        rear = watched["entries"][1]
        if rear["accepted_at"]:
            rear_accepted = _parse(rear["accepted_at"])
            front = watched["entries"][0]
            assert front["accepted_at"] is None, "front email must still be waiting"
            assert front["status"] == "waiting"
            assert rear["status"] in ("queued", "ok", "timeout")
            break
        import time

        time.sleep(0.01)
    assert rear_accepted is not None
    assert 0 <= (rear_accepted - t0).total_seconds() < 0.25

    state = _wait_terminal(client, job["id"])
    assert state["status"] == "completed"
    front, rear = state["entries"]
    front_at, rear_at = _parse(front["accepted_at"]), _parse(rear["accepted_at"])
    # Independent scheduling: ~500 ms apart, with the rear arriving first.
    delta = (front_at - rear_at).total_seconds()
    assert 0.40 <= delta <= 0.65, delta
    # Result numbering still follows input order.
    assert front["index"] == 1 and rear["index"] == 2
    assert all(e["status"] == "ok" for e in state["entries"])


def test_switch_interleaved_with_stagger_pins_version_at_acceptance(client):
    """Emails accepted before/after a switch pin different versions.

    The first email's stagger (200 ms) is longer than the switch delay
    (~80 ms), while the second email is accepted immediately.  A serial
    runner would sleep through the first email's stagger before accepting
    either one, so both would end up on the post-switch version.
    """
    client.post("/api/model", json={"version": "v1"})
    job = client.post(
        "/api/jobs",
        json={
            "label": "stagger-switch",
            "emails": [
                # Listed first but accepted only after the switch below.
                {"text": "late free prize", "deadline_ms": 1000,
                 "submit_after_ms": 200},
                # Listed second but accepted immediately -> v1.
                {"text": "early free prize", "deadline_ms": 1000,
                 "submit_after_ms": 0},
            ],
        },
    ).json()

    import time

    for _ in range(100):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["entries"][1]["accepted_at"]:
            break
        time.sleep(0.01)
    assert watched["entries"][1]["model_version"] == "v1"

    switched = client.post("/api/model", json={"version": "v2"}).json()
    assert switched["active"] == "v2"

    state = _wait_terminal(client, job["id"])
    assert state["status"] == "completed"
    late, early = state["entries"]
    assert early["model_version"] == "v1"
    assert late["model_version"] == "v2"
    # Both keep their pinned version forever: a later switch cannot retarget.
    client.post("/api/model", json={"version": "v1"})
    reread = client.get(f"/api/jobs/{job['id']}").json()
    assert [e["model_version"] for e in reread["entries"]] == ["v2", "v1"]
    assert late["accepted_at"] > early["accepted_at"]


def test_each_deadline_is_measured_from_its_own_acceptance(client):
    """Each email's deadline clock starts at its own acceptance instant.

    A slow first email occupies the single worker until ~T0+525 ms.  The
    second email is only accepted at T0+500 ms with a 400 ms deadline
    (expires ~T0+900 ms).  Because intake — and therefore the worker-side
    deadline timer — starts at that email's own acceptance, it is enqueued
    immediately and, once the worker frees up, dispatched and scored inside
    its own window.  The result's latency is measured from acceptance too,
    not from the batch submission T0.
    """
    job = client.post(
        "/api/jobs",
        json={
            "label": "own-deadline",
            "emails": [
                {"text": "blocking slow free winner", "deadline_ms": 900,
                 "submit_after_ms": 0, "simulate_latency_ms": 300},
                {"text": "delayed free winner", "deadline_ms": 400,
                 "submit_after_ms": 500},
            ],
        },
    ).json()
    state = _wait_terminal(client, job["id"])
    assert state["status"] == "completed"
    blocker, delayed = state["entries"]
    assert blocker["status"] == "ok"
    # Independent clock: accepted at ~T0+500 ms, deadline runs to ~T0+900 ms,
    # worker is free by ~T0+525 ms -> this email still makes it.
    assert delayed["status"] == "ok", delayed
    t0 = _parse(job["created_at"])
    accepted = _parse(delayed["accepted_at"])
    completed = _parse(delayed["completed_at"])
    assert 0.40 <= (accepted - t0).total_seconds() <= 0.65
    # End-to-end latency is measured from this email's own acceptance, not T0.
    assert delayed["latency_ms"] < 400


def test_late_result_rule_still_holds_with_stagger(client):
    """Timeout + late-result discard semantics survive staggered intake."""
    job = client.post(
        "/api/classify",
        json={
            "label": "stagger-late",
            "emails": [
                {"text": "slow free winner", "deadline_ms": 150,
                 "submit_after_ms": 300, "simulate_latency_ms": 500},
                {"text": "fast free winner", "deadline_ms": 1000,
                 "submit_after_ms": 300},
            ],
        },
    ).json()
    slow, fast = job["entries"]
    assert slow["status"] == "timeout"
    assert slow["score"] is None
    assert slow["late_result_discarded"] is True
    assert fast["status"] == "ok"
    # Same batch (same version, both accepted at the same instant).
    assert slow["batch_id"] == fast["batch_id"]


def test_sync_classify_uses_independent_stagger_timing(client):
    """The synchronous endpoint shares the same per-entry scheduling path."""
    result = client.post(
        "/api/classify",
        json={
            "label": "sync-stagger",
            "emails": [
                {"text": "front sync free", "deadline_ms": 1000,
                 "submit_after_ms": 300},
                {"text": "rear sync free", "deadline_ms": 1000,
                 "submit_after_ms": 0},
            ],
        },
    ).json()
    assert result["status"] == "completed"
    front, rear = result["entries"]
    assert front["index"] == 1 and rear["index"] == 2
    front_at, rear_at = _parse(front["accepted_at"]), _parse(rear["accepted_at"])
    assert 0.2 <= (front_at - rear_at).total_seconds() <= 0.45
    assert all(e["status"] == "ok" for e in result["entries"])
