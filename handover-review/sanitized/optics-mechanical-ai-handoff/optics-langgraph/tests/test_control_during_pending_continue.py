"""A pause/stop requested while a continue is waiting for the worker must be honoured, and a control request
older than the continue must still be cleared (Runtime level, no HTTP, no model).

The worker is held deterministically by holding data/runtime.lock (what another process would do): the thread
running Runtime.execute blocks inside `with self.graph(...)`, before it starts the continue.
"""
import fcntl
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from optics_graph.runtime import Runtime, RunInput

MICHELSON = "Assemble a 15-component complete Michelson interferometer."
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class Recorder(dict):
    """runtime.live_tasks stand-in that remembers every node that ever started."""
    def __init__(self):
        super().__init__()
        self.started = []

    def __setitem__(self, key, value):
        self.started.append(value)
        super().__setitem__(key, value)


@pytest.fixture
def runtime(tmp_path):
    rt = Runtime(tmp_path)
    rt.live_tasks = Recorder()
    return rt


def make_run(rt, **options):
    return rt.create(RunInput(request=MICHELSON, profile="auto", mode="github_local", simulation_backend="algorithm_v1",
                              provider="replay", algorithm_options=options or {"delay_s": 0}))


def paused_run(rt, stored="paused"):
    """First node done, paused at the boundary; control.json (the honoured pause) is left behind, as today."""
    run_id = make_run(rt)
    rt.control(run_id, "pause")
    assert rt.execute(run_id)["status"] == "paused"
    assert (rt.run_dir(run_id) / "control.json").is_file()
    if stored != "paused":  # `running` is what a crash leaves; `interrupted` what a failed node leaves
        (rt.run_dir(run_id) / "run.json").write_text(json.dumps({**rt.get(run_id), "status": stored}))
    rt.live_tasks.started.clear()
    return run_id


class PendingContinue:
    """Accept a continue (capture the control sequence, as the callers do), then let it wait on runtime.lock."""
    def __init__(self, rt, run_id, capture=True):
        self.rt, self.run_id, self.out = rt, run_id, {}
        self.lock = (rt.data / "runtime.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        seq = {"control_seq": rt.control_seq(run_id)} if capture else {}

        def work():
            try:
                self.out["result"] = rt.execute(run_id, continue_run=True, **seq)
            except Exception as exc:  # noqa: BLE001
                self.out["error"] = exc
        self.thread = threading.Thread(target=work)
        self.thread.start()

    def release(self):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        self.thread.join(60)
        assert not self.thread.is_alive()
        return self.out


# ---- 1. honoured while the continue waits ---------------------------------------------------------------
@pytest.mark.parametrize("action,expected", [("pause", "paused"), ("stop", "stopped")])
@pytest.mark.parametrize("stored", ["running", "paused", "interrupted"])
def test_request_while_a_continue_waits_is_honoured_before_any_node(runtime, stored, action, expected):
    run_id = paused_run(runtime, stored)
    before = runtime.get(run_id)["next_node"]
    pending = PendingContinue(runtime, run_id)
    runtime.control(run_id, action)
    out = pending.release()
    assert "error" not in out, out  # (3) a clean outcome, not a worker ValueError
    assert out["result"]["status"] == expected and runtime.get(run_id)["status"] == expected
    assert runtime.get(run_id)["next_node"] == before
    assert runtime.live_tasks.started == [], "a node ran between the request and the stop"


def test_stop_during_a_pending_resume_ends_stopped_with_no_exception(runtime):
    """(3) the old accidental path: stored `paused`, control() rewrote run.json to stopped, execute raised."""
    run_id = paused_run(runtime, "paused")
    pending = PendingContinue(runtime, run_id)
    runtime.control(run_id, "stop")
    assert runtime.get(run_id)["status"] == "stopped"
    out = pending.release()
    assert "error" not in out and out["result"]["status"] == "stopped"
    assert runtime.live_tasks.started == []


# ---- 2. stale control.json is still cleared --------------------------------------------------------------
@pytest.mark.parametrize("stored", ["running", "paused", "interrupted"])
def test_control_older_than_the_continue_is_cleared_and_the_run_completes(runtime, stored):
    run_id = paused_run(runtime, stored)
    stale = json.loads((runtime.run_dir(run_id) / "control.json").read_text())
    assert stale["action"] == "pause" and stale["seq"] == runtime.control_seq(run_id)
    out = PendingContinue(runtime, run_id).release()  # nothing follows: (5) pause, continue, nothing
    assert "error" not in out and out["result"]["status"] == "algorithm_passed"
    assert not (runtime.run_dir(run_id) / "control.json").exists()


def test_a_continue_with_no_captured_sequence_still_clears_everything_before_it(runtime):
    """The CLI calls execute(continue_run=True) with no sequence: today's behaviour is kept."""
    run_id = paused_run(runtime, "paused")
    out = PendingContinue(runtime, run_id, capture=False).release()
    assert "error" not in out and out["result"]["status"] == "algorithm_passed"


def test_sequence_only_grows_across_clears_and_is_per_request(runtime):
    run_id = make_run(runtime)
    assert runtime.control_seq(run_id) == 0
    runtime.control(run_id, "pause")
    runtime.control(run_id, "stop")
    assert runtime.control_seq(run_id) == 2
    assert json.loads((runtime.run_dir(run_id) / "control.json").read_text()) == {"action": "stop", "seq": 2}
    (runtime.run_dir(run_id) / "control.json").unlink()
    runtime.control(run_id, "pause")
    assert runtime.control_seq(run_id) == 3


# ---- 5. sequences ------------------------------------------------------------------------------------------
def test_pause_continue_then_stop_before_the_worker_starts_stops(runtime):
    run_id = paused_run(runtime, "running")  # a pause is already on disk from before
    pending = PendingContinue(runtime, run_id)
    runtime.control(run_id, "stop")
    out = pending.release()
    assert out["result"]["status"] == "stopped" and runtime.live_tasks.started == []


def test_stop_then_continue_on_a_stopped_run_is_refused_as_before(runtime):
    run_id = paused_run(runtime, "paused")
    runtime.control(run_id, "stop")  # no continue pending: stored paused -> stopped
    assert runtime.get(run_id)["status"] == "stopped"
    out = PendingContinue(runtime, run_id).release()
    assert isinstance(out["error"], ValueError) and "no interrupted computation" in str(out["error"])
    assert runtime.get(run_id)["status"] == "stopped" and runtime.live_tasks.started == []


def test_a_request_honoured_by_one_continue_does_not_leak_into_the_next(runtime):
    run_id = paused_run(runtime, "running")
    pending = PendingContinue(runtime, run_id)
    runtime.control(run_id, "pause")
    assert pending.release()["result"]["status"] == "paused"
    out = PendingContinue(runtime, run_id).release()
    assert out["result"]["status"] == "algorithm_passed"


# ---- 4. unchanged behaviour ----------------------------------------------------------------------------------
def test_boundary_pause_and_stop_and_resume_after_pause_are_unchanged(runtime):
    run_id = make_run(runtime)
    runtime.control(run_id, "pause")
    assert runtime.execute(run_id)["status"] == "paused"
    assert runtime.execute(run_id, continue_run=True)["status"] == "algorithm_passed"
    second = make_run(runtime)
    runtime.control(second, "stop")
    assert runtime.execute(second)["status"] == "stopped"


# ---- cross-process: control() from another process, through files only ---------------------------------------
def test_a_stop_from_another_process_is_honoured_by_a_waiting_continue(runtime):
    run_id = paused_run(runtime, "running")
    pending = PendingContinue(runtime, run_id)
    code = ("import sys; from pathlib import Path; from optics_graph.runtime import Runtime; "
            "Runtime(Path(sys.argv[1])).control(sys.argv[2], 'stop')")
    subprocess.run([sys.executable, "-c", code, str(runtime.data), run_id], cwd=PROJECT, check=True, timeout=120)
    out = pending.release()
    assert out["result"]["status"] == "stopped" and runtime.live_tasks.started == []
