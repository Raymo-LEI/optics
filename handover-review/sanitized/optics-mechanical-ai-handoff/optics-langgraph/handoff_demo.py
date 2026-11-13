"""Offline graph fixture: no API calls, no physics and no hardware."""
from optics_graph.runtime import RunInput, Runtime
from optics_graph.settings import DATA
import json

if __name__ == "__main__":
    runtime = Runtime(DATA)
    run_id = runtime.create(RunInput(request="Assemble a 15-component complete Michelson interferometer.",profile="auto",mode="github_local",simulation_backend="algorithm_v1",provider="replay",algorithm_options={"delay_s":0.0}))
    state = runtime.execute(run_id)
    assert state["status"] == "algorithm_passed", state.get("error")
    for h in state["algorithm_history"]:
        assert h["moved_ids"] == ["L25"]
        assert all(h["before"][k] == h["after"][k] for k in h["before"] if k != "L25")
    folder = runtime.run_dir(run_id)
    assert (folder/state["plan_path"]).read_bytes() == (folder/"team/execution-01/plan.json").read_bytes()
    assert state["result"]["motions_sent"] == 0
    print(json.dumps({"run_id":run_id,"status":state["status"],"source":"offline rule fixture, not a model-designed experiment","rounds":len(state["algorithm_history"]),"repair_object":"L25","same_plan":True,"motions_sent":0,"physical_verified":False},ensure_ascii=False,indent=2))
