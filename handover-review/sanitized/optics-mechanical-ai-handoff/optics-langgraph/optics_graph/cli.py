import argparse
import json
from pathlib import Path
from .runtime import Runtime, RunInput
from .settings import DATA, DEFAULT_REQUEST, PROJECT


def main():
    parser=argparse.ArgumentParser(description="Local LangGraph workbench for optical experiments")
    sub=parser.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve");serve.add_argument("--port",type=int,default=8766)
    run=sub.add_parser("run")
    run.add_argument("--provider",choices=["replay","openai"],default="replay")
    run.add_argument("--mode",choices=["mock","design","local_sim","remote_sim","real","legacy_evidence","legacy_mujoco","github_local","github_sim"],default="mock")
    run.add_argument("--sim-scenario", choices=["nominal", "transient", "persistent"], default="nominal")
    run.add_argument("--team-fault", choices=["none","vision","prompt","act","controller","sim","output","execution","matrix"], default="none")
    run.add_argument("--team-transport", choices=["direct", "file_v1"], help="The team workflow uses the separate file-job interface by default")
    run.add_argument("--prob0", type=Path, help="Existing PROB0 JSON. When provided, use the natural-language entry")
    run.add_argument("--request",default=DEFAULT_REQUEST)
    run.add_argument("--review",action="store_true")
    run.add_argument("--fail-action")
    show=sub.add_parser("show");show.add_argument("run_id")
    resume=sub.add_parser("resume");resume.add_argument("run_id");resume.add_argument("--reject",action="store_true");resume.add_argument("--continue-run",action="store_true")
    graph_parser = sub.add_parser("graph")
    graph_parser.add_argument("--team",action="store_true")
    sub.add_parser("doctor")
    args=parser.parse_args()
    if args.command=="serve":
        import uvicorn
        uvicorn.run("optics_graph.server:app",host="127.0.0.1",port=args.port,log_level="warning",access_log=False)
        return
    runtime=Runtime()
    if args.command=="doctor":
        import importlib.metadata
        from .legacy import probe_legacy
        from .settings import public_settings
        output={"langgraph":importlib.metadata.version("langgraph"),"model":public_settings(),"legacy":probe_legacy(PROJECT),"real_motion_enabled":False}
    elif args.command=="graph":
        from .graph import build_graph
        from .team_flow import build_team_graph
        builder = build_team_graph if args.team else build_graph
        print(builder(DATA).get_graph().draw_mermaid());return
    elif args.command=="run":
        from .sim_io import read_json
        request=RunInput(provider=args.provider,mode=args.mode,request=args.request,require_review=args.review,fail_action=args.fail_action,sim_scenario=args.sim_scenario,team_fault=args.team_fault,
                         team_transport=args.team_transport or ("file_v1" if args.mode.startswith("github_") else "direct"),
                         team_entry="nl" if args.prob0 else "images",prob0=read_json(args.prob0) if args.prob0 else None,
                         profile="legacy_vision_michelson_15" if args.mode.startswith("legacy_") else "phase1_michelson")
        run_id=runtime.create(request)
        print(json.dumps({"run_id":run_id,"status":"started"}),flush=True)
        output=runtime.execute(run_id)
    elif args.command=="resume":
        state=runtime.get(args.run_id)
        output=runtime.execute(args.run_id,continue_run=True) if args.continue_run else runtime.execute(args.run_id,resume={"approved":not args.reject,"plan_sha256":state["plan_sha256"]})
    else:output=runtime.get(args.run_id)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
