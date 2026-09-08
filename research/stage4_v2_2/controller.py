"""Four-worker detached controller; same dispatch path is used in integration fixtures."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from research.stage4_v2_2 import core as c,a40


def dispatch(jobs,workers,worker_command,validate_completion,root,record,log_path):
    """A zero process exit and validated committed completion are both mandatory."""
    if workers!=4:raise RuntimeError("Frozen operational worker count is four")
    pending=list(jobs);active={};done=set();failures=[]
    flags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)
    try:
        while pending or active:
            while len(active)<workers and not failures:
                ready=next((j for j in pending if j["depends_on"] is None or j["depends_on"] in done),None)
                if ready is None:break
                pending.remove(ready);jid=ready["job_id"]
                out,err=log_path(jid);out.parent.mkdir(parents=True,exist_ok=True)
                oh=out.open("xb");eh=err.open("xb")
                try:p=subprocess.Popen(worker_command(ready),cwd=root,stdout=oh,stderr=eh,creationflags=flags)
                except BaseException:oh.close();eh.close();raise
                active[jid]=(p,oh,eh,ready)
                record(jid+".launch.json",{"status":"LAUNCHED","job_id":jid,"pid":p.pid,"coordinator_pid":os.getpid(),"workers":4})
            if not active:
                if pending:raise RuntimeError("Dependency deadlock or undispatched jobs")
                break
            finished=[jid for jid,(p,_,_,_) in active.items() if p.poll() is not None]
            if not finished:time.sleep(.2);continue
            for jid in finished:
                p,oh,eh,job=active.pop(jid);oh.close();eh.close();completion=None;error=None
                if p.returncode==0:
                    try:completion=validate_completion(job)
                    except Exception as e:error=type(e).__name__+": "+str(e)
                success=p.returncode==0 and completion is not None
                record(jid+".exit.json",{"status":"EXITED_ZERO_WITH_COMPLETION" if success else "FAILED",
                    "job_id":jid,"pid":p.pid,"exit_code":p.returncode,"completion":completion,"validation_error":error})
                if success:done.add(jid)
                else:failures.append({"job_id":jid,"exit_code":p.returncode,"error":error})
            if failures:raise RuntimeError("Genuine worker failure stops dispatch: "+json.dumps(failures))
        if len(done)!=len(jobs):raise RuntimeError("Incomplete controller workload")
        return {"status":"EXITED_ZERO_WITH_ALL_COMPLETIONS","completed":len(done),"workers":4}
    finally:
        for jid,(p,oh,eh,job) in active.items():
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            oh.close();eh.close()
            record(jid+".exit.json",{"status":"ABORTED_AFTER_CONTROLLER_FAILURE","job_id":jid,"pid":p.pid,"exit_code":p.returncode})


def worker_command(job,manifest_sha,pf):
    return [sys.executable,"-X","utf8","-B","-m","research.stage4_v2_2.a40","job",
            "--manifest-sha256",manifest_sha,"--job-id",job["job_id"],"--preflight",pf["path"],"--preflight-sha256",pf["sha256"]]


def launch(manifest_sha,resume=False):
    jobs,cache=a40.verify_package(manifest_sha)
    pf=a40.preflight(manifest_sha,resume)
    run=c.OUT+"controller_runs/"+uuid.uuid4().hex+"/"
    c.write(run+"controller_started.json",{"status":"RUNNING","coordinator_pid":os.getpid(),"manifest_sha256":manifest_sha,
             "preflight":pf,"workers":4,"source_fits":72,"adaptation_fits":72,"timestamp_utc":c.utc()})
    def record(name,value):
        return c.write(run+"workers/"+name,{**value,"manifest_sha256":manifest_sha,"timestamp_utc":c.utc()})
    def validate(job):
        a40.validate_completed(job,manifest_sha,cache)
        return c.entry(a40.completed_path(job["job_id"]))
    def logs(jid):return c.path(run+"workers/"+jid+".stdout.log",True),c.path(run+"workers/"+jid+".stderr.log",True)
    try:
        result=dispatch(jobs,4,lambda job:worker_command(job,manifest_sha,pf),validate,c.ROOT,record,logs)
        c.write(run+"controller_complete.json",{**result,"manifest_sha256":manifest_sha,
            "source_fits":72,"adaptation_fits":72,"freeze_started":False,
            "next":"Run read-only postrun validation, then freeze; STOP before final experiment",
            "final_target_labels_accessed":False,"final_experiment_started":False,"timestamp_utc":c.utc()})
        print(json.dumps(result),flush=True)
    except BaseException as e:
        c.write(run+"controller_failed.json",{"status":"FAILED","error":type(e).__name__+": "+str(e),
                 "manifest_sha256":manifest_sha,"no_more_jobs_dispatched":True,"final_target_labels_accessed":False,"timestamp_utc":c.utc()})
        raise


def startup_fixture(config_path):
    """Explicit synthetic integration mode: no production data, runtime or fit call."""
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("purpose")!="NON_SCIENTIFIC_CONTROLLER_STARTUP_FIXTURE":raise RuntimeError("Explicit synthetic fixture purpose required")
    root=Path(cfg["output_root"]).resolve()
    allowed=(c.ROOT/"tmp/stage4_v2_2_synthetic_tests").resolve()
    root.relative_to(allowed);root.mkdir(parents=True,exist_ok=True)
    jobs=cfg["jobs"]
    if any(not j["job_id"].startswith("synthetic_") for j in jobs):raise RuntimeError("Real job IDs forbidden in fixture")
    def record(name,value):
        with (root/name).open("x",encoding="utf-8") as f:json.dump(value,f)
    def validate(job):
        p=root/(job["job_id"]+".completion.json")
        r=json.loads(p.read_text());c.require(r=={"purpose":cfg["purpose"],"job_id":job["job_id"]},"Synthetic completion")
        return {"path":str(p),"sha256":__import__("hashlib").sha256(p.read_bytes()).hexdigest()}
    def command(job):
        return [sys.executable,"-X","utf8","-B","-m","research.stage4_v2_2.synthetic_worker",
                "--config",str(Path(config_path).resolve()),"--job-id",job["job_id"]]
    try:
        result=dispatch(jobs,4,command,validate,c.ROOT,record,
                        lambda jid:(root/(jid+".stdout.log"),root/(jid+".stderr.log")))
        record("controller_complete.json",{**result,"purpose":cfg["purpose"],"real_optimizer_updates":0,"real_evaluations":0})
    except BaseException as e:
        record("controller_failed.json",{"error":str(e),"purpose":cfg["purpose"]});raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest-sha256");p.add_argument("--workers",type=int,default=4)
    p.add_argument("--resume",action="store_true");p.add_argument("--startup-fixture")
    args=p.parse_args()
    c.require(args.workers==4,"Exactly four operational workers")
    if args.startup_fixture:
        c.require(args.manifest_sha256 is None and not args.resume,"Fixture cannot carry production authorization")
        startup_fixture(args.startup_fixture)
    else:
        c.require(args.manifest_sha256 is not None,"Explicit manifest hash required")
        with a40.execution_lock("stage4_controller"):launch(args.manifest_sha256,args.resume)


if __name__=="__main__":main()
