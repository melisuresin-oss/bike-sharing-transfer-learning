"""Restart-safe four-worker controller using immutable completion evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from . import core, validation

def dispatch(jobs, command, validate_completion, log_root: Path, record, workers=4, already_done=()):
    if workers != 4: raise RuntimeError("R7 operational concurrency is fixed at four")
    pending = list(jobs); active = {}; done = set(already_done); newly_done = set(); flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    while pending or active:
        while pending and len(active) < workers:
            ready = next((j for j in pending if j.get("depends_on") is None or j["depends_on"] in done), None)
            if ready is None: break
            pending.remove(ready); attempt = str(uuid.uuid4()); jid = ready["job_id"]
            stdout = log_root / f"{jid}.{attempt}.stdout.log"; stderr = log_root / f"{jid}.{attempt}.stderr.log"
            stdout.parent.mkdir(parents=True, exist_ok=True); oh = stdout.open("xb"); eh = stderr.open("xb")
            try: process = subprocess.Popen(command(ready, attempt), stdout=oh, stderr=eh, cwd=core.ROOT, creationflags=flags)
            except BaseException: oh.close(); eh.close(); raise
            active[jid] = (process, oh, eh, ready, attempt)
            record({"status": "LAUNCHED", "job_id": jid, "attempt_uuid": attempt, "pid": process.pid,
                    "controller_pid": os.getpid(), "workers": 4, "stdout": str(stdout), "stderr": str(stderr)})
        if not active:
            if pending: raise RuntimeError("dependency deadlock")
            break
        finished = [jid for jid,(process,*_) in active.items() if process.poll() is not None]
        if not finished: time.sleep(.5); continue
        for jid in finished:
            process, oh, eh, job, attempt = active.pop(jid); oh.close(); eh.close()
            if process.returncode != 0:
                record({"status": "FAILED", "job_id": jid, "attempt_uuid": attempt, "exit_code": process.returncode})
                for other,(p,o,e,_,a) in active.items():
                    if p.poll() is None:
                        p.terminate()
                        try: p.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            p.kill(); p.wait(timeout=30)
                    o.close(); e.close(); record({"status": "ABORTED_AFTER_PEER_FAILURE", "job_id": other, "attempt_uuid": a, "exit_code": p.returncode})
                raise RuntimeError(f"worker {jid} failed; dispatch stopped")
            validate_completion(job); done.add(jid); newly_done.add(jid)
            record({"status": "EXITED_ZERO_WITH_VALID_COMPLETION", "job_id": jid, "attempt_uuid": attempt, "exit_code": 0})
    return newly_done

def launch(package_sha: str, controller_uuid: str | None = None):
    from .locks import LifetimeLock
    controller_uuid=controller_uuid or uuid.uuid4().hex
    owner={"schema_version":"final_v2_2_r7_r1.controller_lock.1","controller_uuid":controller_uuid,
           "pid":os.getpid(),"package_manifest_sha256":package_sha,"acquired_utc":core.utc()}
    lock=LifetimeLock(core.path(f"{core.OUT}/locks/controller.lock",output=True),
                      core.path(f"{core.OUT}/locks/controller.owner.json",output=True),owner)
    with lock:
        core.atomic_json(f"{core.OUT}/controller_handshakes/{controller_uuid}.accepted.json",{"status":"GLOBAL_CONTROLLER_LOCK_ACQUIRED","controller_uuid":controller_uuid,"pid":os.getpid(),"package_manifest_sha256":package_sha,"accepted_utc":core.utc()})
        validation.preflight(package_sha,strict=True)
        existing=validation.validate_existing(package_sha)
        if existing["active_job_claims"]: raise RuntimeError("active orphaned job claims require resolution before relaunch: "+repr(existing["active_job_claims"]))
        complete=set(existing["completed_job_ids"])
        jobs=core.read_json(core.JOB_MANIFEST)["fits"]
        pending=[j for j in jobs if j["job_id"] not in complete]
        run_id=controller_uuid; run_root=core.path(f"{core.OUT}/controller_runs/{run_id}",output=True); logs=run_root/"worker_logs"
        core.atomic_json(f"{core.OUT}/controller_runs/{run_id}/started.json",{"status":"RUNNING","run_id":run_id,
            "controller_pid":os.getpid(),"package_manifest_sha256":package_sha,"workers":4,
            "already_completed":len(complete),"remaining":len(pending),"started_utc":core.utc()})
        counter={"value":0}
        def record(value):
            counter["value"]+=1
            core.atomic_json(f"{core.OUT}/controller_runs/{run_id}/events/{counter['value']:06d}.json",{**value,"timestamp_utc":core.utc()})
        def command(job,attempt):
            return [sys.executable,"-X","utf8","-B","-m","research.final_v2_2_r7_r1.worker","--job-id",job["job_id"],
                    "--attempt-uuid",attempt,"--package-sha256",package_sha,"--device","cuda"]
        try:
            done=dispatch(pending,command,lambda j:validation.validate_one_completion(j,package_sha),logs,record,4,complete)
            result=validation.validate_existing(package_sha)
            core.atomic_json(f"{core.OUT}/controller_runs/{run_id}/completed.json",{"status":"CONTROLLER_COMPLETED",
                "run_id":run_id,"new_completions":len(done),"validation":result,"completed_utc":core.utc()})
            return result
        except BaseException as exc:
            core.atomic_json(f"{core.OUT}/controller_runs/{run_id}/failed.json",{"status":"CONTROLLER_FAILED",
                "run_id":run_id,"error":type(exc).__name__+": "+str(exc),"failed_utc":core.utc()})
            raise

def controller_status():
    from .locks import probe
    lock_path=core.path(f"{core.OUT}/locks/controller.lock",output=True)
    metadata=core.path(f"{core.OUT}/locks/controller.owner.json",output=True)
    owner=None
    try: owner=json.loads(metadata.read_text(encoding="utf-8"))
    except (FileNotFoundError,json.JSONDecodeError): pass
    return {"status":"CONTROLLER_STATUS","lock_currently_held":probe(lock_path),"last_owner_metadata":owner,
            "wmi_cim_used":False,"scientific_completion_source":"immutable completion records plus validators"}
def synthetic_dispatch(config_path: str):
    from .locks import LifetimeLock
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("purpose")!="R7_NON_SCIENTIFIC_PRODUCTION_PATH_MICRO_TEST": raise PermissionError("synthetic purpose required")
    jobs=cfg["jobs"]; root=Path(cfg["output_root"]).resolve(); root.mkdir(parents=True,exist_ok=True); events=[]
    owner={"schema_version":"r7.synthetic.controller_lock.1","controller_uuid":uuid.uuid4().hex,"pid":os.getpid()}
    with LifetimeLock(root/"controller.lock",root/"controller.owner.json",owner):
        def command(job,attempt): return [sys.executable,"-X","utf8","-B","-m","research.final_v2_2_r7_r1.worker",
            "--job-id",job["job_id"],"--attempt-uuid",attempt,"--synthetic-config",str(Path(config_path).resolve()),"--device","cpu"]
        def validate(job):
            value=json.loads((root/f"{job['job_id']}.completion.json").read_text())
            if value["status"]!="SYNTHETIC_COMPLETED" or value["job_id"]!=job["job_id"]: raise RuntimeError("synthetic completion invalid")
            return value
        done=dispatch(jobs,command,validate,root/"logs",events.append,4)
        result={"status":"PASS","completed":len(done),"events":events}
        out=root/"controller_result.json"
        if out.exists(): raise FileExistsError("synthetic controller result already exists")
        out.write_text(json.dumps(result),encoding="utf-8")
        return {"status":"PASS_SYNTHETIC_CONTROLLER","completed":len(done)}