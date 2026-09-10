"""Reusable fail-closed four-process controller for future authorized fits."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

def dispatch(items, command, validate, output:Path, workers=4):
    if workers!=4:raise RuntimeError("operational concurrency is fixed at four for this package")
    pending=list(items);active={};done=set();output.mkdir(parents=True,exist_ok=True)
    flags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)
    while pending or active:
        while pending and len(active)<workers:
            j=next((x for x in pending if x.get("depends_on") is None or x["depends_on"] in done),None)
            if j is None:break
            pending.remove(j);jid=j["job_id"]
            out=(output/(jid+".stdout.log")).open("xb");err=(output/(jid+".stderr.log")).open("xb")
            p=subprocess.Popen(command(j),stdout=out,stderr=err,creationflags=flags)
            active[jid]=(p,out,err,j)
        if not active:raise RuntimeError("dependency deadlock")
        finished=[k for k,v in active.items() if v[0].poll() is not None]
        if not finished:time.sleep(.2);continue
        for jid in finished:
            p,out,err,j=active.pop(jid);out.close();err.close()
            if p.returncode!=0:raise RuntimeError(f"worker {jid} failed with {p.returncode}")
            validate(j);done.add(jid)
    return done
