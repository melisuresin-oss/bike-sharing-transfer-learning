"""Prepare or execute the immutable Stage-1 V2.2 map. Never starts later stages."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
sys.path[:0]=[str(ROOT/"tmp/neural_build/pydeps"),str(ROOT)]
from research.stage1_v2_2.core import (
    OUT,CACHE,NEW_CODE,setup,authority,prepare_cache,freeze_design,preflight,
    read,write,sha,path,worker,close_stage1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.prepare==args.execute:parser.error("Choose exactly one of --prepare or --execute")
    setup()
    if args.prepare:
        reg=authority()
        cache=prepare_cache(reg)
        contract,jobmap=freeze_design(cache)
        result=preflight(reg,cache,contract,jobmap)
        print("PREFLIGHT PASS "+str(result),flush=True)
        return 0
    contract=read(OUT+"scientific_manifest.json")
    for p,h in contract["code_sha256"].items():
        if sha(p)!=h:raise RuntimeError("Sealed executable changed: "+p)
    pf=read(OUT+"preflight_report.json")
    if pf["status"]!="PASS" or any(c["status"]!="PASS" for c in pf["checks"]):
        raise RuntimeError("Preflight did not pass")
    if sha(OUT+"job_map.json")!=pf["job_map"]["sha256"]:raise RuntimeError("Job map changed")
    # Recheck authoritative input identities at the launch boundary.
    authority()
    tasks=read(OUT+"job_map.json")["tasks"]
    if any(path(OUT+f"jobs/{t['task_id']}.json").exists() for t in tasks):
        raise RuntimeError("Execution records exist; do not silently repeat scientific fits")
    write(OUT+"launch_record.json",{"status":"LAUNCHED","workers":6,"threads_per_worker":2,
         "job_map_sha256":sha(OUT+"job_map.json"),"preflight_sha256":sha(OUT+"preflight_report.json"),
         "scientific_manifest_sha256":sha(OUT+"scientific_manifest.json"),"source_fits":24,"adaptation_fits":24})
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,t):t["task_id"] for t in tasks}
        for f in as_completed(futures):
            result=f.result()
            print("TASK COMPLETE "+futures[f]+" "+str(result),flush=True)
    close_stage1()
    return 0


if __name__=="__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

