"""Non-scientific subprocess fixture; cannot train, evaluate or read datasets."""
import argparse
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--job-id",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());assert cfg["purpose"]=="NON_SCIENTIFIC_CONTROLLER_STARTUP_FIXTURE"
    j=next(j for j in cfg["jobs"] if j["job_id"]==a.job_id);assert a.job_id.startswith("synthetic_")
    root=Path(cfg["output_root"]).resolve();root.relative_to(Path(__file__).resolve().parents[2]/"tmp/stage4_v2_2_synthetic_tests")
    if j.get("fail"):raise RuntimeError("Intentional synthetic worker failure")
    if not j.get("omit_completion"):
        with (root/(a.job_id+".completion.json")).open("x",encoding="utf-8") as f:
            json.dump({"purpose":cfg["purpose"],"job_id":a.job_id},f)

if __name__=="__main__":main()
