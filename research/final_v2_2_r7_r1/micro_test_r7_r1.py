from __future__ import annotations
import json
from pathlib import Path
from . import controller, synthetic_predictions

def run(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    config={"purpose":"R7_NON_SCIENTIFIC_PRODUCTION_PATH_MICRO_TEST","output_root":str(root.resolve()),
            "jobs":[{"job_id":"synthetic_target_only_graph","seed":17,"updates":1,"depends_on":None}]}
    config_path=root/"synthetic_manifest.json";config_path.write_text(json.dumps(config),encoding="utf-8")
    dispatch=controller.synthetic_dispatch(str(config_path));pipeline=synthetic_predictions.run(str(config_path))
    return {"status":"PASS","dispatch":dispatch,"pipeline":pipeline,"real_final_optimizer_updates":0,
            "real_final_predictions":False,"phase_b_labels_opened":False}
