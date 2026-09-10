"""Data-gate-bound Phase-A package checker and dry-run."""
from __future__ import annotations
import argparse,json,platform,sys
from . import core,data_gate,jobs,verify_data_gate

PACKAGE_MANIFEST='deployment/final_v2_2_a40/BUNDLE_MANIFEST_DATA_GATE_R2.json'

def package_check(manifest_sha):
 core.verify_authority();m=core.read_json(PACKAGE_MANIFEST)
 if core.sha(PACKAGE_MANIFEST)!=manifest_sha:raise RuntimeError('package manifest hash mismatch')
 for e in m['files']:
  if core.sha(e['path'])!=e['sha256']:raise RuntimeError('package member hash mismatch: '+e['path'])
 if core.sha(data_gate.FINAL_MANIFEST)!=m['final_data_manifest']['sha256']:raise RuntimeError('final-data binding mismatch')
 result=verify_data_gate.verify()
 return {'status':'PASS','package_members':len(m['files']),**{k:result[k] for k in ('neural_fits','evaluation_passes','reporting_cells')},'phase_a_execution_data_constructed':True,'phase_b_targets_materialized':False,'final_predictions_generated':False,'final_metrics_computed':False,'final_experiment_started':False,'optimizer_updates':0,'model_forward_calls_on_final_labels':0}

def preflight(sha):
 return {**package_check(sha),'status':'PASS_PHASE_A_DATA_BOUND_NO_EXECUTION','runtime':{'python':sys.version,'platform':platform.platform()},'determinism':core.deterministic_runtime(),'phase_b_authorized':False}
def dry_run(sha):return {**preflight(sha),'status':'PASS_DRY_RUN','files_written':[]}

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('package-check','preflight','dry-run'));p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
 print(json.dumps(package_check(a.manifest_sha256) if a.command=='package-check' else preflight(a.manifest_sha256) if a.command=='preflight' else dry_run(a.manifest_sha256),indent=2))
if __name__=='__main__':main()
