"""R3 package-closure command surface; scientific/data semantics equal R2."""
from __future__ import annotations
import argparse,json,platform,sys
from . import core,data_gate,verify_data_gate
MANIFEST='deployment/final_v2_2_a40/BUNDLE_MANIFEST_DATA_GATE_R3.json'
def check(sha):
 core.verify_authority();m=core.read_json(MANIFEST)
 if core.sha(MANIFEST)!=sha:raise RuntimeError('package manifest hash mismatch')
 for e in m['files']:
  if core.sha(e['path'])!=e['sha256']:raise RuntimeError('member mismatch: '+e['path'])
 if core.sha(data_gate.FINAL_MANIFEST)!=m['final_data_manifest']['sha256']:raise RuntimeError('data manifest mismatch')
 v=verify_data_gate.verify()
 return {'status':'PASS','package_members':len(m['files']),'neural_fits':v['neural_fits'],'evaluation_passes':v['evaluation_passes'],'reporting_cells':v['reporting_cells'],'phase_a_execution_data_constructed':True,'phase_b_targets_materialized':False,'final_predictions_generated':False,'final_metrics_computed':False,'final_experiment_started':False,'optimizer_updates':0,'model_forward_calls_on_final_labels':0}
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('package-check','preflight','dry-run'));p.add_argument('--manifest-sha256',required=True);a=p.parse_args();r=check(a.manifest_sha256)
 if a.command!='package-check':r.update(status='PASS_PHASE_A_DATA_BOUND_NO_EXECUTION' if a.command=='preflight' else 'PASS_DRY_RUN',runtime={'python':sys.version,'platform':platform.platform()},determinism=core.deterministic_runtime(),phase_b_authorized=False)
 if a.command=='dry-run':r['files_written']=[]
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
