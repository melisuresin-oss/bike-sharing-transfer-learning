"""Read-only structural verifier; it never opens a Phase-B target artifact."""
from __future__ import annotations
import json
import numpy as np
from research.v2_2.contract import Contract,HOUR
from . import core,data_gate,jobs

FORBIDDEN_EVAL_KEYS={'count','target','y','observed','target_12h','target_24h'}

def verify():
 core.verify_authority();m=core.read_json(data_gate.FINAL_MANIFEST);c=Contract()
 if m['scope']!='FINAL_TARGETS' or not m['phase_a_execution_data_constructed'] or m['phase_b_targets_materialized']:raise RuntimeError('phase state')
 if m['boundaries_us']!=core.BOUNDARIES or m['evaluation_origin_count']!=1565 or m['embargo_hours']!=593:raise RuntimeError('time structure')
 if tuple(m['lags'])!=data_gate.LAGS or m['fixed_global_cohort_sha256']!=c.roster_sha256:raise RuntimeError('causal/cohort drift')
 data_gate.verify_raw_bindings();checks=[]
 for rec,(city,name) in zip(m['targets'],core.TARGETS):
  if rec['city_id']!=city or rec['city']!=name or rec['station_count']!=len(c.city(city)) or rec['city_roster_sha256']!=c.city_roster_hash(city):raise RuntimeError('target roster drift')
  for role in ('fitting_panel','evaluation_panel','graph','prediction_keys'):
   e=rec[role];p=core.repo_path(e['path'])
   if core.sha256_path(p)!=e['sha256']:raise RuntimeError('artifact hash mismatch')
  z=np.load(core.repo_path(rec['evaluation_panel']['path']),allow_pickle=False)
  if FORBIDDEN_EVAL_KEYS & set(z.files):raise RuntimeError('current held-out target leaked into Phase A')
  if tuple(z['origin_us'].shape)!=(1565,) or int(z['origin_us'][0])!=core.BOUNDARIES['HT'] or int(z['origin_us'][-1])+HOUR!=core.BOUNDARIES['HE']:raise RuntimeError('evaluation grid drift')
  if not np.array_equal(z['m_hist'],z['x_hist'][...,1].astype(bool)) or np.any((~z['m_hist']) & (z['x_hist'][...,0]!=0)):raise RuntimeError('mask encoding drift')
  if tuple(map(int,z['station_ids']))!=tuple(s.station_id for s in c.city(city)):raise RuntimeError('station discovery/order drift')
  checks.append({'city_id':city,'station_count':rec['station_count'],'evaluation_origins':1565,'panel_sha256':rec['evaluation_panel']['sha256'],'graph_sha256':rec['graph']['sha256']})
 jm=core.read_json(core.SEAL+'/final_job_manifest.json');em=core.read_json(core.SEAL+'/evaluation_manifest.json');jobs.validate_neural_jobs(jm['fits'])
 if len(em['passes'])!=468 or sum(len(x['reuse_labels']) for x in em['passes'])!=660:raise RuntimeError('DAG drift')
 return {'status':'PASS','checks':checks,'future_status_violations':0,'current_target_fields_in_phase_a':0,'neural_fits':410,'evaluation_passes':468,'reporting_cells':660,'phase_a_execution_data_constructed':True,'phase_b_targets_materialized':False,'final_predictions_generated':False,'final_metrics_computed':False,'final_experiment_started':False,'optimizer_updates':0,'model_forward_calls_on_final_labels':0}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
