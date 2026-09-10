from __future__ import annotations
import json,os,subprocess,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import baselines,core,data,integrity,metrics_bootstrap as metrics,phase_b_gate,validation

class R7R1Tests(unittest.TestCase):
 def test_001_frozen_hashes(self):core.verify_frozen()
 def test_002_seeds(self):self.assertEqual(core.SEEDS,(17,29,43,71,101))
 def test_003_targets(self):self.assertEqual(core.TARGETS,(195,199,237,617))
 def test_004_sources(self):self.assertEqual(core.SOURCES,(129,194,438,467,476,532,619,658))
 def test_005_domain_map(self):self.assertEqual(core.DOMAIN_MAP,{129:0,194:1,438:2,467:3,476:4,532:5,619:6,658:7})
 def test_006_no_merge_or_omission(self):self.assertEqual((len(core.DOMAIN_MAP),len(set(core.DOMAIN_MAP.values()))),(8,8))
 def test_007_budgets(self):self.assertEqual(core.UPDATES,{"0":0,"1":100,"7":300,"30":600,"full":1200})
 def test_008_job_universe(self):
  jobs=core.read_json(core.JOB_MANIFEST)['fits'];self.assertEqual((len(jobs),len({j['job_id'] for j in jobs})),(410,410))
 def test_009_job_categories(self):self.assertEqual(Counter(j['category'] for j in core.read_json(core.JOB_MANIFEST)['fits']),Counter(source=10,adaptation=160,target_only=160,pooled=80))
 def test_010_eval_accounting(self):
  rows=core.read_json(core.EVALUATION_MANIFEST)['passes'];self.assertEqual((len(rows),sum(len(x['reuse_labels']) for x in rows)),(468,660))
 def test_011_all_families(self):self.assertEqual({j['method'] for j in core.read_json(core.JOB_MANIFEST)['fits']},{'ordinary_source_graphgru','grl_l50_constant_source_graphgru','target_only_vanilla','target_only_graph','ordinary_source_adapted','pooled_graph','grl_l50_constant_adapted'})
 def test_012_adaptation_dependencies(self):self.assertTrue(all(j['depends_on'] for j in core.read_json(core.JOB_MANIFEST)['fits'] if j['category']=='adaptation'))
 def test_013_no_partial_resume(self):self.assertTrue(all(j['attempt_policy']=='fresh_uuid_from_update_0_no_partial_resume' for j in core.read_json(core.JOB_MANIFEST)['fits']))
 def test_014_source_windows(self):
  rows=core.read_json(core.DATA_MANIFEST)['source_refit_partitions'];self.assertEqual([x['request']['city_ids'][0] for x in rows],list(core.SOURCES));self.assertTrue(all(x['request']['start']==core.BOUNDARIES['H0'] and x['request']['cutoff']==core.BOUNDARIES['HF'] for x in rows))
 def test_015_target_window_cutoffs(self):
  m=core.read_json(core.DATA_MANIFEST);self.assertTrue(all(a['request']['cutoff']==core.BOUNDARIES['HF'] and a['request']['start']<=core.BOUNDARIES['HF'] for t in m['targets'] for a in t['adaptations']))
 def test_016_eval_phase_a_no_target_fields(self):self.assertEqual(core.read_json(core.DATA_MANIFEST)['phase_a_evaluation_schema_excludes'],['count','target','y','observed_target'])
 def test_017_source_has_no_final_target(self):self.assertTrue(set(core.SOURCES).isdisjoint(core.TARGETS))
 def test_018_structural_binding_source(self):
  j=next(j for j in core.read_json(core.JOB_MANIFEST)['fits'] if j['category']=='source');b=data.structural_job_binding(j);self.assertEqual(len(b['source_snapshots']),8)
 def test_019_structural_binding_target(self):
  j=next(j for j in core.read_json(core.JOB_MANIFEST)['fits'] if j['category']=='target_only');b=data.structural_job_binding(j);self.assertEqual(b['target_snapshot']['path'].split('_')[-2],j['budget'])
 def test_020_structural_binding_pooled(self):
  j=next(j for j in core.read_json(core.JOB_MANIFEST)['fits'] if j['category']=='pooled');self.assertEqual(len(data.structural_job_binding(j)['source_graphs']),8)
 def test_021_path_firewall(self):
  for name in ('x/final_labels/y','sealed_final/x','x/phase_b_targets/y'):self.assertRaises(PermissionError,core.path,name)
 def test_022_runtime_determinism(self):
  r=core.runtime_contract();self.assertEqual((r['CUBLAS_WORKSPACE_CONFIG'],r['max_workers'],r['partial_checkpoint_resume'],r['early_stopping']),(':4096:8',4,False,False))
 def test_023_cli_has_no_phase_b(self):
  from .cli import COMMANDS
  self.assertEqual(set(COMMANDS),{'package-check','preflight','dry-run','validate-existing','controller-status','launch','postrun-validate','predict','validate-predictions','commit-predictions','archive-phase-a'})
 def test_024_validate_existing_zero(self):
  result=validation.validate_existing(core.sha(core.PACKAGE_MANIFEST));self.assertEqual((result['completed'],result['remaining']),(0,410))
 def test_025_postrun_refuses_zero(self):self.assertRaises(RuntimeError,validation.postrun_validate,'0'*64)
 def test_026_grl_shape_and_class7(self):
  import torch
  from .grl_final import FinalSourceInvariantGraphGRU
  model=FinalSourceInvariantGraphGRU();self.assertEqual(model.discriminator[-1].out_features,8)
  inputs={'x_hist':torch.zeros(1,24,2,2),'m_hist':torch.zeros(1,24,2,dtype=torch.bool),'x_week':torch.zeros(1,2,2),'x_static':torch.zeros(2,2),'x_calendar':torch.zeros(1,6),'adjacency':torch.eye(2)}
  for cls in range(8):model.objectives(inputs,torch.ones(1,2),torch.ones(1,2,dtype=torch.bool),cls)
  with self.assertRaises(ValueError):model.objectives(inputs,torch.ones(1,2),torch.ones(1,2,dtype=torch.bool),8)
 def test_027_stage4_implementation_untouched(self):self.assertEqual(core.sha('research/stage4_v2_2/method.py'),'2579d5af8f79e3abd49a16a654f2ca1116fb0fc178628aed52f1354cd1c26782')
 def test_028_persistence_no_fallback(self):
  p=SimpleNamespace(x_hist=np.zeros((1,24,2,2),np.float32));p.x_hist[0,0,:,0]=[3,4];p.x_hist[0,0,:,1]=[1,0];v,a=baselines.persistence(p);self.assertEqual(v.tolist(),[[3,4]]);self.assertEqual(a.tolist(),[[True,False]])
 def test_029_seasonal_no_fallback(self):
  p=SimpleNamespace(x_week=np.array([[[5,1],[0,0]]],np.float32));v,a=baselines.seasonal_naive(p);self.assertEqual((v.tolist(),a.tolist()),([[5,0]],[[True,False]]))
 def test_030_metrics_observed_zero(self):
  rows=[{'station_id':1,'hour':i,'label':0,'prediction':0,'label_valid':True,'prediction_available':True} for i in range(24)];self.assertEqual(metrics.count_metrics(rows)['valid_n'],24)
 def test_031_metrics_unknown_unavailable_excluded(self):self.assertEqual(metrics.count_metrics([{'station_id':1,'hour':0,'label':1,'prediction':0,'label_valid':False,'prediction_available':True}])['valid_n'],0)
 def test_032_station_macro_24h(self):
  rows=[{'station_id':1,'hour':i,'label':2,'prediction':1,'label_valid':True,'prediction_available':True} for i in range(24)];self.assertEqual(metrics.count_metrics(rows)['station_macro_mae'],1)
 def test_033_population_sd(self):self.assertEqual(metrics.aggregate_five_seeds([0,1,2,3,4]),{'mean':2.0,'population_sd':2**.5})
 def test_034_bootstrap_contract(self):
  self.assertEqual((sum(metrics.BLOCK_LENGTHS),metrics.BLOCK_LENGTHS),(1565,(168,)*9+(53,)));self.assertEqual(metrics.block_draws(195),metrics.block_draws(195))
 def test_035_terminal_block_retained(self):self.assertEqual(len(metrics.block_indices((9,)*10)),530)
 def test_036_linear_interpolation(self):self.assertEqual(metrics.linear_quantile([0,10],.25),2.5)
 def test_037_paired_gain(self):
  result=metrics.paired_gain_bootstrap(195,[2.0]*1565,[1.0]*1565);self.assertEqual(result['ci'],{'lower':1.0,'upper':1.0,'reason':None})
 def test_038_equal_city_macro(self):self.assertEqual(metrics.equal_city_macro({195:1,199:2,237:3,617:4}),2.5)
 def test_039_phase_b_requires_explicit_auth(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);[(p/n).write_text('{}') for n in ('a','c','l')]
   with patch('research.final_v2_2_r7_r1.phase_b_gate.commitment.verify'):
    with patch('research.final_v2_2_r7_r1.phase_b_gate.required_bindings',return_value={'x':'y'}):self.assertRaises(PermissionError,phase_b_gate.authorize,p/'a',p/'c',p/'l','p')
 def test_040_phase_b_toctou(self):
  cap=phase_b_gate.PhaseBCapability('a','b','c')
  with patch('research.final_v2_2_r7_r1.phase_b_gate.authorize',side_effect=[cap,phase_b_gate.PhaseBCapability('x','b','c')]):self.assertRaises(PermissionError,phase_b_gate.invoke_target_reader,cap,'a','c','l','p',lambda x:x)
 def test_041_tamper_matrix(self):
  names=['delete_prediction','add_prediction','modify_prediction','duplicate_prediction','alter_checkpoint','alter_completion','alter_job_manifest','alter_evaluation_manifest','alter_data_manifest','alter_implementation','alter_package_manifest','alter_protocol','alter_grl_clarification']
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);entries=[]
   for name in names:
    p=root/name;p.write_text(name);entries.append({'path':name,'bytes':p.stat().st_size,'sha256':core.sha256_path(p)})
   self.assertEqual(integrity.verify_entries(entries,lambda x:root/x)['entries'],13)
   for entry in entries:
    p=root/entry['path'];original=p.read_bytes();p.write_bytes(original+b'x')
    with self.subTest(entry=entry['path']):self.assertRaises(PermissionError,integrity.verify_entries,entries,lambda x:root/x)
    p.write_bytes(original)
   (root/names[0]).unlink();self.assertRaises(PermissionError,integrity.verify_entries,entries,lambda x:root/x)
 def test_042_lazy_import_without_torch(self):
  env={k:v for k,v in os.environ.items() if k.upper()!='PYTHONPATH'};code='import sys;import research.final_v2_2_r7_r1.cli;assert "torch" not in sys.modules'
  subprocess.run([sys.executable,'-c',code],cwd=core.ROOT,env=env,check=True)
 def test_043_synthetic_production_microtest(self):
  from .micro_test_r7_r1 import run
  with tempfile.TemporaryDirectory(dir=core.path('tmp')) as d:
   result=run(Path(d));self.assertEqual((result['status'],result['dispatch']['completed'],result['pipeline']['prediction_rows']),('PASS',1,6))

 def test_044_mechanical_job_contract(self):
  jobs=core.read_json(core.JOB_MANIFEST)['fits'];self.assertEqual(validation._validate_job_universe(jobs),Counter(source=10,adaptation=160,target_only=160,pooled=80))
 def test_045_evaluation_routing(self):
  from .predictions import LEARNED,_job_for
  jobs=core.read_json(core.JOB_MANIFEST)['fits'];rows=core.read_json(core.EVALUATION_MANIFEST)['passes'];learned=[r for r in rows if r['method'] in LEARNED]
  self.assertEqual(len(learned),440)
  for row in learned:self.assertIn(_job_for(row,jobs),jobs)
 def test_046_evaluation_method_counts(self):
  rows=core.read_json(core.EVALUATION_MANIFEST)['passes'];self.assertEqual(Counter(x['method'] for x in rows),Counter({'target_only_vanilla':80,'target_only_graph':80,'ordinary_source_adapted':80,'pooled_graph':80,'grl_l50_constant_adapted':80,'ordinary_source_graphgru':20,'grl_l50_constant_source_graphgru':20,'HA_TARGET':16,'HA_SOURCE':4,'persistence':4,'seasonal_naive':4}))
 def test_047_rng_namespace_preserved_from_r6(self):
  job=core.read_json(core.JOB_MANIFEST)['fits'][0]
  import hashlib
  expected=int.from_bytes(hashlib.sha256(f"final-v2.2-r6|{job['job_id']}|model-initialization|{job['seed']}".encode()).digest()[:8],'big')
  self.assertEqual(core.seed_for(job,'model-initialization'),expected)
 def test_048_explicit_binding_schemas(self):
  from . import bindings
  self.assertEqual((bindings.COMPLETION_SCHEMA,bindings.CHECKPOINT_SCHEMA),('final_v2_2_r7_r1.completion.2','final_v2_2_r7_r1.checkpoint.2'))
  self.assertEqual(bindings.grl_policy()['domain_map'],{'129':0,'194':1,'438':2,'467':3,'476':4,'532':5,'619':6,'658':7})
if __name__=='__main__':unittest.main(verbosity=2)
