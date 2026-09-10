from __future__ import annotations
import builtins, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from . import bootstrap as b, core, firewall, jobs

class FinalV22Tests(unittest.TestCase):
 def test_authority_hashes(self):core.verify_authority()
 def test_exact_dag(self):
  x=jobs.neural_jobs();self.assertEqual(len(x),410);self.assertEqual(len({j['job_id'] for j in x}),410)
 def test_evaluation_accounting(self):
  x=jobs.evaluations();self.assertEqual(len(x),468);self.assertEqual(sum(map(lambda r:len(r['reuse_labels']),x)),660)
 def test_frozen_axes(self):self.assertEqual(core.SEEDS,(17,29,43,71,101));self.assertEqual(tuple(x[0] for x in core.TARGETS),(195,199,237,617));self.assertEqual(core.UPDATES,{"0":0,"1":100,"7":300,"30":600,"full":1200})
 def test_architectures_and_grl(self):
  x=jobs.neural_jobs();self.assertTrue(all(j['architecture'] in ('VGRU_H032_D00','GGRU_K04_H032_D00') for j in x));self.assertTrue(all('l50_constant' in j['method'] for j in x if j['method'].startswith('grl')))
 def test_checkpoint_reuse(self):
  x=jobs.neural_jobs();self.assertEqual(len([j for j in x if j['category']=='source']),10);self.assertTrue(all(j['depends_on'] for j in x if j['category']=='adaptation'))
 def test_bootstrap_blocks(self):self.assertEqual(sum(b.BLOCK_LENGTHS),1565);self.assertEqual(b.BLOCK_LENGTHS[-1],53)
 def test_bootstrap_seeds_and_draws(self):
  for city,s in core.BOOTSTRAP_SEEDS.items():self.assertEqual(b.city_seed(city),s);d=b.block_draws(city);self.assertEqual((len(d),len(d[0])),(2000,10));self.assertEqual(d,b.block_draws(city))
 def test_seed_summary_population_sd(self):self.assertEqual(b.seed_summary([0,1,2,3,4]),{'mean':2.0,'population_sd':2**.5})
 def test_metrics_missing_and_no_zero_fill(self):
  rows=[{'station_id':1,'hour':i,'label':2,'prediction':1,'label_valid':True,'prediction_available':True} for i in range(24)]
  rows += [{'station_id':1,'hour':24,'label':0,'prediction':0,'label_valid':True,'prediction_available':True},{'station_id':2,'hour':0,'label':None,'prediction':0,'label_valid':False,'prediction_available':True}]
  m=b.metrics(rows);self.assertEqual(m['valid_n'],25);self.assertEqual(m['station_macro_mae'],24/25)
 def test_unavailable_exclusion(self):self.assertEqual(b.metrics([{'label':1,'prediction':0,'label_valid':True,'prediction_available':False}])['valid_n'],0)
 def test_terminal_ci_undefined_not_redrawn(self):self.assertEqual(b.confidence_interval([1.]*1999+[None])['reason'],'undefined_replicate')
 def test_phase_a_path_rejection(self):
  for p in ('processed/protocol_v2_2/final_labels/x','SEALED_final/x'):self.assertRaises(PermissionError,firewall.phase_a_path,p)
 def test_import_opens_no_labels(self):
  original=builtins.open
  def guarded(file,*a,**k):
   firewall.reject_labelish_path(str(file));return original(file,*a,**k)
  with patch('builtins.open',guarded):__import__('research.final_v2_2.execution')
 def test_retry_policy(self):self.assertTrue(all(j['attempt_policy']=='fresh_uuid_from_update_0_no_partial_resume' for j in jobs.neural_jobs()))
 def test_phase_b_needs_capability(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a').write_text('{}');(p/'c').write_text('{}');(p/'l').write_text('{}')
   self.assertRaises(PermissionError,firewall.authorize_phase_b,p/'a',p/'c',p/'l')

if __name__=='__main__':unittest.main(verbosity=2)
