from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from . import core,phase_b,data_gate,verify_data_gate,jobs

class GateR2Tests(unittest.TestCase):
 def test_data_verifier(self):self.assertEqual(verify_data_gate.verify()['status'],'PASS')
 def test_dag_unchanged(self):self.assertEqual((len(jobs.neural_jobs()),len(jobs.evaluations()),sum(len(x['reuse_labels']) for x in jobs.evaluations())),(410,468,660))
 def fixture(self,d):
  p=Path(d);(p/'commit').write_text('{}');(p/'label').write_text('{}')
  base={'status':'FINAL_LABEL_SCORING_AUTHORIZED','prediction_commitment_sha256':core.sha256_path(p/'commit'),'label_binding_sha256':core.sha256_path(p/'label'),'package_manifest_sha256':'p','job_manifest_sha256':'j','final_data_manifest_sha256':'d','implementation_contract_sha256':'i','final_protocol_markdown_sha256':core.AUTHORITY['FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md'],'final_protocol_json_sha256':core.AUTHORITY['final_evaluation_protocol_v2_2.json']}
  return p,base
 def refuse(self,mutate=None):
  with tempfile.TemporaryDirectory() as d:
   p,a=self.fixture(d)
   if mutate:mutate(a,p)
   (p/'auth').write_text(json.dumps(a))
   with self.assertRaises(PermissionError):phase_b.verify_gate(p/'auth',p/'commit',p/'label','p','j','d','i')
 def test_missing_authorization(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);[(p/n).write_text('{}') for n in ('auth','commit','label')]
   with self.assertRaises(PermissionError):phase_b.verify_gate(p/'auth',p/'commit',p/'label','p','j','d','i')
 def test_wrong_commitment(self):self.refuse(lambda a,p:a.update(prediction_commitment_sha256='x'))
 def test_altered_predictions(self):self.refuse(lambda a,p:(p/'commit').write_text('{"changed":1}'))
 def test_wrong_package(self):self.refuse(lambda a,p:a.update(package_manifest_sha256='x'))
 def test_wrong_data_manifest(self):self.refuse(lambda a,p:a.update(final_data_manifest_sha256='x'))
 def test_wrong_protocol(self):self.refuse(lambda a,p:a.update(final_protocol_json_sha256='x'))
 def test_wrong_implementation(self):self.refuse(lambda a,p:a.update(implementation_contract_sha256='x'))

if __name__=='__main__':unittest.main(verbosity=2)
