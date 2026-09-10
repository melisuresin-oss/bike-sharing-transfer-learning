from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.final_v2_2_r7_r1 import metrics_bootstrap as frozen
from . import authority, authorization, core, materializer, scorer, staging, validator


class PB1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, payload=b"x"):
        path=self.root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);return path

    def test_001_authorization_precedes_target_read_event(self):
        stage=self.root/'stage';stage.mkdir();out=stage/core.PB1_OUT;out.mkdir(parents=True)
        auth=self.write('stage/auth.json',b'auth');target_auth=self.write('stage/target_authority.json',b'{}');package=self.write('stage/package.json',b'{}')
        cap=authorization.PhaseBCapability(core.sha256_file(auth),core.sha256_file(target_auth),'c','p',str(stage),str(self.root),'e','2026-01-01T00:00:00+00:00')
        arrays={'city_id':np.array(core.TARGETS,np.int64),'station_id':np.arange(4,dtype=np.int64),'forecast_origin_us':np.full(4,core.HT,np.int64),'opaque_label_join_key':np.arange(4,dtype=np.uint64),'count':np.zeros(4,np.int64),'label_valid':np.ones(4,bool)}
        def assert_event(*args,**kwargs):
            self.assertTrue((out/'first_label_access.json').is_file());return [self.write('t')]*6,[self.write('s')]*5,[]
        with patch.object(authorization,'validate',return_value=cap),patch.object(materializer,'verify_raw_sources_after_authorization',side_effect=assert_event),patch.object(materializer,'_materialize_arrays',return_value=arrays):
            materializer.materialize(auth,stage,self.root,target_auth,package,'x')

    def _phase_a_fixture(self):
        stage=self.root/'stage';(stage/core.PHASE_A_OUT/'predictions').mkdir(parents=True)
        prediction={'schema_version':'final_v2_2_r7_r1.predictions.1','count':468,'final_labels_joined':False}
        pp=stage/core.PHASE_A_OUT/'predictions/prediction_manifest.json';pp.write_text(json.dumps(prediction),encoding='utf-8')
        bound={'prediction_manifest':{'sha256':core.sha256_file(pp)}}
        commitment={'schema_version':'final_v2_2_r7_r1.prediction_commitment.1','status':'PHASE_A_PREDICTIONS_STRONGLY_COMMITTED','bound':bound,'bound_root_sha256':core.canonical_hash(bound)}
        cp=stage/core.PHASE_A_OUT/'predictions/prediction_commitment.json';cp.write_text(json.dumps(commitment),encoding='utf-8')
        return stage,cp,pp

    def test_002_invalid_commitment_rejected(self):
        stage,cp,pp=self._phase_a_fixture();cp.write_bytes(b'bad')
        with patch.object(core,'R7_PATHS',{}),patch.object(core,'PREDICTION_COMMITMENT_SHA','0'*64),patch.object(core,'PREDICTION_MANIFEST_SHA',core.sha256_file(pp)):
            with self.assertRaises(PermissionError):authorization._phase_a_bindings(stage)

    def test_003_invalid_prediction_manifest_rejected(self):
        stage,cp,pp=self._phase_a_fixture();pp.write_bytes(b'bad')
        with patch.object(core,'R7_PATHS',{}),patch.object(core,'PREDICTION_COMMITMENT_SHA',core.sha256_file(cp)),patch.object(core,'PREDICTION_MANIFEST_SHA','0'*64):
            with self.assertRaises(PermissionError):authorization._phase_a_bindings(stage)

    def _manifest(self):
        files=[]
        for i in range(11):
            role='trips' if i<6 else 'station_status';files.append({'path':f'x/{role}/f{i}.parquet','bytes':i+1,'sha256':hashlib.sha256(str(i).encode()).hexdigest()})
        return {'raw_bindings':{'dataset':'synthetic','revision':'r','files':files}}

    def test_004_invalid_target_authority_rejected(self):
        manifest=self.write('manifest.json',json.dumps(self._manifest()).encode());expected=core.sha256_file(manifest)
        output=self.write('authority.json',b'{}')
        with patch.object(core,'FINAL_DATA_SHA',expected):
            with self.assertRaises(PermissionError):authority.validate(output,manifest)

    def test_005_path_substitution_rejected(self):
        with self.assertRaises(ValueError):core.canonical_relative('../x')

    def test_006_symlink_substitution_rejected(self):
        target=self.write('real',b'x');link=self.root/'link'
        try:
            os.symlink(target,link)
        except OSError:
            # Windows test accounts may not hold SeCreateSymbolicLinkPrivilege.
            # Exercise the same explicit rejection branch without weakening it.
            with patch.object(Path,'is_symlink',return_value=True):
                with self.assertRaises(PermissionError):core.no_symlink_path(target)
        else:
            with self.assertRaises(PermissionError):core.no_symlink_path(link)

    def test_007_modified_raw_file_rejected(self):
        path=self.write('raw',b'a');expected={'path':'raw','bytes':1,'sha256':core.sha256_file(path),'role':'trips'};path.write_bytes(b'b')
        with self.assertRaises(PermissionError):materializer._hash_bound_file(path,expected)

    def test_008_invalid_capability_rejected(self):
        with self.assertRaises(PermissionError):materializer.verify_raw_sources_after_authorization(object(),{'files':[]},self.root)

    def test_009_toctou_mutation_rejected(self):
        path=self.write('raw',b'a');expected={'path':'raw','bytes':1,'sha256':core.sha256_file(path),'role':'trips'}
        with self.assertRaises(PermissionError):materializer._hash_bound_file(path,expected,lambda p:p.write_bytes(b'bb'))

    def test_010_duplicate_publication_rejected(self):
        path=self.write('published',b'a')
        with self.assertRaises(FileExistsError):core.atomic_bytes(path,b'b')

    def _prediction_fixture(self,kind='valid',count=468,duplicate=False):
        stage=self.root/'stage';stage.mkdir();payload=stage/'payload.npz'
        arrays={'city_id':np.array([195],np.int64),'station_id':np.array([1],np.int64),'forecast_origin_us':np.array([core.HT],np.int64),'prediction':np.array([1],np.float32),'availability':np.array([True],bool),'opaque_label_join_key':np.array([195<<32],np.uint64)}
        if kind=='malformed':arrays.pop('availability')
        if kind=='key':arrays['station_id']=np.array([2],np.int64)
        payload.write_bytes(core.deterministic_npz(arrays));entry=core.entry(payload,relative_to=stage)
        plans=[];items=[]
        for i in range(count):
            eid='e0' if duplicate else f'e{i}'
            plans.append({'evaluation_id':eid,'target_city_id':195,'method':'m','budget':'0','seed':None,'reuse_labels':['0']})
            items.append({'evaluation_item_id':eid,'payload':entry})
        (stage/'eval.json').write_text(json.dumps({'passes':plans}),encoding='utf-8');(stage/'pred.json').write_text(json.dumps({'items':items}),encoding='utf-8')
        labels={195:{'city_id':np.array([195],np.int64),'station_id':np.array([1],np.int64),'forecast_origin_us':np.array([core.HT],np.int64),'opaque_label_join_key':np.array([195<<32],np.uint64),'count':np.array([1],np.int64),'label_valid':np.array([True],bool)}}
        return stage,labels

    def _load_fixture(self,stage,labels):
        paths={'evaluation_manifest':('eval.json','x')}
        with patch.object(core,'R7_PATHS',paths),patch.object(core,'PHASE_A_OUT',Path('.')):
            (stage/'predictions').mkdir(exist_ok=True);os.replace(stage/'pred.json',stage/'predictions/prediction_manifest.json')
            return scorer._load_evaluations(stage,labels)

    def test_011_missing_prediction_rejected(self):
        stage,labels=self._prediction_fixture(count=467)
        with self.assertRaises(ValueError):self._load_fixture(stage,labels)

    def test_012_duplicate_prediction_rejected(self):
        stage,labels=self._prediction_fixture(duplicate=True)
        with self.assertRaises(ValueError):self._load_fixture(stage,labels)

    def test_013_malformed_prediction_rejected(self):
        stage,labels=self._prediction_fixture(kind='malformed')
        with self.assertRaises(ValueError):self._load_fixture(stage,labels)

    def test_014_key_mismatch_rejected(self):
        stage,labels=self._prediction_fixture(kind='key')
        with self.assertRaises(ValueError):self._load_fixture(stage,labels)

    def test_015_label_valid_preserves_zero_and_unknown(self):
        path=self.write('labels.npz',core.deterministic_npz({'city_id':np.repeat(np.array(core.TARGETS,np.int64),2),'station_id':np.tile(np.array([1,2],np.int64),4),'forecast_origin_us':np.tile(np.array([core.HT,core.HT],np.int64),4),'opaque_label_join_key':np.arange(8,dtype=np.uint64),'count':np.tile(np.array([0,-1],np.int64),4),'label_valid':np.tile(np.array([True,False],bool),4)}))
        with patch.object(core,'EVAL_HOURS',1),patch.object(core,'HE',core.HT+core.HOUR):
            result=scorer._load_labels(path)
        self.assertEqual(int(result[195]['count'][0]),0);self.assertTrue(bool(result[195]['label_valid'][0]));self.assertFalse(bool(result[195]['label_valid'][1]))

    def test_016_metric_correctness(self):
        label=np.array([1.,3.]);pred=np.array([2.,1.]);station=np.array([1,1]);hour=np.array([0,1]);valid=np.ones(2,bool)
        point=scorer._point(scorer._sufficient(label,pred,valid,station,hour,np.array([1])))
        self.assertTrue(np.allclose(point,[1.5,math.sqrt(2.5),.75,1.5]))

    def test_017_wape_zero_denominator_undefined(self):
        stats=scorer._sufficient(np.zeros(2),np.ones(2),np.ones(2,bool),np.array([1,1]),np.array([0,1]),np.array([1]))
        self.assertTrue(np.isnan(scorer._point(stats)[2]))

    def test_018_five_seed_arithmetic_mean(self):
        self.assertEqual(frozen.aggregate_five_seeds([1,2,3,4,5])['mean'],3)

    def test_019_population_sd_denominator_five(self):
        self.assertAlmostEqual(frozen.aggregate_five_seeds([1,2,3,4,5])['population_sd'],np.std([1,2,3,4,5],ddof=0))

    def test_020_equal_city_macro(self):
        self.assertEqual(frozen.equal_city_macro({195:1,199:2,237:3,617:4}),2.5)

    def test_021_deterministic_bootstrap_rng(self):
        self.assertEqual(frozen.block_draws(195),frozen.block_draws(195));self.assertEqual(len(frozen.block_draws(195)),2000)

    def test_022_exact_block_semantics(self):
        self.assertEqual(frozen.BLOCK_LENGTHS,(168,)*9+(53,));self.assertEqual(len(frozen.block_indices(tuple(range(10)))),1565)

    def test_023_terminal_block_semantics(self):
        indices=frozen.block_indices((9,)*10);self.assertEqual(len(indices),530);self.assertEqual((min(indices),max(indices)),(1512,1564))

    def test_024_undefined_bootstrap_not_redrawn(self):
        reps=np.zeros((2000,4));reps[3,0]=np.nan
        self.assertEqual(scorer._intervals(reps)['mae']['reason'],'undefined_replicate')

    def test_025_paired_gain_sign(self):
        n=24;station=np.ones(n,np.int64);hour=np.arange(n);label=np.ones(n);valid=np.ones(n,bool)
        method=scorer.EvaluationData('m',195,'m','0',None,station,hour,label,valid,np.ones(n),valid)
        reference=scorer.EvaluationData('r',195,'r','0',None,station,hour,label,valid,np.zeros(n),valid)
        result,_=scorer._paired(195,{None:method},{None:reference},scorer._draw_weights(195))
        self.assertEqual(result['point_gain']['mae'],1.0)

    def _frozen_fixture(self):
        stage=self.root/'stage';out=stage/core.PB1_OUT;out.mkdir(parents=True)
        data=self.write('stage/data.bin',b'data');data_entry=core.entry(data,relative_to=stage)
        payload={'schema_version':core.PAYLOAD_MANIFEST_SCHEMA,'files':[data_entry]};payload_path=out/'frozen_result_payload_manifest.json';payload_path.write_text(json.dumps(payload),encoding='utf-8')
        archive=out/'final_result_archive.zip'
        with zipfile.ZipFile(archive,'w') as z:z.write(data,data_entry['path'])
        bound={'frozen_payload_manifest':core.entry(payload_path,relative_to=stage)}
        manifest={'schema_version':core.FINAL_MANIFEST_SCHEMA,'status':'FINAL_V2_2_RESULTS_FROZEN','bound_artifacts':bound,'final_result_archive':core.entry(archive,relative_to=stage)}
        manifest_path=out/'final_result_manifest.json';manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
        return stage,manifest_path,archive

    def test_026_result_manifest_tampering_rejected(self):
        stage,manifest,archive=self._frozen_fixture();value=json.loads(manifest.read_text());value['status']='bad';manifest.write_text(json.dumps(value))
        with self.assertRaises(PermissionError):validator.validate_frozen(stage)

    def test_027_final_archive_tampering_rejected(self):
        stage,manifest,archive=self._frozen_fixture()
        with archive.open('ab') as handle:handle.write(b'x')
        with self.assertRaises(PermissionError):validator.validate_frozen(stage)

    def test_028_phase_a_source_archive_no_write(self):
        source=self.root/'source.zip'
        with zipfile.ZipFile(source,'w') as z:z.writestr('x/file',b'data')
        before=core.sha256_file(source);destination=self.root/'destination';destination.mkdir()
        staging._safe_extract(source,destination,expected_sha=before)
        self.assertEqual(before,core.sha256_file(source))


if __name__ == '__main__':
    unittest.main(verbosity=2)
