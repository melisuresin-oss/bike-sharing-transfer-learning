from __future__ import annotations
import io,json,multiprocessing,os,subprocess,sys,tempfile,time,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from . import bindings,core,validation

def _wait(path:Path,seconds=15):
    end=time.time()+seconds
    while time.time()<end:
        if path.exists(): return
        time.sleep(.02)
    raise TimeoutError(str(path))

def _atomic_writer(root_rel,event,queue,payload):
    from research.final_v2_2_r7_r1 import core as child_core
    child_core.OUT=root_rel; event.wait()
    try:
        child_core.atomic_bytes(root_rel+"/winner.bin",payload.encode()); queue.put(("published",payload))
    except FileExistsError: queue.put(("refused",payload))

class R7R1ExecutionIntegrityTests(unittest.TestCase):
    def _env(self):
        env=dict(os.environ); env["PYTHONDONTWRITEBYTECODE"]="1"; return env
    def _synthetic_config(self,root:Path):
        cfg={"purpose":"R7_NON_SCIENTIFIC_PRODUCTION_PATH_MICRO_TEST","output_root":str(root),"hold_claim_seconds":2,
             "jobs":[{"job_id":"synthetic_concurrency_job","seed":17,"updates":1,"depends_on":None}]}
        path=root.parent/(root.name+".json"); path.write_text(json.dumps(cfg),encoding="utf-8"); return path
    def test_001_two_controllers_one_global_owner(self):
        with tempfile.TemporaryDirectory(dir=core.path("tmp")) as d:
            root=Path(d)/"out"; cfg=self._synthetic_config(root)
            code="from research.final_v2_2_r7_r1.controller import synthetic_dispatch;import sys;synthetic_dispatch(sys.argv[1])"
            p1=subprocess.Popen([sys.executable,"-c",code,str(cfg)],cwd=core.ROOT,env=self._env(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            _wait(root/"controller.owner.json")
            p2=subprocess.run([sys.executable,"-c",code,str(cfg)],cwd=core.ROOT,env=self._env(),capture_output=True,text=True)
            self.assertNotEqual(p2.returncode,0); self.assertEqual(p1.wait(timeout=45),0)
            self.assertEqual(json.loads((root/"controller_result.json").read_text())["completed"],1)
            self.assertEqual(len(list(root.glob("synthetic_concurrency_job.*.started.json"))),1)
            self.assertEqual(json.loads((root/"synthetic_concurrency_job.completion.json").read_text())["updates"],1)
    def test_002_two_workers_one_job_claim(self):
        with tempfile.TemporaryDirectory(dir=core.path("tmp")) as d:
            root=Path(d)/"out"; cfg=self._synthetic_config(root)
            base=[sys.executable,"-m","research.final_v2_2_r7_r1.worker","--job-id","synthetic_concurrency_job","--synthetic-config",str(cfg),"--device","cpu"]
            p1=subprocess.Popen(base+["--attempt-uuid",str(uuid.uuid4())],cwd=core.ROOT,env=self._env(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            _wait(root/"locks"/"synthetic_concurrency_job.owner.json")
            p2=subprocess.run(base+["--attempt-uuid",str(uuid.uuid4())],cwd=core.ROOT,env=self._env(),capture_output=True,text=True)
            self.assertNotEqual(p2.returncode,0); self.assertEqual(p1.wait(timeout=45),0)
            self.assertEqual(len(list(root.glob("synthetic_concurrency_job.*.started.json"))),1)
            self.assertEqual(json.loads((root/"synthetic_concurrency_job.completion.json").read_text())["updates"],1)
    def test_003_atomic_no_clobber_two_writers(self):
        with tempfile.TemporaryDirectory(dir=core.path("tmp")) as d:
            root=Path(d); rel=root.relative_to(core.ROOT).as_posix(); ctx=multiprocessing.get_context("spawn")
            event=ctx.Event(); queue=ctx.Queue(); processes=[ctx.Process(target=_atomic_writer,args=(rel,event,queue,p)) for p in ("alpha","beta")]
            for process in processes: process.start()
            event.set()
            for process in processes: process.join(15); self.assertEqual(process.exitcode,0)
            outcomes=sorted(queue.get(timeout=2)[0] for _ in processes)
            self.assertEqual(outcomes,["published","refused"]); self.assertIn((root/"winner.bin").read_bytes(),(b"alpha",b"beta"))

    def test_004_dispatch_accepts_validated_parent_dependency(self):
        from .controller import dispatch
        with tempfile.TemporaryDirectory(dir=core.path("tmp")) as d:
            job={"job_id":"synthetic_child","depends_on":"synthetic_parent"}; launched=[]
            done=dispatch([job],lambda j,a:[sys.executable,"-c","pass"],lambda j:launched.append(j["job_id"]),Path(d),lambda value:None,4,{"synthetic_parent"})
            self.assertEqual((done,launched),({"synthetic_child"},["synthetic_child"]))
    def _fixture(self,job):
        import torch
        from research.models.common import NeuralModelConfig
        from research.models.graph_gru import GraphGRU
        from research.models.vanilla_gru import VanillaGRU
        package_sha=core.sha(core.PACKAGE_MANIFEST); parent=None
        if job["category"]=="adaptation":
            parent_job=next(j for j in core.read_json(core.JOB_MANIFEST)["fits"] if j["job_id"]==job["depends_on"])
            parent_path=core.path(validation.completion_path(parent_job["job_id"]))
            parent_record=json.loads(parent_path.read_text()) if parent_path.exists() else self._fixture(parent_job)
            parent=parent_record["output_checkpoint"]
        attempt=str(uuid.uuid4()); scientific=bindings.expected(job,package_sha,parent,"cuda")
        attempt_rel=f"{core.OUT}/attempts/{job['job_id']}/{attempt}"
        started={"status":"STARTED_FROM_UPDATE_0","job":job,"attempt_uuid":attempt,"package_manifest_sha256":package_sha,
            "scientific_execution_binding_sha256":scientific["scientific_execution_binding_sha256"],"data_binding":scientific["data"],
            "runtime":scientific["runtime"],"started_utc":"synthetic-fixture","partial_resume":False}
        core.atomic_json(attempt_rel+"/started.json",started)
        if job["category"]=="source" and job["method"].startswith("grl_"):
            from .grl_final import FinalSourceInvariantGraphGRU
            model=FinalSourceInvariantGraphGRU(); forecast_state=model.forecast.state_dict()
        else:
            cfg=NeuralModelConfig(model_type="vanilla_gru" if job["architecture"].startswith("VGRU") else "graph_gru",
                hidden_size=32,dropout=0.0,output_mode="log1p_target")
            model=VanillaGRU(cfg) if cfg.model_type=="vanilla_gru" else GraphGRU(cfg); forecast_state=None
        lr=1e-3 if job["category"] in ("source","pooled") else 2e-4
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); opt_state=opt.state_dict()
        first=opt_state["param_groups"][0]["params"][0]; opt_state["state"]={first:{"step":torch.tensor(float(job["updates"]))}}
        auth=scientific["authorities"]
        metadata={"schema_version":bindings.CHECKPOINT_SCHEMA,"job_id":job["job_id"],"job_manifest_sha256":auth["job_manifest_sha256"],
            "package_manifest_sha256":package_sha,"code_manifest_sha256":auth["code_manifest_sha256"],
            "implementation_contract_sha256":auth["implementation_contract_sha256"],"final_data_manifest_sha256":auth["final_data_manifest_sha256"],
            "protocol_hashes":auth["protocol_hashes"],"grl_clarification_hashes":auth["grl_clarification_hashes"],
            "scientific_execution_binding":scientific,"scientific_execution_binding_sha256":scientific["scientific_execution_binding_sha256"],
            "seed":job["seed"],"method":job["method"],"model_binding":scientific["model"],"data_binding":scientific["data"],
            "input_checkpoint":parent,"runtime":scientific["runtime"],"optimizer":scientific["optimizer"],
            "expected_updates":job["updates"],"completed_updates":job["updates"],"early_stopping":False,"partial_resume":False}
        payload={"schema_version":bindings.CHECKPOINT_SCHEMA,"metadata":metadata,"model_state":model.state_dict(),"optimizer_state":opt_state}
        if forecast_state is not None: payload["forecast_state"]=forecast_state
        buf=io.BytesIO(); torch.save(payload,buf); checkpoint=core.atomic_bytes(attempt_rel+"/checkpoint.pt",buf.getvalue())
        record={"schema_version":bindings.COMPLETION_SCHEMA,"status":"COMPLETED_VALID_FIT","success":True,
            "job_id":job["job_id"],"job_manifest_sha256":auth["job_manifest_sha256"],"package_manifest_sha256":package_sha,
            "code_manifest_sha256":auth["code_manifest_sha256"],"implementation_contract_sha256":auth["implementation_contract_sha256"],
            "final_data_manifest_sha256":auth["final_data_manifest_sha256"],"protocol_hashes":auth["protocol_hashes"],
            "grl_clarification_hashes":auth["grl_clarification_hashes"],"scientific_execution_binding":scientific,
            "scientific_execution_binding_sha256":scientific["scientific_execution_binding_sha256"],"method":job["method"],
            "phase":job["category"],"seed":job["seed"],"target":job["target_city_id"],"budget":job["budget"],"attempt_uuid":attempt,
            "model_binding":scientific["model"],"data_binding":scientific["data"],"input_checkpoint":parent,
            "output_checkpoint":checkpoint,"optimizer":scientific["optimizer"],"optimizer_update_count":job["updates"],
            "runtime_determinism":scientific["runtime"],"artifact_hashes":{"checkpoint":checkpoint["sha256"]},
            "diagnostics":[{"step":job["updates"]}],"started_record":core.entry(attempt_rel+"/started.json"),
            "completed_utc":"synthetic-fixture","early_stopping":False,"partial_resume":False,
            "final_target_labels_accessed":False,"final_predictions_generated":False}
        core.atomic_json(validation.completion_path(job["job_id"]),record); return record

    def test_005_fabricated_completions_all_rejected(self):
        import torch
        jobs=core.read_json(core.JOB_MANIFEST)["fits"]
        ordinary=next(j for j in jobs if j["method"]=="ordinary_source_graphgru" and j["seed"]==17)
        grl=next(j for j in jobs if j["method"]=="grl_l50_constant_source_graphgru" and j["seed"]==17)
        target=next(j for j in jobs if j["method"]=="target_only_graph" and j["seed"]==17 and j["target_city_id"]==195 and j["budget"]=="1")
        adapted=next(j for j in jobs if j["method"]=="ordinary_source_adapted" and j["seed"]==17 and j["target_city_id"]==195 and j["budget"]=="1")
        cases=[("missing package hash",ordinary,lambda r:r.pop("package_manifest_sha256")),
            ("wrong package hash",ordinary,lambda r:r.__setitem__("package_manifest_sha256","0"*64)),
            ("wrong code",ordinary,lambda r:r.__setitem__("code_manifest_sha256","0"*64)),
            ("wrong method",ordinary,lambda r:r.__setitem__("method","pooled_graph")),
            ("wrong phase",ordinary,lambda r:r.__setitem__("phase","pooled")),("wrong seed",ordinary,lambda r:r.__setitem__("seed",29)),
            ("wrong target",target,lambda r:r.__setitem__("target",199)),("wrong budget",target,lambda r:r.__setitem__("budget","7")),
            ("wrong architecture",ordinary,lambda r:r["model_binding"].__setitem__("architecture","VGRU_H032_D00")),
            ("wrong GRL lambda",grl,lambda r:r["model_binding"]["grl_source_policy"].__setitem__("lambda",0.1)),
            ("wrong GRL cardinality",grl,lambda r:r["model_binding"]["grl_source_policy"].__setitem__("output_dim",7)),
            ("wrong target snapshot",target,lambda r:r["data_binding"].__setitem__("target_snapshot",{"sha256":"0"*64})),
            ("wrong source pool",ordinary,lambda r:r["data_binding"].__setitem__("source_snapshots",[])),
            ("wrong graph",target,lambda r:r["data_binding"].__setitem__("target_graph",{"sha256":"0"*64})),
            ("wrong optimizer",ordinary,lambda r:r["optimizer"].__setitem__("name","SGD")),
            ("wrong LR",ordinary,lambda r:r["optimizer"].__setitem__("learning_rate",9.0)),
            ("wrong updates",ordinary,lambda r:r.__setitem__("optimizer_update_count",1)),
            ("wrong parent checkpoint",adapted,lambda r:r.__setitem__("input_checkpoint",{"sha256":"0"*64})),
            ("altered completion",ordinary,lambda r:r.__setitem__("status","ALTERED")),
            ("completion copied from another job",ordinary,lambda r:r.__setitem__("job_id","ordinary_source_seed-29"))]
        for name,job,mutate in cases:
            with self.subTest(case=name),tempfile.TemporaryDirectory(dir=core.path("tmp")) as d:
                rel=Path(d).relative_to(core.ROOT).as_posix()
                with patch.object(core,"OUT",rel):
                    record=self._fixture(job); path=core.path(validation.completion_path(job["job_id"])); altered=json.loads(path.read_text()); mutate(altered); path.write_text(json.dumps(altered),encoding="utf-8")
                    with self.assertRaises(Exception): validation.validate_existing(core.sha(core.PACKAGE_MANIFEST))
        with self.subTest(case="valid checkpoint bytes but mismatched metadata"),tempfile.TemporaryDirectory(dir=core.path("tmp")) as d:
            rel=Path(d).relative_to(core.ROOT).as_posix()
            with patch.object(core,"OUT",rel):
                record=self._fixture(ordinary); cp=core.path(record["output_checkpoint"]["path"]); payload=torch.load(cp,map_location="cpu",weights_only=False)
                payload["metadata"]["method"]="pooled_graph"; torch.save(payload,cp)
                record["output_checkpoint"]["bytes"]=cp.stat().st_size; record["output_checkpoint"]["sha256"]=core.sha256_path(cp)
                record["artifact_hashes"]["checkpoint"]=record["output_checkpoint"]["sha256"]
                core.path(validation.completion_path(ordinary["job_id"])).write_text(json.dumps(record),encoding="utf-8")
                with self.assertRaises(Exception): validation.validate_existing(core.sha(core.PACKAGE_MANIFEST))

if __name__=="__main__": unittest.main(verbosity=2)
