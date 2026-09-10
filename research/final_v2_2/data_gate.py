"""Phase-A final data construction using the validated V2.2 causal primitives."""
from __future__ import annotations
import hashlib, json, os, uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
from research.v2_2.contract import Contract, HOUR, canonical_bytes
from research.v2_2.history import StatusEvent, StatusIndex, IdealizedCountFeed, CausalHistory, coverage
from research.v2_2.snapshots import FitRequest, FitRow, FitSnapshot, seal_training_keys
from research.v2_2.artifacts import ArtifactIdentity
from research.graphs.geographic_graph import build_geographic_graph
from . import core

LAGS=tuple(range(1,25))+(168,)
PHASE_A_ROOT="processed/protocol_v2_2/final_phase_a"
FINAL_MANIFEST="processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json"
TRIPS_GLOB="feasibility_test/full_audit/data/trips/*.parquet"
STATUS_GLOB="feasibility_test/full_audit/coverage/data/station_status/*.parquet"

def atomic_npz(relative, **arrays):
 p=core.repo_path(relative);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(relative)
 pending=p.with_name(p.name+".pending-"+uuid.uuid4().hex+".npz")
 np.savez_compressed(pending,**arrays)
 with pending.open("rb") as f:os.fsync(f.fileno())
 pending.replace(p)
 return {"path":relative,"bytes":p.stat().st_size,"sha256":core.sha256_path(p)}

def verify_raw_bindings():
 trip=core.read_json("feasibility_test/full_audit/source_manifest.json")
 stat=core.read_json("feasibility_test/full_audit/coverage/station_status_source_manifest.json")
 if trip["revision"]!=stat["revision"]:raise RuntimeError("upstream revision mismatch")
 entries=[]
 for x in trip["files"]:
  if not x["path"].startswith("trips/"):continue
  rel="feasibility_test/full_audit/data/"+x["path"]
  if core.sha(rel)!=x["sha256"]:raise RuntimeError("raw trip hash mismatch: "+rel)
  entries.append({"path":rel,"sha256":x["sha256"],"bytes":x["size"]})
 for x in stat["files"]:
  rel="feasibility_test/full_audit/coverage/data/station_status/"+x["path"]
  if core.sha(rel)!=x["sha256"]:raise RuntimeError("raw status hash mismatch: "+rel)
  entries.append({"path":rel,"sha256":x["sha256"],"bytes":x["size"]})
 return {"dataset":trip["source"],"revision":trip["revision"],"files":entries}

def _queries(con, city, ids, h0, he):
 sqlids=",".join(map(str,ids))
 sq=f"SELECT DISTINCT station_id::BIGINT,epoch_us(to_timestamp(time)) FROM read_parquet('{STATUS_GLOB}') WHERE station_id IN ({sqlids}) AND time IS NOT NULL AND isfinite(time) ORDER BY 1,2"
 tq=f"SELECT station_id_start::BIGINT,epoch_us(date_trunc('hour',to_timestamp(time_start))),count(*)::BIGINT FROM read_parquet('{TRIPS_GLOB}') WHERE city_id={city} AND station_id_start IN ({sqlids}) AND isfinite(station_id_start) AND station_id_start=trunc(station_id_start) AND time_start IS NOT NULL AND isfinite(time_start) AND time_start>={h0}/1000000.0 AND time_start<{he}/1000000.0 GROUP BY 1,2 ORDER BY 1,2"
 events=[StatusEvent(int(i),int(t)) for i,t in con.execute(sq).fetchall()]
 counts={(int(i),int(t)):int(v) for i,t,v in con.execute(tq).fetchall()}
 return events,counts

def _panel(engine, city, origins, role):
 stations=engine.contract.city(city);ids=np.array([s.station_id for s in stations],dtype=np.int64)
 H,N=len(origins),len(ids);x=np.zeros((H,24,N,2),np.float32);m=np.zeros((H,24,N),bool)
 w=np.zeros((H,N,2),np.float32);cal=np.zeros((H,6),np.float32);fp=np.empty(H,dtype='S64')
 for ti,t in enumerate(origins):
  hist=engine.for_inference(city,int(t));a=hist.arrays();x[ti]=a['x_hist'];m[ti]=a['m_hist'];w[ti]=a['x_week'];cal[ti]=a['x_calendar'];fp[ti]=hist.fingerprint().encode()
  if not np.array_equal(m[ti],x[ti,:,:,1].astype(bool)):raise RuntimeError("history mask drift")
 rel=f"{PHASE_A_ROOT}/city_{city}_{role}_predictors.npz"
 e=atomic_npz(rel,origin_us=origins,station_ids=ids,x_hist=x,m_hist=m,x_week=w,x_static=a['x_static'],x_calendar=cal,origin_feature_sha256=fp)
 return {**e,"city_id":city,"role":role,"origin_count":H,"station_count":N,"schema":{"origin_us":"int64","station_ids":"int64","x_hist":"float32[H,24,N,2]","m_hist":"bool[H,24,N]","x_week":"float32[H,N,2]","x_static":"float32[N,2]","x_calendar":"float32[H,6]"},"current_target_field_present":False},fp

def _snapshot(contract, status, counts, city, origins, fp, request):
 idpos={int(t):i for i,t in enumerate(origins)};rows=[]
 if request.budget!='zero':
  for t in origins:
   t=int(t)
   if not request.start<=t<request.cutoff:continue
   cv=coverage(contract,status,city,t,request.cutoff)
   for station,ok in zip(contract.city(city),cv.observed):
    if ok:rows.append(FitRow(city,station.station_id,t,int(counts.get((station.station_id,t),0)),fp[idpos[t]].decode()))
 snap=FitSnapshot(request,ArtifactIdentity.create(contract,'fit_snapshot',request.cutoff,request.city_ids),tuple(sorted(rows,key=lambda r:(r.city_id,r.station_id,r.origin))),'SCIENTIFIC_ARTIFACT')
 seal,digest=seal_training_keys(snap)
 rel=f"{PHASE_A_ROOT}/fit_snapshots/{request.kind}_city_{city}_{request.budget}_HF.npz"
 data=atomic_npz(rel,city_id=np.full(len(rows),city,np.int64),station_id=np.array([r.station_id for r in rows],np.int64),origin_us=np.array([r.origin for r in rows],np.int64),count=np.array([r.count for r in rows],np.int64),observed=np.ones(len(rows),bool),origin_feature_sha256=np.array([r.origin_feature_sha256 for r in rows],dtype='S64'))
 return {"request":asdict(request),"data":data,"training_key_seal":seal,"training_key_seal_sha256":digest,"row_count":len(rows)}

def build():
 core.verify_authority();raw=verify_raw_bindings();c=Contract();h0,hf,ht,he=(c.boundaries[k] for k in ('H0','HF','HT','HE'))
 if core.repo_path(FINAL_MANIFEST).exists():raise FileExistsError("append-only final manifest already exists")
 import duckdb
 con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='3GB'")
 fit_origins=np.arange(h0,hf,HOUR,dtype=np.int64);eval_origins=np.arange(ht,he,HOUR,dtype=np.int64)
 targets=[];source_snapshots=[]
 # Reuse already validated development predictor bytes for source refit; only cutoff-HF keys are new.
 for city in core.SOURCES:
  dev=core.repo_path(f"processed/protocol_v2_2/development/city_{city}_predictors.npz")
  z=np.load(dev,allow_pickle=False);orig=z['origin_us'];fp=z['origin_feature_sha256'];ids=list(map(int,z['station_ids']))
  events,counts=_queries(con,city,ids,h0,hf);status=StatusIndex(c,events)
  req=FitRequest.registered(c,phase='final',kind='source')
  # One city partition; the eight partitions are combined by the final source job loader.
  local=FitRequest('final','source','full',h0,hf,(city,))
  source_snapshots.append(_snapshot(c,status,counts,city,orig,fp,local))
 for city,name in core.TARGETS:
  stations=c.city(city);ids=[s.station_id for s in stations]
  events,counts=_queries(con,city,ids,h0,he);status=StatusIndex(c,events);feed=IdealizedCountFeed(c,ids,h0,he,counts);engine=CausalHistory(c,status,feed)
  fit_panel,fp=_panel(engine,city,fit_origins,'fitting')
  eval_panel,_=_panel(engine,city,eval_origins,'evaluation')
  adaptations=[]
  for budget in ('zero','1','7','30','full'):
   req=FitRequest.registered(c,phase='final',kind='adaptation',budget=budget,target_city=city)
   adaptations.append(_snapshot(c,status,counts,city,fit_origins,fp,req))
  graph=build_geographic_graph(np.array(ids,np.int64),np.array([s.latitude for s in stations]),np.array([s.longitude for s in stations]),k=4,
    roster_hash_sha256=core.canonical_hash(ids),coordinate_hash_sha256=core.canonical_hash([(s.station_id,s.latitude,s.longitude) for s in stations]))
  ge=atomic_npz(f"{PHASE_A_ROOT}/graphs/city_{city}_k04.npz",station_ids=np.array(ids,np.int64),adjacency=graph.normalized_adjacency)
  keys=atomic_npz(f"{PHASE_A_ROOT}/prediction_keys/city_{city}.npz",city_id=np.full(len(eval_origins)*len(ids),city,np.int64),station_id=np.tile(np.array(ids,np.int64),len(eval_origins)),forecast_origin_us=np.repeat(eval_origins,len(ids)))
  targets.append({"city_id":city,"city":name,"station_count":len(ids),"city_roster_sha256":c.city_roster_hash(city),"fitting_panel":fit_panel,"evaluation_panel":eval_panel,"graph":ge,"prediction_keys":keys,"adaptations":adaptations})
 con.close()
 manifest={"schema_version":"final_v2_2.data_gate.1","protocol_version":"2.2","scope":"FINAL_TARGETS",
  "phase_a_execution_data_constructed":True,"phase_b_targets_materialized":False,"prediction_commitment_required_for_phase_b":True,
  "final_predictions_generated":False,"final_metrics_computed":False,"final_experiment_started":False,"optimizer_updates":0,"model_forward_calls_on_final_labels":0,
  "targets":targets,"source_refit_partitions":source_snapshots,"raw_bindings":raw,"fixed_global_cohort_sha256":c.roster_sha256,
  "v2_2_specification_sha256":"c85c8fdbcab26e7239bfb4528b570b720e7d31ea5933934d9258b63ad90f9277",
  "causal_observability_implementation":{"path":"research/v2_2/history.py","sha256":core.sha("research/v2_2/history.py")},
  "builder":{"path":"research/final_v2_2/data_gate.py","sha256":core.sha("research/final_v2_2/data_gate.py")},
  "boundaries_us":core.BOUNDARIES,"evaluation_origin_count":len(eval_origins),"embargo_hours":(ht-hf)//HOUR,"lags":list(LAGS),
  "phase_a_evaluation_schema_excludes":["count","target","y","observed_target"],"created_utc":core.utc()}
 p=core.repo_path(FINAL_MANIFEST);p.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(manifest,indent=2,allow_nan=False).encode()+b'\n'
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 return {"path":FINAL_MANIFEST,"bytes":p.stat().st_size,"sha256":core.sha256_path(p)}
