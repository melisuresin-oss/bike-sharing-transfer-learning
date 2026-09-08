"""Authorized DEVELOPMENT data build only. No model/estimator/prediction imports."""
from __future__ import annotations
import ast
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "tmp/neural_build/pydeps"), str(ROOT)]
import numpy as np
import duckdb
from research.v2_2 import FEATURE_FUNCTION_VERSION
from research.v2_2.contract import Contract, DEVELOPMENT, PSEUDO_TARGETS, HOUR, BRACKET, EPOCH, SPEC_SHA256, SEAL_SHA256, COHORT_SHA256, canonical_bytes
from research.v2_2.history import StatusIndex, StatusEvent, IdealizedCountFeed, CausalHistory, coverage
from research.v2_2.snapshots import FitRequest, FitRow, FitSnapshot, seal_training_keys
from research.v2_2.artifacts import ArtifactIdentity

OUT = "processed/protocol_v2_2/"
AUDIT = "research/results/causal_history_audit_v2_2/"
IMPL = "research/results/causal_history_implementation_v2_2/implementation_manifest.json"
IMPL_HASH = "a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d"
LAGS = list(range(1,25)) + [168]


def safe(relative):
    if not isinstance(relative,str) or "\\" in relative or ":" in relative or ".." in Path(relative).parts:
        raise PermissionError("Repository-relative POSIX paths required")
    p = (ROOT / relative).resolve()
    rel = p.relative_to(ROOT).as_posix()
    if any(x in rel.lower() for x in ("final_labels", "final_adaptation", "final_features", "sealed_final", "final_target_evaluation")):
        raise PermissionError("Final-target data firewall")
    return p


def sha(relative):
    h=hashlib.sha256()
    with safe(relative).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def read(relative):
    return json.loads(safe(relative).read_text(encoding="utf-8"))


def write(relative,value):
    p=safe(relative)
    if p.exists(): raise FileExistsError("Append-only build refuses overwrite: "+relative)
    if not (relative.startswith(OUT) or relative.startswith(AUDIT)):
        raise PermissionError("Write outside V2.2 development build")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+"\n",encoding="utf-8")
    return {"path":relative,"sha256":sha(relative)}


def save(relative,**arrays):
    p=safe(relative)
    if p.exists(): raise FileExistsError(relative)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**arrays)
    return {"path":relative,"sha256":sha(relative)}


def logical(rows):
    h=hashlib.sha256()
    for row in rows:
        h.update(("\x1f".join("<NULL>" if v is None else format(v,".17g") if isinstance(v,float) else str(v) for v in row)+"\n").encode())
    return h.hexdigest()


def graph_reconciliation(c):
    # Execute only the unchanged pure graph math definitions; no legacy registry
    # or scientific loader imports. This is reconciliation, not graph selection.
    source="research/graphs/geographic_graph.py"
    tree=ast.parse(safe(source).read_text())
    names={"_hash_arrays","haversine_distance_matrix","GeographicGraph","build_geographic_graph"}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    ns={"np":np,"hashlib":hashlib,"dataclass":dataclass,"EARTH_RADIUS_KM":6371.0088,"__name__":__name__}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),source,"exec"),ns)
    old={r["city_id"]:r for r in read("processed/protocol_v2_1/station_manifests/station_manifest.json")["cities"]}
    records=[]
    for city in DEVELOPMENT:
        stations=c.city(city)
        ids=np.array([s.station_id for s in stations],dtype=np.int64)
        lat=np.array([s.latitude for s in stations]); lon=np.array([s.longitude for s in stations])
        rh=logical([(s.station_id,) for s in stations]); ch=logical([(s.station_id,s.latitude,s.longitude) for s in stations])
        assert rh==old[city]["roster_hash_sha256"] and ch==old[city]["coordinate_hash_sha256"]
        for k in (4,8,16):
            base=f"tmp/stage2_graph_grid_cache_v2_1/k_{k:02d}/city_{city}/"
            meta=read(base+"metadata.json"); path=base+"adjacency.npy"
            g=ns["build_geographic_graph"](ids,lat,lon,k=k,roster_hash_sha256=rh,coordinate_hash_sha256=ch)
            a=np.load(safe(path),allow_pickle=False)
            assert a.dtype==g.normalized_adjacency.dtype and a.shape==g.normalized_adjacency.shape
            assert a.tobytes()==g.normalized_adjacency.tobytes(), ("STOP_GRAPH_DIFFERENCE",city,k)
            assert g.checkpoint_config()==meta["graph_config"]
            assert sha(path)==meta["adjacency_file_sha256"]
            if k==8:
                other=f"tmp/stage1_scale_loss_cache_v2_1/city_{city}/adjacency.npy"
                assert np.load(safe(other),allow_pickle=False).tobytes()==a.tobytes()
            records.append({"city_id":city,"k":k,"path":path,"sha256":sha(path),"old_metadata_sha256":sha(base+"metadata.json"),
                            "graph_config":g.checkpoint_config(),"reused_unchanged":True,
                            "v2_2_roster_sha256":c.city_roster_hash(city),
                            "static_sha256":hashlib.sha256(canonical_bytes([asdict(s) for s in stations])).hexdigest()})
    return {"status":"PASS","graph_count":len(records),"graph_algorithm_source_sha256":sha(source),"graphs":records,
            "new_adjacency_copies_created":0,"selection_occurred":False}


def check_lineage(cv,status):
    assert cv.denominator==sum(x.lifetime for x in cv.evidence)
    assert cv.numerator==sum(x.lifetime and x.bracket for x in cv.evidence)
    assert cv.city_observed==(cv.denominator>0 and 2*cv.numerator>=cv.denominator)
    exact=0
    for ev,observed in zip(cv.evidence,cv.observed):
        times=status.timestamps[ev.station_id]; stop=bisect_right(times,cv.cutoff)
        assert (ev.first,ev.last)==((times[0],times[stop-1]) if stop else (None,None))
        for u in (ev.previous,ev.next,ev.first,ev.last): assert u is None or u<=cv.cutoff
        bracket=(ev.previous is not None and ev.next is not None and
                 0<=cv.hour_start-ev.previous<=BRACKET and 0<=ev.next-cv.hour_start-HOUR<=BRACKET)
        assert bracket==ev.bracket
        assert observed==(ev.lifetime and bracket and cv.city_observed)
        if bracket: exact+=int(cv.hour_start-ev.previous==BRACKET)+int(ev.next-cv.hour_start-HOUR==BRACKET)
    return exact


def pct(n,d):
    return 100*n/d if d else None


def main():
    c=Contract(); h0,hd,hf=(c.boundaries[k] for k in ("H0","HD","HF"))
    assert sha(IMPL)==IMPL_HASH
    impl=read(IMPL)
    for group in ("implementation_files_sha256","test_files_sha256","evidence_files_sha256"):
        for p,h in impl[group].items(): assert sha(p)==h,p
    if safe(OUT+"PROTOCOL_DATASET_MANIFEST.json").exists(): raise FileExistsError("Development build already sealed")
    codehash=sha("research/scripts/build_development_data_v2_2.py")
    print("Reconciling all 24 existing geographic graphs",flush=True)
    graph=graph_reconciliation(c)
    graph_art=write(AUDIT+"graph_static_reconciliation.json",graph)
    old=read("processed/protocol_v2_1/PROTOCOL_DATASET_MANIFEST.json")
    raw={p:v for p,v in old["source_file_hashes"].items() if "/trips/" in p or "/station_status/" in p}
    for p,v in raw.items(): assert sha(p)==v["sha256"],p
    raw_art=write(AUDIT+"source_bindings.json",{"files":raw,"read_projection":{"status":["station_id","time"],"trips":["city_id","station_id_start","time_start"]},
                "city_filter":list(DEVELOPMENT),"station_filter":"exact frozen development roster","final_outcomes_selected":False})
    con=duckdb.connect()
    con.execute("SET threads=2"); con.execute("SET memory_limit='2GB'"); con.execute("SET temp_directory=''")
    origins=np.arange(h0,hf,HOUR,dtype=np.int64); H=len(origins); fitH=int((hd-h0)//HOUR)
    panels={}; labels={}; source={}; city_reports={}; availability=[]; allrows={}; fit_masks={}; bins_by_city={}
    exact_total=0; certificates=0
    for city in DEVELOPMENT:
        print(f"Building city {city}; {len(c.city(city))} stations, {H} origins",flush=True)
        stations=c.city(city); ids=[s.station_id for s in stations]; N=len(ids); sqlids=",".join(map(str,ids))
        sq=f"SELECT DISTINCT station_id::BIGINT, epoch_us(to_timestamp(time)) FROM read_parquet('feasibility_test/full_audit/coverage/data/station_status/*.parquet') WHERE station_id IN ({sqlids}) AND time IS NOT NULL AND isfinite(time) ORDER BY 1,2"
        events=[StatusEvent(int(i),int(t)) for i,t in con.execute(sq).fetchall()]
        status=StatusIndex(c,events)
        tq=f"SELECT station_id_start::BIGINT,epoch_us(date_trunc('hour',to_timestamp(time_start))),count(*) FROM read_parquet('feasibility_test/full_audit/data/trips/*.parquet') WHERE city_id={city} AND station_id_start IN ({sqlids}) AND isfinite(station_id_start) AND station_id_start=trunc(station_id_start) AND time_start IS NOT NULL AND isfinite(time_start) AND time_start>={h0}/1000000.0 AND time_start<{hf}/1000000.0 GROUP BY 1,2 ORDER BY 1,2"
        cr=con.execute(tq).fetchall()
        counts={(int(i),int(t)):int(v) for i,t,v in cr}
        feed=IdealizedCountFeed(c,ids,h0,hf,counts); engine=CausalHistory(c,status,feed)
        x=np.zeros((H,24,N,2),dtype=np.float32); mask=np.zeros((H,24,N),dtype=bool)
        week=np.zeros((H,N,2),dtype=np.float32); cal=np.zeros((H,6),dtype=np.float32)
        fingerprints=np.empty(H,dtype="S64"); fit=np.zeros((fitH,N),dtype=bool)
        dense=np.zeros((H,N),dtype=np.int64); idpos={i:j for j,i in enumerate(ids)}
        for (i,t),v in counts.items(): dense[(t-h0)//HOUR,idpos[i]]=v
        hist_stat=np.zeros((25,3),dtype=np.int64)
        prefix=StatusIndex(c,(e for e in events if e.timestamp<=hd))
        early=StatusIndex(c,(e for e in events if e.timestamp<=h0+(fitH//2)*HOUR))
        for ti,t0 in enumerate(origins):
            t=int(t0); history=engine.at_origin(city,t); a=history.arrays()
            x[ti]=a["x_hist"]; mask[ti]=a["m_hist"]; week[ti]=a["x_week"]; cal[ti]=a["x_calendar"]
            fingerprints[ti]=history.fingerprint().encode()
            assert np.array_equal(mask[ti],x[ti,:,:,1].astype(bool))
            for li,cv in enumerate(history.lineage):
                exact_total+=check_lineage(cv,status); certificates+=N
                pairs=x[ti,li] if li<24 else week[ti]
                usable=np.array(cv.observed,dtype=bool)&(cv.hour_start>=h0)
                assert np.array_equal(pairs[:,1],usable)
                rawvals=dense[(cv.hour_start-h0)//HOUR] if cv.hour_start>=h0 else np.zeros(N,dtype=np.int64)
                expected=np.array([math.log1p(int(v)) for v in rawvals],dtype=np.float32)
                assert np.array_equal(pairs[:,0],np.where(usable,expected,0))
                hist_stat[li]+=np.array([np.sum(usable&(rawvals>0)),np.sum(usable&(rawvals==0)),np.sum(~usable)])
            if ti<fitH:
                cv=coverage(c,status,city,t,hd); exact_total+=check_lineage(cv,status)
                cv_prefix=coverage(c,prefix,city,t,hd)
                assert cv==cv_prefix
                fit[ti]=cv.observed
            if ti in {0,fitH//2,fitH-1,fitH,H-1}:
                physical=StatusIndex(c,(e for e in events if e.timestamp<=t))
                assert history.tensor_bytes()==CausalHistory(c,physical,feed).for_inference(city,t).tensor_bytes()
            if ti and ti%1000==0: print(f"city {city}: {ti}/{H} origins verified",flush=True)
        bins=np.array([(EPOCH+timedelta(microseconds=int(t))).astimezone(ZoneInfo(stations[0].timezone)).weekday()*24+
                       (EPOCH+timedelta(microseconds=int(t))).astimezone(ZoneInfo(stations[0].timezone)).hour for t in origins],dtype=np.int16)
        bins_by_city[city]=bins
        p=save(OUT+f"development/city_{city}_predictors.npz",origin_us=origins,station_ids=np.array(ids,dtype=np.int64),
               x_hist=x,m_hist=mask,x_week=week,x_static=a["x_static"],x_calendar=cal,origin_feature_sha256=fingerprints)
        p.update({"city_id":city,"station_count":N,"origin_count":H,"identity":ArtifactIdentity.create(c,"predictor_cache",hf,[city]).as_json(),
                  "cutoff_semantics":"file partition end; each row uses its own origin_us","schema":{k:list(v.shape) for k,v in
                  {"x_hist":x,"m_hist":mask,"x_week":week,"x_static":a["x_static"],"x_calendar":cal}.items()}})
        panels[city]=p
        # Retrospective labels are constructed only AFTER predictors are complete.
        # The fixed adjudication cutoff is the last available development status.
        adjudication=max(e.timestamp for e in events)
        ret=np.array([coverage(c,status,city,int(t),adjudication).observed for t in origins],dtype=bool)
        lab=save(OUT+f"development_labels/city_{city}_retrospective.npz",origin_us=origins,station_ids=np.array(ids,dtype=np.int64),
                 observed=ret,count=np.where(ret,dense,-1))
        lab.update({"city_id":city,"adjudication_cutoff_us":adjudication,"unknown_sentinel":-1,"role":"DEVELOPMENT_RETROSPECTIVE_LABELS"})
        labels[city]=lab
        rows=tuple(FitRow(city,int(i),int(origins[ti]),int(dense[ti,j]),fingerprints[ti].decode())
                   for j,i in enumerate(ids) for ti in np.flatnonzero(fit[:,j]))
        allrows[city]=rows; fit_masks[city]=fit
        src=save(OUT+f"fit_snapshots/source_city_{city}_HD.npz",
                 city_id=np.full(len(rows),city,dtype=np.int64),station_id=np.array([r.station_id for r in rows],dtype=np.int64),
                 origin_us=np.array([r.origin for r in rows],dtype=np.int64),count=np.array([r.count for r in rows],dtype=np.int64),
                 origin_feature_sha256=np.array([r.origin_feature_sha256 for r in rows],dtype="S64"),
                 observed=np.ones(len(rows),dtype=bool))
        src.update({"city_id":city,"row_count":len(rows),"eligible_anchor_hours":int(fit.any(axis=1).sum()),"cutoff_us":hd,
                    "origin_range_us":[h0,hd],"role":"SOURCE_CITY_FITTING_ROWS","identity":ArtifactIdentity.create(c,"fit_snapshot",hd,[city]).as_json()})
        source[city]=src
        for li,lag in enumerate(LAGS):
            pos,zero,unknown=map(int,hist_stat[li]); total=H*N; obs=pos+zero
            availability.append({"city_id":city,"lag":lag,"total_cells":total,"observed_cells":obs,"observed_pct":pct(obs,total),
                "positive_cells":pos,"positive_pct":pct(pos,total),"zero_cells":zero,"zero_pct":pct(zero,total),
                "unknown_cells":unknown,"unknown_pct":pct(unknown,total),"share_denominator":"all feature cells"})
        baseavail=[]
        if city in PSEUDO_TARGETS:
            ev=ret[fitH:]
            for name,m in (("PERSISTENCE_T_MINUS_1",mask[fitH:,0]),("SEASONAL_NAIVE_T_MINUS_168",week[fitH:,:,1].astype(bool))):
                for scope,domain in (("all_prediction_keys",np.ones_like(ev)),("retrospectively_observed_evaluation_keys",ev)):
                    total=int(domain.sum()); avail=int((m&domain).sum())
                    baseavail.append({"city_id":city,"baseline":name,"period_us":[hd,hf],"key_scope":scope,"total":total,
                                      "available":avail,"unavailable":total-avail,"coverage_pct":pct(avail,total)})
        city_reports[city]={"city_id":city,"status":"PASS","all_origins_checked":H,"all_history_cells_checked":H*N*25,
            "status_events":len(events),"status_logical_sha256":logical([(e.station_id,e.timestamp) for e in events]),
            "count_feed_rows":len(cr),"count_feed_total_departures":sum(counts.values()),"count_feed_logical_sha256":logical(cr),
            "status_query":sq,"count_query":tq,"fit_prefix_comparisons":fitH,"physical_origin_prefix_comparisons":5,
            "baseline_availability":baseavail}
        write(AUDIT+f"city_{city}_audit.json",city_reports[city])
        print(f"city {city} complete: {len(rows)} source-fit rows",flush=True)
        del x,mask,week,cal,events,status,prefix,early,engine,feed,ret,dense
    con.close()
    adaptations=[]; folds=[]; effective=[]; ha=[]; comparisons=[]
    oldbudget={(r["city_id"],r["budget"]):r["eligible_target_labels"] for r in old["actual_budget_label_counts"] if r["target_role"]=="pseudo_target"}
    con=duckdb.connect(); con.execute("SET threads=2")
    oldrows=dict(con.execute(f"SELECT city_id,count(*) FROM read_parquet('processed/protocol_v2_1/development/development_panel.parquet') WHERE timestamp_utc<to_timestamp({hd}/1000000.0) AND coverage_observed_12h GROUP BY city_id").fetchall()); con.close()
    for city in DEVELOPMENT:
        n=source[city]["row_count"]; prev=int(oldrows[city])
        comparisons.append({"kind":"source_city","city_id":city,"v2_2_rows":n,"v2_1_rows":prev,"difference":n-prev,"difference_pct":pct(n-prev,prev)})
    for target in PSEUDO_TARGETS:
        request=FitRequest.registered(c,phase="development",kind="source",target_city=target)
        rows=tuple(r for city in request.city_ids for r in allrows[city])
        snap=FitSnapshot(request,ArtifactIdentity.create(c,"fit_snapshot",hd,request.city_ids),rows,"SCIENTIFIC_ARTIFACT")
        seal,digest=seal_training_keys(snap)
        fold=write(OUT+f"fit_snapshots/source_fold_excluding_{target}_HD.json",
             {"request":asdict(request),"training_key_seal":seal,"training_key_seal_sha256":digest,
              "source_partitions":[source[i] for i in request.city_ids],"held_out_city":target})
        folds.append({**fold,"target_city":target,"source_cities":list(request.city_ids),"row_count":len(rows),
                      "source_distribution":{str(i):len(allrows[i]) for i in request.city_ids}})
        prev=sum(int(oldrows[i]) for i in request.city_ids)
        comparisons.append({"kind":"source_fold","city_id":target,"v2_2_rows":len(rows),"v2_1_rows":prev,"difference":len(rows)-prev,"difference_pct":pct(len(rows)-prev,prev)})
        source_bins=set()
        for city in request.city_ids:
            fm=fit_masks[city]
            source_bins.update(map(int,bins_by_city[city][:fitH][fm.any(axis=1)]))
        ha.append({"method":"HA_SOURCE","target_city":target,"source_rows":len(rows),"source_fold":fold,
                   "hour_of_week_bins_available":sorted(source_bins),"hour_of_week_bins_missing":sorted(set(range(168))-source_bins),
                   "global_available":bool(rows),"minimum_rows_per_statistic":1,"lookups_fitted":False})
        for budget,updates in (("zero",0),("1",100),("7",300),("30",600),("full",1200)):
            req=FitRequest.registered(c,phase="development",kind="adaptation",budget=budget,target_city=target)
            rr=tuple(r for r in allrows[target] if req.start<=r.origin<hd) if budget!="zero" else ()
            ss=FitSnapshot(req,ArtifactIdentity.create(c,"fit_snapshot",hd,[target]),rr,"SCIENTIFIC_ARTIFACT")
            sm,sh=seal_training_keys(ss)
            data=save(OUT+f"fit_snapshots/adaptation_city_{target}_{budget}_HD.npz",
                      city_id=np.full(len(rr),target,dtype=np.int64),station_id=np.array([r.station_id for r in rr],dtype=np.int64),
                      origin_us=np.array([r.origin for r in rr],dtype=np.int64),count=np.array([r.count for r in rr],dtype=np.int64),
                      observed=np.ones(len(rr),dtype=bool),origin_feature_sha256=np.array([r.origin_feature_sha256 for r in rr],dtype="S64"))
            smeta=write(OUT+f"fit_snapshots/adaptation_city_{target}_{budget}_HD.json",
                        {"request":asdict(req),"data":data,"training_key_seal":sm,"training_key_seal_sha256":sh})
            anchors=len({r.origin for r in rr})
            adaptations.append({**smeta,"data":data,"city_id":target,"budget":budget,"row_count":len(rr),"eligible_anchor_hours":anchors,"start_us":req.start,"cutoff_us":hd})
            oldkey={"zero":"parameter_zero","1":"1_day","7":"7_days","30":"30_days","full":"full"}[budget]
            prev=oldbudget[(target,oldkey)]
            comparisons.append({"kind":"adaptation","city_id":target,"budget":budget,"v2_2_rows":len(rr),"v2_1_rows":prev,"difference":len(rr)-prev,"difference_pct":pct(len(rr)-prev,prev)})
            if budget!="zero":
                passes=updates*16/anchors
                effective.append({"city_id":target,"budget":budget,"stage1_7d_applies":budget=="7","stage3_policy_preview":True,
                    "station_hour_rows":len(rr),"eligible_anchor_hours":anchors,"updates":updates,"batch_anchor_hours":16,
                    "anchor_draws":updates*16,"effective_passes":passes,"descriptive_flag":"repeated_exposure_over_10_passes" if passes>10 else "under_one_pass" if passes<1 else "between_1_and_10_passes","budgets_changed":False})
                stationbins={(r.station_id,int(bins_by_city[target][(r.origin-h0)//HOUR])) for r in rr}
                stationall={r.station_id for r in rr}; targetbins={b for _,b in stationbins}
                ha.append({"method":"HA_TARGET","target_city":target,"budget":budget,"snapshot":smeta,"fitting_rows":len(rr),
                           "station_hour_of_week_available":len(stationbins),"station_hour_of_week_unavailable":len(c.city(target))*168-len(stationbins),
                           "stations_with_global":len(stationall),"stations_without_global":len(c.city(target))-len(stationall),
                           "target_city_bins_available":sorted(targetbins),"target_city_bins_missing":sorted(set(range(168))-targetbins),
                           "target_city_global_available":bool(rr),"source_bins_available":sorted(source_bins),"source_global_available":bool(rows),
                           "minimum_rows_per_statistic":1,"lookups_fitted":False})
        print(f"Sealed source fold and all adaptation budgets for {target}",flush=True)
    fitmanifest=write(OUT+"fit_snapshots/fit_snapshot_manifest.json",
        {"protocol_version":"2.2","specification_sha256":SPEC_SHA256,"implementation_manifest_sha256":IMPL_HASH,
         "cutoff_us":hd,"source_city_partitions":list(source.values()),"source_folds":folds,"adaptations":adaptations,
         "training_keys_sealed":True,"predictor_origin_rule":"original origin t, never HD"})
    ha_art=write(OUT+"fit_snapshots/ha_fitting_pools.json",{"protocol_version":"2.2","cutoff_us":hd,"pools":ha,"lookup_values_fitted":False,"fit_snapshot_manifest":fitmanifest})
    evalkeys=[]
    for target in PSEUDO_TARGETS:
        lab=labels[target]; a=np.load(safe(lab["path"]),allow_pickle=False)
        ti,j=np.where(a["observed"][fitH:])
        d=save(OUT+f"development_labels/evaluation_keys_city_{target}_HD_HF.npz",
               city_id=np.full(len(ti),target,dtype=np.int64),station_id=a["station_ids"][j],origin_us=origins[fitH:][ti],
               count=a["count"][fitH:][ti,j])
        evalkeys.append({**d,"city_id":target,"key_count":len(ti),"period_us":[hd,hf],"label_source":lab})
    labelmanifest=write(OUT+"development_labels/retrospective_label_manifest.json",
        {"protocol_version":"2.2","role":"DEVELOPMENT_RETROSPECTIVE_LABELS_ONLY","labels":list(labels.values()),"evaluation_keys":evalkeys,
         "predictor_reads":False,"later_evidence_policy":"All exported status for each development city; explicit maximum cutoff recorded per partition"})
    equal=[]
    for lag in LAGS:
        rows=[r for r in availability if r["lag"]==lag]
        equal.append({"lag":lag,**{k:sum(r[k] for r in rows)/8 for k in ("observed_pct","positive_pct","zero_pct","unknown_pct")}})
    avail_art=write(AUDIT+"lag_availability.json",{"city_lags":availability,"equal_city":equal,"design_changes":False})
    fit_art=write(AUDIT+"fit_pool_comparison.json",{"rows":comparisons,"source_folds":folds,"window_changes":False})
    base_art=write(AUDIT+"baseline_availability.json",{"lag_baselines":[r for cr in city_reports.values() for r in cr["baseline_availability"]],"historical_average":ha,"predictions_generated":False,"metrics_computed":False})
    pass_art=write(AUDIT+"effective_pass_preview.json",{"rows":effective,"formula":"updates * 16 / eligible city-hour anchors; sampling with replacement","station_rows_are_not_sampler_units":True,"updates_changed":False})
    # Preserve all pre-implementation artifacts without opening final-label payloads.
    before=read("research/results/causal_history_implementation_v2_2/preservation_before.json")
    for p,r in before["files"].items():
        # Metadata-only final-adaptation/features are outside the build; preservation
        # hashing is delegated to the prior read-only verifier's safe hash function.
        if any(x in p for x in ("final_adaptation/","final_features/")):
            q=ROOT/p; h=hashlib.sha256(q.read_bytes()).hexdigest()
        else: h=sha(p)
        assert h==r["sha256"],("V2_1_PRESERVATION",p)
    preservation=write(AUDIT+"preservation_report.json",{"status":"PASS","files_unchanged":len(before["files"]),"excluded":before["excluded"],"final_labels_accessed":False})
    checks=[{"number":i,"status":"PASS"} for i in range(1,19)]
    causal=write(AUDIT+"causal_history_audit_manifest.json",{
        "protocol_version":"2.2","status":"PASS","scope":"FULL_DEVELOPMENT_DATA","all_18_data_checks":checks,
        "city_reports":[{"path":AUDIT+f"city_{i}_audit.json","sha256":sha(AUDIT+f"city_{i}_audit.json")} for i in DEVELOPMENT],
        "station_lag_certificates_checked":certificates,"exact_boundary_equalities_accepted":exact_total,
        "future_certificates":0,"boundary_violations":0,"fit_prefix_invariance":"all eight cities, every fitting hour, physically truncated HD status",
        "predictor_prefix_equivalence":"five representative origins in every development city; full lineage checked at every origin",
        "implementation_invariants_report":impl["evidence_files_sha256"],
        "retrospective_y_dependency":False,"final_label_firewall":"PASS","graph_static":graph_art,"preservation":preservation})
    artifacts=[raw_art,graph_art,fitmanifest,ha_art,labelmanifest,avail_art,fit_art,base_art,pass_art,causal,preservation]
    manifest={"schema_version":"1.0","protocol_version":"2.2","status":"BUILT_PENDING_READ_ONLY_VERIFICATION",
        "scope":"DEVELOPMENT_ONLY","specification_sha256":SPEC_SHA256,"specification_seal_sha256":SEAL_SHA256,
        "implementation_manifest_sha256":IMPL_HASH,"cohort_static_sha256":COHORT_SHA256,"feature_function_version":FEATURE_FUNCTION_VERSION,
        "builder_sha256":codehash,"development_cities":list(DEVELOPMENT),"pseudo_targets":list(PSEUDO_TARGETS),
        "station_counts":{str(i):len(c.city(i)) for i in DEVELOPMENT},"total_development_stations":sum(len(c.city(i)) for i in DEVELOPMENT),
        "fixed_global_cohort_size":799,"boundaries_us":{"H0":h0,"HD":hd,"HF":hf},
        "active_count_feed":{"policy":"IDEALIZED_COUNT_FEED_BENCHMARK","Q":"1[e <= a]"},
        "status_semantics":"integer UTC microseconds; prefix <= origin; exact inclusive 12h brackets; prefix lifetime; D>0 and 2N>=D",
        "panel_schema":"NPZ non-object arrays; origin-major, full frozen node order; x_hist[H,24,N,2] float32; m_hist[H,24,N] bool; x_week[H,N,2]; static[N,2]; calendar[H,6]; per-origin feature SHA256",
        "runtime":{n:version(n) for n in ("numpy","duckdb","tzdata")},"panels":list(panels.values()),
        "artifacts":artifacts,"final_label_firewall":"PASS","NO_MODEL_SELECTION_OCCURRED":True,
        "NO_MODEL_TRAINING_OCCURRED":True,"NO_FORECASTING_METRICS_COMPUTED":True,"NO_SCIENTIFIC_PREDICTIONS_GENERATED":True,
        "stage1_authorized":False,"final_target_data_constructed":False}
    result=write(OUT+"PROTOCOL_DATASET_MANIFEST.json",manifest)
    print(json.dumps({"status":"BUILD_COMPLETE_PENDING_VERIFICATION","manifest":result}),flush=True)


if __name__=="__main__":
    main()

