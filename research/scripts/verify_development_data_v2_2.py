"""Read-only V2.2 development-data verifier. Never trains, predicts or writes."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/"tmp/neural_build/pydeps"),str(ROOT)]
import numpy as np
from research.development_data_v2_2.registry import DevelopmentRegistry
from research.v2_2.contract import DEVELOPMENT,PSEUDO_TARGETS,HOUR,EPOCH,canonical_bytes
from research.v2_2.artifacts import ArtifactIdentity
from research.v2_2.snapshots import seal_training_keys,require_training_keys
OUT="processed/protocol_v2_2/"
AUDIT="research/results/causal_history_audit_v2_2/"


def digest(path):
    p=Path(path)
    if "final_labels" in p.as_posix().lower() or "sealed_final_evaluation_labels" in p.name.lower():
        raise PermissionError("Final-label firewall")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest-sha256",required=True)
    args=parser.parse_args()
    checks=[]
    def check(name,ok):
        checks.append({"check":name,"status":"PASS" if ok else "FAIL"})
        if not ok: raise ValueError(name)
    report={"status":"FAIL","purpose":"READ_ONLY_DEVELOPMENT_DATA_VERIFICATION","checks":checks,
            "manifest_sha256":args.manifest_sha256,"files_written":[],"models_trained":0,
            "forecasting_metrics_computed":False,"scientific_predictions_generated":False,
            "final_target_evaluation_labels_accessed":False}
    try:
        reg=DevelopmentRegistry(ROOT,OUT+"PROTOCOL_DATASET_MANIFEST.json",args.manifest_sha256)
        c=reg.contract; m=reg.manifest; h0,hd,hf=(c.boundaries[x] for x in ("H0","HD","HF"))
        H=int((hf-h0)//HOUR); F=int((hd-h0)//HOUR)
        check("builder_binding",digest(ROOT/"research/scripts/build_development_data_v2_2.py")==m["builder_sha256"])
        impl=json.loads((ROOT/"research/results/causal_history_implementation_v2_2/implementation_manifest.json").read_text())
        check("accepted_implementation_manifest",digest(ROOT/"research/results/causal_history_implementation_v2_2/implementation_manifest.json")==m["implementation_manifest_sha256"])
        for group in ("implementation_files_sha256","test_files_sha256","evidence_files_sha256"):
            for p,h in impl[group].items(): check("implementation_binding:"+p,digest(ROOT/p)==h)
        check("exact_development_cohort",m["total_development_stations"]==523 and
              m["station_counts"]=={str(i):len(c.city(i)) for i in DEVELOPMENT} and m["fixed_global_cohort_size"]==799)
        check("fixed_boundaries",m["boundaries_us"]=={"H0":h0,"HD":hd,"HF":hf})
        check("count_policy",m["active_count_feed"]=={"policy":"IDEALIZED_COUNT_FEED_BENCHMARK","Q":"1[e <= a]"})
        for e in m["artifacts"]:
            reg.bound(e); check("artifact_bytes:"+e["path"],True)
        raw=reg.json_artifact(AUDIT+"source_bindings.json")
        for path,entry in raw["files"].items():
            check("raw_source_binding:"+path,digest(ROOT/path)==entry["sha256"])
        availability=reg.json_artifact(AUDIT+"lag_availability.json")
        av={(r["city_id"],r["lag"]):r for r in availability["city_lags"]}
        labels_authority=reg.json_artifact(OUT+"development_labels/retrospective_label_manifest.json")
        fit_authority=reg.json_artifact(OUT+"fit_snapshots/fit_snapshot_manifest.json")
        baseline=reg.json_artifact(AUDIT+"baseline_availability.json")
        comparisons=reg.json_artifact(AUDIT+"fit_pool_comparison.json")
        exposure=reg.json_artifact(AUDIT+"effective_pass_preview.json")
        causal=reg.json_artifact(AUDIT+"causal_history_audit_manifest.json")
        check("18_full_development_checks",causal["all_18_data_checks"]==[{"number":n,"status":"PASS"} for n in range(1,19)]
              and causal["future_certificates"]==0 and causal["boundary_violations"]==0)
        panels={}; sources={}; bins={}
        for city in DEVELOPMENT:
            a=reg.predictors(city); N=len(c.city(city)); panels[city]=a
            expected={"x_hist":((H,24,N,2),np.float32),"m_hist":((H,24,N),bool),"x_week":((H,N,2),np.float32),
                      "x_static":((N,2),np.float32),"x_calendar":((H,6),np.float32)}
            check(f"schema:{city}",all(a[k].shape==shape and a[k].dtype==dtype for k,(shape,dtype) in expected.items()))
            check(f"origin_keys:{city}",np.array_equal(a["origin_us"],np.arange(h0,hf,HOUR,dtype=np.int64)))
            check(f"indicator_equals_mask:{city}",np.array_equal(a["x_hist"][:,:,:,1],a["m_hist"]))
            for lag in list(range(1,25))+[168]:
                v=a["x_hist"][:,lag-1] if lag!=168 else a["x_week"]
                obs=v[:,:,1]==1; values=v[:,:,0]
                check(f"states:{city}:{lag}",np.isfinite(v).all() and np.all((v[:,:,1]==0)|obs)
                      and np.all(values[~obs]==0) and np.all(values>=0))
                pos=int(np.sum(obs&(values>0))); zero=int(np.sum(obs&(values==0))); unknown=int(np.sum(~obs))
                r=av[(city,lag)]; total=H*N
                check(f"availability_counts:{city}:{lag}",r["total_cells"]==total and r["observed_cells"]==pos+zero
                      and r["positive_cells"]==pos and r["zero_cells"]==zero and r["unknown_cells"]==unknown
                      and abs(r["observed_pct"]-100*(pos+zero)/total)<1e-10)
                check(f"history_H0_padding:{city}:{lag}",np.all(v[:min(lag,H)]==0) if lag>=H else np.all(v[:lag]==0))
            # Byte binding proves saved predictors match original-origin construction.
            for ti,t in enumerate(a["origin_us"]):
                identity=ArtifactIdentity.create(c,"predictor_cache",int(t),[city])
                prefix=canonical_bytes((identity.as_json(),city,int(t),tuple(int(i) for i in a["station_ids"])))
                body=b"".join(np.ascontiguousarray(z,dtype="<f4").tobytes() for z in
                              (a["x_hist"][ti],a["x_week"][ti],a["x_static"],a["x_calendar"][ti]))+a["m_hist"][ti].astype(np.uint8).tobytes()
                if hashlib.sha256(prefix+body).hexdigest()!=a["origin_feature_sha256"][ti].decode():
                    raise ValueError(f"Original-origin fingerprint mismatch {city}/{ti}")
            check(f"all_original_origin_fingerprints:{city}",True)
            local=[(EPOCH+timedelta(microseconds=int(t))).astimezone(ZoneInfo(c.city(city)[0].timezone)) for t in a["origin_us"]]
            bins[city]=np.array([d.weekday()*24+d.hour for d in local])
            calendar=np.array([(math.sin(2*math.pi*d.hour/24),math.cos(2*math.pi*d.hour/24),
                    math.sin(2*math.pi*d.weekday()/7),math.cos(2*math.pi*d.weekday()/7),float(d.weekday()>=5),
                    d.utcoffset().total_seconds()/43200) for d in local],dtype=np.float32)
            check(f"calendar_and_UTC_offset:{city}",np.array_equal(calendar,a["x_calendar"]))
            static=np.array([(np.float32(math.log1p(s.bike_racks)),1) if s.bike_racks is not None and s.bike_racks>0 else (0,0) for s in c.city(city)],dtype=np.float32)
            check(f"frozen_static:{city}",np.array_equal(static,a["x_static"]))
            label=reg.retrospective_labels(city)
            check(f"separate_labels:{city}",set(label)=={"origin_us","station_ids","observed","count"}
                  and np.array_equal(label["origin_us"],a["origin_us"]) and np.array_equal(label["station_ids"],a["station_ids"])
                  and label["count"].shape==(H,N) and label["observed"].dtype==bool
                  and np.all(label["count"][~label["observed"]]==-1) and np.all(label["count"][label["observed"]]>=0))
            src=next(s for s in fit_authority["source_city_partitions"] if s["city_id"]==city)
            sa=reg.arrays(src); sources[city]=sa
            idx=((sa["origin_us"]-h0)//HOUR).astype(np.int64)
            check(f"source_scope_cutoff:{city}",np.all(sa["city_id"]==city) and np.all((sa["origin_us"]>=h0)&(sa["origin_us"]+HOUR<=hd))
                  and set(sa["station_id"])<=set(a["station_ids"]) and np.all(sa["observed"])
                  and np.all(sa["count"]>=0) and len(sa["origin_us"])==src["row_count"])
            check(f"source_original_features:{city}",np.array_equal(sa["origin_feature_sha256"],a["origin_feature_sha256"][idx]))
            check(f"source_unique_sorted_keys:{city}",len(set(zip(sa["station_id"],sa["origin_us"])))==len(idx)
                  and list(zip(sa["station_id"],sa["origin_us"]))==sorted(zip(sa["station_id"],sa["origin_us"])))
            cityaudit=next(r for r in causal["city_reports"] if f"city_{city}_" in r["path"])
            cr=json.loads(reg.bound(cityaudit).read_text())
            check(f"full_data_coverage_audit:{city}",cr["all_origins_checked"]==H and cr["all_history_cells_checked"]==H*N*25
                  and cr["fit_prefix_comparisons"]==F and cr["status"]=="PASS")
            if city in PSEUDO_TARGETS:
                ev=label["observed"][F:]
                for r in [r for r in baseline["lag_baselines"] if r["city_id"]==city]:
                    obs=a["m_hist"][F:,0] if r["baseline"]=="PERSISTENCE_T_MINUS_1" else a["x_week"][F:,:,1].astype(bool)
                    domain=ev if r["key_scope"]=="retrospectively_observed_evaluation_keys" else np.ones_like(ev)
                    avail=int((obs&domain).sum()); total=int(domain.sum())
                    check(f"baseline_availability:{city}:{r['baseline']}:{r['key_scope']}",r["available"]==avail and
                          r["unavailable"]==total-avail and r["total"]==total and abs(r["coverage_pct"]-100*avail/total)<1e-10)
                ek=next(r for r in labels_authority["evaluation_keys"] if r["city_id"]==city)
                keys=reg.arrays(ek); ti,j=np.where(ev)
                check(f"evaluation_keys:{city}",np.array_equal(keys["origin_us"],a["origin_us"][F:][ti])
                      and np.array_equal(keys["station_id"],a["station_ids"][j])
                      and np.array_equal(keys["count"],label["count"][F:][ti,j]) and np.all(keys["city_id"]==city))
            print(f"Verified city {city}",file=sys.stderr,flush=True)
        for target in PSEUDO_TARGETS:
            ss=reg.sealed_fit(target=target)
            check(f"source_exclusion:{target}",ss.request.city_ids==tuple(i for i in DEVELOPMENT if i!=target)
                  and all(r.city_id!=target for r in ss.rows))
            for budget in ("zero","1","7","30","full"):
                snap=reg.sealed_fit(target=target,budget=budget)
                source=sources[target]; start=snap.request.start
                expected=(source["origin_us"]>=start)&(source["origin_us"]<hd) if budget!="zero" else np.zeros(len(source["origin_us"]),dtype=bool)
                check(f"adaptation_exact_subset:{target}:{budget}",[(r.station_id,r.origin,r.count) for r in snap.rows]==
                      list(zip(source["station_id"][expected],source["origin_us"][expected],source["count"][expected])))
                if budget=="zero":
                    check(f"zero_budget_empty:{target}",not snap.rows)
                    sm,sh=seal_training_keys(snap)
                    try: require_training_keys(snap,sm,sh,scientific=True)
                    except PermissionError: check(f"zero_cannot_start_fit:{target}",True)
                    else: check(f"zero_cannot_start_fit:{target}",False)
                else:
                    row=next(r for r in exposure["rows"] if r["city_id"]==target and r["budget"]==budget)
                    anchors=len({r.origin for r in snap.rows})
                    check(f"effective_passes:{target}:{budget}",row["eligible_anchor_hours"]==anchors
                          and row["station_hour_rows"]==len(snap.rows) and row["updates"]=={"1":100,"7":300,"30":600,"full":1200}[budget]
                          and abs(row["effective_passes"]-row["updates"]*16/anchors)<1e-12)
                    try: require_training_keys(snap,None,"",scientific=True)
                    except PermissionError: check(f"missing_seal_rejected:{target}:{budget}",True)
                    else: check(f"missing_seal_rejected:{target}:{budget}",False)
            print(f"Verified sealed fold and budgets {target}",file=sys.stderr,flush=True)
        for r in comparisons["rows"]:
            check(f"comparison_arithmetic:{r['kind']}:{r['city_id']}:{r.get('budget','')}",
                  r["difference"]==r["v2_2_rows"]-r["v2_1_rows"] and
                  (r["difference_pct"] is None if r["v2_1_rows"]==0 else abs(r["difference_pct"]-100*r["difference"]/r["v2_1_rows"])<1e-10))
        for pool in baseline["historical_average"]:
            target=pool["target_city"]
            sourcecities=[i for i in DEVELOPMENT if i!=target]
            sourcebins=set()
            for city in sourcecities:
                sourcebins.update(map(int,bins[city][((sources[city]["origin_us"]-h0)//HOUR).astype(np.int64)]))
            if pool["method"]=="HA_SOURCE":
                check(f"HA_SOURCE_pool:{target}",pool["source_rows"]==sum(len(sources[i]["count"]) for i in sourcecities)
                      and pool["hour_of_week_bins_available"]==sorted(sourcebins)
                      and pool["hour_of_week_bins_missing"]==sorted(set(range(168))-sourcebins)
                      and pool["global_available"]==any(len(sources[i]["count"]) for i in sourcecities))
            else:
                budget=pool["budget"]; start=h0 if budget=="full" else hd-int(budget)*24*HOUR
                a=sources[target]; keep=a["origin_us"]>=start
                sb=set(zip(map(int,a["station_id"][keep]),map(int,bins[target][((a["origin_us"][keep]-h0)//HOUR).astype(np.int64)])))
                stationall={s for s,b in sb}; tb={b for s,b in sb}
                check(f"HA_TARGET_pool:{target}:{budget}",pool["fitting_rows"]==int(keep.sum()) and pool["station_hour_of_week_available"]==len(sb)
                      and pool["station_hour_of_week_unavailable"]==len(c.city(target))*168-len(sb)
                      and pool["stations_with_global"]==len(stationall) and pool["stations_without_global"]==len(c.city(target))-len(stationall)
                      and pool["target_city_bins_available"]==sorted(tb) and pool["target_city_bins_missing"]==sorted(set(range(168))-tb)
                      and pool["source_bins_available"]==sorted(sourcebins) and not pool["lookups_fitted"])
        for r in availability["equal_city"]:
            cityrows=[x for x in availability["city_lags"] if x["lag"]==r["lag"]]
            check(f"equal_city:{r['lag']}",all(abs(r[k]-sum(x[k] for x in cityrows)/8)<1e-10 for k in ("observed_pct","positive_pct","zero_pct","unknown_pct")))
        graph=reg.json_artifact(AUDIT+"graph_static_reconciliation.json")
        check("all_24_graphs_reusable",graph["status"]=="PASS" and graph["graph_count"]==24 and graph["new_adjacency_copies_created"]==0)
        for g in graph["graphs"]:
            check(f"immutable_graph:{g['city_id']}:{g['k']}",g["city_id"] in DEVELOPMENT and g["k"] in (4,8,16)
                  and digest(ROOT/g["path"])==g["sha256"] and g["v2_2_roster_sha256"]==c.city_roster_hash(g["city_id"])
                  and g["static_sha256"]==hashlib.sha256(canonical_bytes([__import__("dataclasses").asdict(s) for s in c.city(g["city_id"])])).hexdigest())
            adjacency,identity=reg.graph(g["city_id"],g["k"])
            check(f"versioned_static_graph_access:{g['city_id']}:{g['k']}",adjacency.shape==(len(c.city(g["city_id"])),)*2
                  and identity.protocol_version=="2.2" and identity.graph_binding_sha256==g["graph_config"]["adjacency_hash_sha256"])
        for path in ("processed/protocol_v2_1/development/development_panel.parquet",
                     "processed/protocol_v2_1/final_labels/SEALED_final_evaluation_labels.parquet",
                     "processed/protocol_v2_2/final_labels/x.npz","../escape"):
            try: reg.path(path)
            except (ValueError,PermissionError): check("rejected_path:"+path,True)
            else: check("rejected_path:"+path,False)
        for city in (195,199,237,617):
            try: reg.predictors(city)
            except PermissionError: check(f"final_city_rejected:{city}",True)
            else: check(f"final_city_rejected:{city}",False)
        from unittest.mock import patch
        for changed in ({"protocol_version":"2.1"},{"specification_sha256":"0"*64},
                        {"cohort_static_sha256":"0"*64},{"implementation_manifest_sha256":"0"*64}):
            altered=dict(m); altered.update(changed); payload=json.dumps(altered).encode()
            with patch.object(Path,"read_bytes",return_value=payload):
                try: DevelopmentRegistry(ROOT,OUT+"PROTOCOL_DATASET_MANIFEST.json",hashlib.sha256(payload).hexdigest())
                except ValueError: check("mismatched_manifest_rejected:"+next(iter(changed)),True)
                else: check("mismatched_manifest_rejected:"+next(iter(changed)),False)
        original=reg.panels[129]["path"]
        reg.panels[129]["path"]=OUT+"development_labels/city_129_retrospective.npz"
        try:
            with patch.object(reg,"arrays",side_effect=AssertionError("Label read attempted")):
                try: reg.predictors(129)
                except PermissionError: check("predictor_label_path_rejected_before_read",True)
                else: check("predictor_label_path_rejected_before_read",False)
        finally: reg.panels[129]["path"]=original
        old=json.loads((ROOT/"research/results/causal_history_implementation_v2_2/preservation_before.json").read_text())
        for p,v in old["files"].items():
            if digest(ROOT/p)!=v["sha256"]: raise ValueError("Historical byte change: "+p)
        check("1182_historical_files_preserved",len(old["files"])==1182)
        report["preserved_historical_files"]=len(old["files"]); report["preservation_exclusions"]=old["excluded"]
        check("no_scientific_model_execution",all(m[x] is True for x in ("NO_MODEL_SELECTION_OCCURRED","NO_MODEL_TRAINING_OCCURRED",
                   "NO_FORECASTING_METRICS_COMPUTED","NO_SCIENTIFIC_PREDICTIONS_GENERATED")) and m["stage1_authorized"] is False)
        for d in ("stage1_scale_loss_v2_2","stage2_graphgru_selection_v2_2","stage2b_vanilla_selection_v2_2","stage4_adversarial_method_v2_2"):
            check("no_result_directory:"+d,not (ROOT/"research/results"/d).exists())
        check("no_final_data_partitions",all(not (ROOT/OUT/d).exists() for d in ("final_labels","final_adaptation","final_features")))
        report["status"]="PASS"
        report["verdict"]="A. V2_2_DEVELOPMENT_DATA_VERIFIED_READY_FOR_STAGE1_AUTHORIZATION"
    except (OSError,ValueError,KeyError,TypeError,AssertionError) as exc:
        report["error"]=str(exc)
        report["verdict"]="B. V2_2_DEVELOPMENT_DATA_BLOCKED_BY_LISTED_FAILURES"
    report["check_count"]=len(checks)
    report["verifier_sha256"]=digest(Path(__file__))
    report["registry_sha256"]=digest(ROOT/"research/development_data_v2_2/registry.py")
    print(json.dumps(report,indent=2,ensure_ascii=True))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
