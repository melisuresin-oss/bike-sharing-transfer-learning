"""Frozen V2.2 count-space metrics and paired temporal-block bootstrap primitives.

This module never opens labels. Callers provide already joined retrospective rows.
"""
from __future__ import annotations
import hashlib,math,random,statistics
from collections import defaultdict
from functools import lru_cache

BLOCK_LENGTHS=(168,)*9+(53,)
BOOTSTRAP_NAMESPACE="FINAL_EVALUATION_V2_2|TEMPORAL_BLOCK_BOOTSTRAP|20260908"
BOOTSTRAP_SEEDS={195:12648532355080271797,199:2634071097790620903,237:14234234470032652229,617:6032700276159629083}
FINAL_SEEDS=(17,29,43,71,101)
METRICS=("mae","rmse","wape","station_macro_mae")

def _valid(row):
    if row.get("label_valid") is not True or row.get("prediction_available") is not True: return False
    try: label=float(row["label"]); prediction=float(row["prediction"])
    except (KeyError,TypeError,ValueError): return False
    return math.isfinite(label) and math.isfinite(prediction)

def eligible_stations(rows):
    hours=defaultdict(set)
    for row in rows:
        if _valid(row): hours[row["station_id"]].add(row["hour"])
    return frozenset(station for station,values in hours.items() if len(values)>=24)

def count_metrics(rows,station_eligibility=None):
    valid=[r for r in rows if _valid(r)]
    if not valid: return {"mae":None,"rmse":None,"wape":None,"station_macro_mae":None,"valid_n":0}
    absolute=[abs(float(r["label"])-float(r["prediction"])) for r in valid]
    squared=[(float(r["label"])-float(r["prediction"]))**2 for r in valid]
    denominator=sum(abs(float(r["label"])) for r in valid)
    eligible=eligible_stations(valid) if station_eligibility is None else frozenset(station_eligibility)
    by_station=defaultdict(list)
    for row,error in zip(valid,absolute):
        if row["station_id"] in eligible: by_station[row["station_id"]].append(error)
    station_values=[statistics.fmean(by_station[station]) for station in sorted(eligible) if by_station[station]]
    station_macro=statistics.fmean(station_values) if len(station_values)==len(eligible) and eligible else None
    return {"mae":statistics.fmean(absolute),"rmse":math.sqrt(statistics.fmean(squared)),
        "wape":sum(absolute)/denominator if denominator else None,"station_macro_mae":station_macro,"valid_n":len(valid)}

def aggregate_five_seeds(values):
    if len(values)!=5 or any(value is None for value in values): return {"mean":None,"population_sd":None}
    return {"mean":statistics.fmean(values),"population_sd":statistics.pstdev(values)}

def city_seed(city_id):
    return int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_NAMESPACE}|{city_id}".encode("ascii")).digest()[:8],"big")

@lru_cache(maxsize=4)
def block_draws(city_id,replicates=2000):
    if city_id not in BOOTSTRAP_SEEDS or city_seed(city_id)!=BOOTSTRAP_SEEDS[city_id] or replicates!=2000: raise ValueError("unregistered bootstrap request")
    rng=random.Random(BOOTSTRAP_SEEDS[city_id]); return tuple(tuple(rng.randrange(10) for _ in range(10)) for _ in range(2000))

def block_indices(draw):
    starts=[]; cursor=0
    for length in BLOCK_LENGTHS: starts.append((cursor,cursor+length)); cursor+=length
    if cursor!=1565 or len(draw)!=10: raise RuntimeError("bootstrap block structure drift")
    result=[]
    for block in draw:
        if not 0<=block<10: raise ValueError("block id")
        start,end=starts[block]; result.extend(range(start,end))
    return result

def linear_quantile(values,q):
    if not values:return None
    ordered=sorted(values); position=(len(ordered)-1)*q; low=int(position); high=math.ceil(position)
    return ordered[low] if low==high else ordered[low]+(ordered[high]-ordered[low])*(position-low)

def confidence_interval(values):
    if len(values)!=2000 or any(value is None for value in values):
        return {"lower":None,"upper":None,"reason":"undefined_replicate"}
    return {"lower":linear_quantile(values,.025),"upper":linear_quantile(values,.975),"reason":None}

def _hour_index(row):
    value=row.get("hour_index",row.get("hour"))
    if not isinstance(value,int) or not 0<=value<1565: raise ValueError("registered hour_index 0..1564 required")
    return value

def _resample(rows,indices):
    buckets=defaultdict(list)
    for row in rows: buckets[_hour_index(row)].append(row)
    return [row for index in indices for row in buckets.get(index,())]

def _mean_seed_metrics(per_seed):
    result={}
    for metric in METRICS:
        values=[per_seed[seed][metric] for seed in FINAL_SEEDS]
        result[metric]=statistics.fmean(values) if all(v is not None for v in values) else None
    return result

def bootstrap_metric_cell(city_id,rows_by_seed):
    learned=set(rows_by_seed)==set(FINAL_SEEDS); deterministic=set(rows_by_seed)=={None}
    if not (learned or deterministic): raise ValueError("use five frozen seeds or deterministic None")
    elig={seed:eligible_stations(rows) for seed,rows in rows_by_seed.items()}
    point_by_seed={seed:count_metrics(rows,elig[seed]) for seed,rows in rows_by_seed.items()}
    if learned:
        point=_mean_seed_metrics(point_by_seed)
        dispersion={metric:aggregate_five_seeds([point_by_seed[s][metric] for s in FINAL_SEEDS])["population_sd"] for metric in METRICS}
    else:
        point={metric:point_by_seed[None][metric] for metric in METRICS}; dispersion={metric:None for metric in METRICS}
    replicates=[]
    for draw in block_draws(city_id):
        indices=block_indices(draw); values={seed:count_metrics(_resample(rows,indices),elig[seed]) for seed,rows in rows_by_seed.items()}
        replicates.append(_mean_seed_metrics(values) if learned else {metric:values[None][metric] for metric in METRICS})
    intervals={metric:confidence_interval([r[metric] for r in replicates]) for metric in METRICS}
    return {"point":point,"seed_population_sd":dispersion,"replicates":replicates,"intervals":intervals,
        "training_seeds_outside_bootstrap":True,"draws_sha256":hashlib.sha256(repr(block_draws(city_id)).encode("ascii")).hexdigest()}

def _map(rows):
    result={}
    for row in rows:
        if not _valid(row): continue
        key=(row["station_id"],_hour_index(row))
        if key in result: raise ValueError("duplicate station-hour key")
        result[key]=row
    return result

def paired_gain_metrics(city_id,reference_by_seed,method_by_seed):
    ref_learned=set(reference_by_seed)==set(FINAL_SEEDS); ref_det=set(reference_by_seed)=={None}
    method_learned=set(method_by_seed)==set(FINAL_SEEDS); method_det=set(method_by_seed)=={None}
    if not (ref_learned or ref_det) or not (method_learned or method_det): raise ValueError("invalid seed universe")
    seeds=FINAL_SEEDS if ref_learned or method_learned else (None,)
    cohorts={}
    for seed in seeds:
        ref=_map(reference_by_seed[seed if ref_learned else None]); method=_map(method_by_seed[seed if method_learned else None])
        keys=sorted(set(ref)&set(method),key=lambda x:(x[1],x[0])); left=[]; right=[]
        for key in keys:
            if float(ref[key]["label"])!=float(method[key]["label"]): raise ValueError("paired label mismatch")
            left.append(ref[key]); right.append(method[key])
        cohorts[seed]=(left,right,eligible_stations(left)&eligible_stations(right))
    def gains(indices=None):
        per_seed={}
        for seed,(left,right,eligible) in cohorts.items():
            if indices is not None: left=_resample(left,indices); right=_resample(right,indices)
            lm=count_metrics(left,eligible); rm=count_metrics(right,eligible)
            per_seed[seed]={metric:(lm[metric]-rm[metric] if lm[metric] is not None and rm[metric] is not None else None) for metric in METRICS}
        if seeds==(None,): return per_seed[None]
        return {metric:(statistics.fmean(per_seed[s][metric] for s in FINAL_SEEDS) if all(per_seed[s][metric] is not None for s in FINAL_SEEDS) else None) for metric in METRICS}
    point=gains(); replicates=[gains(block_indices(draw)) for draw in block_draws(city_id)]
    return {"point_gain":point,"replicates":replicates,
        "intervals":{metric:confidence_interval([r[metric] for r in replicates]) for metric in METRICS},
        "gain_definition":"error(reference)-error(method)","fixed_intersection":True,"paired_draws":True}

def paired_gain_bootstrap(city_id,reference_hour_errors,method_hour_errors):
    if len(reference_hour_errors)!=1565 or len(method_hour_errors)!=1565: raise ValueError("1565 hourly inputs required")
    values=[]
    for draw in block_draws(city_id):
        pairs=[(reference_hour_errors[i],method_hour_errors[i]) for i in block_indices(draw) if reference_hour_errors[i] is not None and method_hour_errors[i] is not None]
        values.append(statistics.fmean(a-b for a,b in pairs) if pairs else None)
    return {"replicates":values,"ci":confidence_interval(values),"gain_definition":"error(reference)-error(method)","paired_draws":True,"terminal_53h_block_retained":True}

def equal_city_macro(city_values):
    if set(city_values)!={195,199,237,617} or any(city_values[c] is None for c in city_values): return None
    return statistics.fmean(city_values[c] for c in (195,199,237,617))

def equal_city_macro_replicates(city_replicates):
    if set(city_replicates)!={195,199,237,617} or any(len(v)!=2000 for v in city_replicates.values()): raise ValueError("four 2000-replicate city streams required")
    return [equal_city_macro({city:city_replicates[city][i] for city in (195,199,237,617)}) for i in range(2000)]