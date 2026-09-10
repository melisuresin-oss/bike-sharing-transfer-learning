from __future__ import annotations
import hashlib, math, random, statistics
from collections import defaultdict
from . import core

BLOCK_LENGTHS=(168,)*9+(53,)

def city_seed(city_id):
    return int.from_bytes(hashlib.sha256(f"{core.BOOTSTRAP_NAMESPACE}|{city_id}".encode("ascii")).digest()[:8],"big")

def block_draws(city_id, replicates=2000):
    if city_id not in core.BOOTSTRAP_SEEDS or city_seed(city_id)!=core.BOOTSTRAP_SEEDS[city_id]: raise RuntimeError("bootstrap seed drift")
    rng=random.Random(core.BOOTSTRAP_SEEDS[city_id])
    return tuple(tuple(rng.randrange(10) for _ in range(10)) for _ in range(replicates))

def valid_pairs(rows):
    return [r for r in rows if r.get("label_valid") is True and r.get("prediction_available") is True
            and r.get("label") is not None and r.get("prediction") is not None]

def metrics(rows):
    r=valid_pairs(rows)
    if not r:return {"mae":None,"rmse":None,"wape":None,"station_macro_mae":None,"valid_n":0}
    errors=[abs(float(x["label"])-float(x["prediction"])) for x in r]
    denom=sum(abs(float(x["label"])) for x in r)
    stations=defaultdict(list)
    for x,e in zip(r,errors):stations[x["station_id"]].append((x["hour"],e))
    eligible=[statistics.fmean(e for _,e in v) for v in stations.values() if len({h for h,_ in v})>=24]
    return {"mae":statistics.fmean(errors),"rmse":math.sqrt(statistics.fmean(e*e for e in errors)),
            "wape":sum(errors)/denom if denom else None,
            "station_macro_mae":statistics.fmean(eligible) if eligible else None,"valid_n":len(r)}

def seed_summary(values):
    if len(values)!=5 or any(v is None for v in values):return {"mean":None,"population_sd":None}
    return {"mean":statistics.fmean(values),"population_sd":statistics.pstdev(values)}

def linear_quantile(values,q):
    if not values:return None
    x=sorted(values); pos=(len(x)-1)*q; lo=int(pos); hi=math.ceil(pos)
    return x[lo] if lo==hi else x[lo]+(x[hi]-x[lo])*(pos-lo)

def confidence_interval(values):
    if len(values)!=2000 or any(v is None for v in values):return {"lower":None,"upper":None,"reason":"undefined_replicate"}
    return {"lower":linear_quantile(values,.025),"upper":linear_quantile(values,.975),"reason":None}
