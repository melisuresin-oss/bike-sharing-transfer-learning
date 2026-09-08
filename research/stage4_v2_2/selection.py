"""Prospective Stage-4 selection; never imports a model or data reader."""
from decimal import Decimal, ROUND_HALF_UP
from functools import cmp_to_key
import math
import statistics

FOLDS=(532,476,619,658)
SEEDS=(17,29,43)
LAMBDAS=(0.01,0.10,0.50)
SCHEDULES=("constant","linear")


def candidate_id(lam,schedule):
    if lam not in LAMBDAS or schedule not in SCHEDULES:raise ValueError("Unregistered candidate")
    return f"grl_l{int(lam*100):02d}_{schedule}"


def quantize_primary(value):
    if not math.isfinite(value) or value<0:raise ValueError("Invalid primary MAE")
    return Decimal(str(value)).quantize(Decimal("0.0001"),rounding=ROUND_HALF_UP)


def summarize(lam,schedule,fold_maes):
    if len(fold_maes)!=4 or any(not math.isfinite(v) or v<0 for v in fold_maes):raise ValueError("Four finite nonnegative fold means required")
    mean=statistics.fmean(fold_maes)
    return {"candidate_id":candidate_id(lam,schedule),"lambda_max":lam,"schedule":schedule,
            "fold_seed_mean_maes":list(fold_maes),"primary_unrounded":mean,
            "primary_4dp":str(quantize_primary(mean)),"fold_population_sd_unrounded":statistics.pstdev(fold_maes)}


def compare(a,b):
    qa,qb=quantize_primary(a["primary_unrounded"]),quantize_primary(b["primary_unrounded"])
    if qa!=qb:
        x,y=a["primary_unrounded"],b["primary_unrounded"]
    elif a["fold_population_sd_unrounded"]!=b["fold_population_sd_unrounded"]:
        x,y=a["fold_population_sd_unrounded"],b["fold_population_sd_unrounded"]
    elif a["lambda_max"]!=b["lambda_max"]:x,y=a["lambda_max"],b["lambda_max"]
    else:x,y=SCHEDULES.index(a["schedule"]),SCHEDULES.index(b["schedule"])
    return (x>y)-(x<y)


def rank_summaries(summaries):
    if len({r["candidate_id"] for r in summaries})!=len(summaries):raise ValueError("Duplicate candidate")
    return [dict(r,rank=i+1) for i,r in enumerate(sorted(summaries,key=cmp_to_key(compare)))]


def select(rows):
    expected={(candidate_id(l,s),f,seed) for l in LAMBDAS for s in SCHEDULES for f in FOLDS for seed in SEEDS}
    keys=[(r["candidate_id"],r["pseudo_target"],r["seed"]) for r in rows]
    if len(keys)!=72 or set(keys)!=expected:raise ValueError("Exactly 72 unique registered post-adaptation evaluations required")
    if any(r["regime"]!="post_7d_adaptation" or not math.isfinite(r["mae"]) or r["mae"]<0 for r in rows):
        raise ValueError("Only finite nonnegative post-7d count-space MAE is selectable")
    lookup={k:r for k,r in zip(keys,rows)};summaries=[];cells=[]
    for l in LAMBDAS:
        for schedule in SCHEDULES:
            cid=candidate_id(l,schedule);means=[]
            for fold in FOLDS:
                values=[lookup[cid,fold,seed]["mae"] for seed in SEEDS]
                mean=statistics.fmean(values);means.append(mean)
                cells.append({"candidate_id":cid,"pseudo_target":fold,"seeds":list(SEEDS),"seed_maes":values,"seed_mean_mae":mean})
            summaries.append(summarize(l,schedule,means))
    ranked=rank_summaries(summaries)
    return {"selected_candidate":ranked[0]["candidate_id"],"ranking":ranked,"fold_cells":cells,
            "primary_metric":"post_7d_adaptation_count_space_mae","evaluation_count":72}
