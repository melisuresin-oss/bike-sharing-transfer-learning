from __future__ import annotations
from collections import Counter
from . import core

def _job(job_id, category, method, seed, target=None, budget=None, depends_on=None, updates=None):
    return {"job_id": job_id, "category": category, "method": method, "seed": seed,
            "target_city_id": target, "budget": budget, "updates": updates,
            "depends_on": depends_on, "attempt_policy": "fresh_uuid_from_update_0_no_partial_resume",
            "architecture": "VGRU_H032_D00" if "vanilla" in method else "GGRU_K04_H032_D00",
            "output_mode": "LOG1P_TARGET_MAE"}

def neural_jobs():
    jobs=[]
    for seed in core.SEEDS:
        ordinary_source=f"ordinary_source_seed-{seed}"
        grl_source=f"grl_l50_constant_source_seed-{seed}"
        jobs += [_job(ordinary_source,"source","ordinary_source_graphgru",seed,updates=12000),
                 _job(grl_source,"source","grl_l50_constant_source_graphgru",seed,updates=12000)]
        for target,_ in core.TARGETS:
            for budget in core.NONZERO_BUDGETS:
                u=core.UPDATES[budget]
                suffix=f"target-{target}_seed-{seed}_budget-{budget}"
                jobs += [
                  _job("target_only_vanilla_"+suffix,"target_only","target_only_vanilla",seed,target,budget,None,u),
                  _job("target_only_graph_"+suffix,"target_only","target_only_graph",seed,target,budget,None,u),
                  _job("ordinary_adaptation_"+suffix,"adaptation","ordinary_source_adapted",seed,target,budget,ordinary_source,u),
                  _job("pooled_graph_"+suffix,"pooled","pooled_graph",seed,target,budget,None,12000+u),
                  _job("grl_adaptation_"+suffix,"adaptation","grl_l50_constant_adapted",seed,target,budget,grl_source,u),
                ]
    validate_neural_jobs(jobs)
    return jobs

def validate_neural_jobs(jobs):
    if len(jobs)!=410 or len({j["job_id"] for j in jobs})!=410: raise RuntimeError("exactly 410 unique fits required")
    counts=Counter(j["category"] for j in jobs)
    if counts != Counter(source=10, adaptation=160, target_only=160, pooled=80):
        raise RuntimeError("fit-category accounting drift: "+repr(counts))
    ids={j["job_id"] for j in jobs}
    for j in jobs:
        if j["seed"] not in core.SEEDS or j["target_city_id"] not in (None, *(x[0] for x in core.TARGETS)):
            raise RuntimeError("seed/target drift")
        if j["depends_on"] is not None and j["depends_on"] not in ids: raise RuntimeError("missing source dependency")
        if j["category"]=="adaptation" and j["updates"]!=core.UPDATES[j["budget"]]: raise RuntimeError("adaptation budget drift")
        if j["method"].startswith("grl") and j["method"]!="grl_l50_constant_source_graphgru" and j["method"]!="grl_l50_constant_adapted": raise RuntimeError("GRL candidate drift")

def evaluations():
    rows=[]
    learned=("target_only_vanilla","target_only_graph","ordinary_source_adapted","pooled_graph","grl_l50_constant_adapted")
    for method in learned:
      for seed in core.SEEDS:
       for target,_ in core.TARGETS:
        for budget in core.NONZERO_BUDGETS:
         rows.append({"evaluation_id":f"{method}_target-{target}_seed-{seed}_budget-{budget}","method":method,"target_city_id":target,"seed":seed,"budget":budget,"reuse_labels":[budget]})
    for method in ("ordinary_source_graphgru","grl_l50_constant_source_graphgru"):
      for seed in core.SEEDS:
       for target,_ in core.TARGETS:
        rows.append({"evaluation_id":f"{method}_target-{target}_seed-{seed}","method":method,"target_city_id":target,"seed":seed,"budget":"0","reuse_labels":list(core.BUDGETS)})
    for method in ("persistence","seasonal_naive"):
      for target,_ in core.TARGETS:
       rows.append({"evaluation_id":f"{method}_target-{target}","method":method,"target_city_id":target,"seed":None,"budget":"0","reuse_labels":list(core.BUDGETS)})
    for target,_ in core.TARGETS:
      rows.append({"evaluation_id":f"ha_source_target-{target}","method":"HA_SOURCE","target_city_id":target,"seed":None,"budget":"0","reuse_labels":["0"]})
      for budget in core.NONZERO_BUDGETS:
       rows.append({"evaluation_id":f"ha_target_target-{target}_budget-{budget}","method":"HA_TARGET","target_city_id":target,"seed":None,"budget":budget,"reuse_labels":[budget]})
    if len(rows)!=468 or len({r["evaluation_id"] for r in rows})!=468: raise RuntimeError("exactly 468 evaluation passes required")
    cells=sum(len(r["reuse_labels"]) for r in rows)
    if cells!=660: raise RuntimeError("exactly 660 reporting cells required")
    return rows

def manifests():
    jobs=neural_jobs(); ev=evaluations()
    return ({"schema_version":"final_v2_2.jobs.1","status":"SEALED_PLAN_NOT_EXECUTED","fits":jobs,
             "counts":{"neural_fits":410,"source":10,"adaptation":160,"target_only":160,"pooled":80}},
            {"schema_version":"final_v2_2.evaluations.1","status":"SEALED_PLAN_NOT_EXECUTED","passes":ev,
             "counts":{"unique_passes":468,"budget_label_reuses":192,"reporting_cells":660}})
