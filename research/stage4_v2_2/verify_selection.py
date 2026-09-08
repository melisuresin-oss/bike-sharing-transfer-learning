"""Independent, synthetic-only reconstruction of the prospective ranking rule."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import hashlib
import json
import math
import random
import statistics
from research.stage4_v2_2 import selection as s

ROOT=Path(__file__).resolve().parents[2]
CLARIFICATION="research/results/stage4_selection_clarification_v2_2/selection_clarification.json"
CLARIFICATION_SHA="852d8c32dd9004a8dfaa17633e285877828672d79414cf5fb67b8550c8ea7d4d"


def independently_rank(candidates):
    """Use separate aggregation/formula and tuple sorting, not s.compare."""
    rows=[]
    for lam,schedule,folds in candidates:
        mean=math.fsum(folds)/4
        sd=math.sqrt(math.fsum((v-mean)**2 for v in folds)/4)
        q=Decimal(str(mean)).quantize(Decimal(".0001"),rounding=ROUND_HALF_UP)
        rows.append(((q,sd,lam,0 if schedule=="constant" else 1),s.candidate_id(lam,schedule)))
    return [cid for _,cid in sorted(rows)]


def run_checks():
    raw=(ROOT/CLARIFICATION).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==CLARIFICATION_SHA
    art=json.loads(raw)
    assert art["historically_defined"] is False and art["created_before_stage4_scientific_execution"]
    assert hashlib.sha256((ROOT/art["selection_code"]["path"]).read_bytes()).hexdigest()==art["selection_code"]["sha256"]
    passed=[]
    def case(name,items,expected):
        production=[r["candidate_id"] for r in s.rank_summaries([s.summarize(*x) for x in items])]
        assert production==independently_rank(items)==expected,name
        passed.append({"test":name,"status":"PASS"})
    case("clearly_different_primary",[(.01,"constant",[2]*4),(.5,"linear",[1]*4)], ["grl_l50_linear","grl_l01_constant"])
    case("unrounded_difference_tied_at_4dp",[(.01,"constant",[1.00004]*4),(.5,"linear",[1.00001]*4)], ["grl_l01_constant","grl_l50_linear"])
    case("primary_tie_lower_population_sd",[(.01,"constant",[0,1,1,2]),(.5,"linear",[1]*4)], ["grl_l50_linear","grl_l01_constant"])
    case("primary_and_sd_tie_lower_lambda",[(.5,"constant",[1]*4),(.1,"linear",[1]*4)], ["grl_l10_linear","grl_l50_constant"])
    case("final_constant_tie_break",[(.1,"linear",[1]*4),(.1,"constant",[1]*4)], ["grl_l10_constant","grl_l10_linear"])
    assert str(s.quantize_primary(1.23445))=="1.2345" and str(s.quantize_primary(1.23444))=="1.2344"
    assert str(s.quantize_primary(1.23455))=="1.2346"
    passed.append({"test":"ROUND_HALF_UP_boundaries","status":"PASS"})
    vals=[0.,0.,2.,2.];sd=s.summarize(.1,"linear",vals)["fold_population_sd_unrounded"]
    assert sd==1.0 and sd!=statistics.stdev(vals)
    passed.append({"test":"population_not_sample_sd","status":"PASS"})
    rows=[{"candidate_id":s.candidate_id(l,sc),"pseudo_target":f,"seed":seed,
           "regime":"post_7d_adaptation","mae":float(1+i)}
          for l in s.LAMBDAS for sc in s.SCHEDULES for i,f in enumerate(s.FOLDS) for seed in s.SEEDS]
    first=s.select(rows)
    for seed in range(8):random.Random(seed).shuffle(rows);assert s.select(rows)==first
    assert first["ranking"][0]["primary_unrounded"]==2.5
    assert first["ranking"][0]["fold_population_sd_unrounded"]==math.sqrt(1.25)
    passed.append({"test":"input_order_independent_72_row_aggregation","status":"PASS"})
    return {"status":"PASS","synthetic_tests":8,"checks":passed,"clarification_sha256":CLARIFICATION_SHA,
            "real_scientific_optimizer_updates":0,"real_scientific_evaluations":0,
            "stage4_scientific_results_inspected":False,"final_target_labels_accessed":False}


if __name__=="__main__":print(json.dumps(run_checks(),indent=2))
