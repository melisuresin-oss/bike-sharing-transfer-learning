"""Dormant held-out materializer. Import is inert; execution requires the full Phase-B capability."""
from __future__ import annotations
import argparse,json,os,uuid
from pathlib import Path
import numpy as np
from research.v2_2.contract import Contract,HOUR
from research.v2_2.history import StatusEvent,StatusIndex,coverage
from . import core,data_gate,phase_b

def materialize(args):
 phase_b.verify_gate(args.authorization,args.commitment,args.label_binding,args.package_sha,args.job_sha,args.data_sha,args.implementation_sha)
 commitment=json.loads(Path(args.commitment).read_text(encoding='utf-8'));pred=commitment.get('prediction_manifest',{})
 pred_path=core.repo_path(pred.get('path',''))
 if core.sha256_path(pred_path)!=pred.get('sha256'):raise PermissionError('prediction manifest changed after commitment')
 if commitment.get('package_manifest_sha256')!=args.package_sha or commitment.get('final_labels_opened_before_commitment') is not False:raise PermissionError('invalid prediction commitment')
 manifest=core.read_json(data_gate.FINAL_MANIFEST);import duckdb
 con=duckdb.connect();rows=[]
 for rec in manifest['targets']:
  city=rec['city_id'];keys=np.load(core.repo_path(rec['prediction_keys']['path']),allow_pickle=False);ids=sorted(set(map(int,keys['station_id'])));sqlids=','.join(map(str,ids))
  sq=f"SELECT DISTINCT station_id::BIGINT,epoch_us(to_timestamp(time)) FROM read_parquet('{data_gate.STATUS_GLOB}') WHERE station_id IN ({sqlids}) AND time IS NOT NULL AND isfinite(time) ORDER BY 1,2"
  events=[StatusEvent(int(i),int(t)) for i,t in con.execute(sq).fetchall()];status=StatusIndex(Contract(),events);adjudication=max(e.timestamp for e in events)
  tq=f"SELECT station_id_start::BIGINT,epoch_us(date_trunc('hour',to_timestamp(time_start))),count(*)::BIGINT FROM read_parquet('{data_gate.TRIPS_GLOB}') WHERE city_id={city} AND station_id_start IN ({sqlids}) AND time_start>={core.BOUNDARIES['HT']}/1000000.0 AND time_start<{core.BOUNDARIES['HE']}/1000000.0 GROUP BY 1,2"
  counts={(int(i),int(t)):int(v) for i,t,v in con.execute(tq).fetchall()};c=Contract()
  for sid,t in zip(keys['station_id'],keys['forecast_origin_us']):
   sid,t=int(sid),int(t);cv=coverage(c,status,city,t,adjudication);ok=cv.observed[[s.station_id for s in c.city(city)].index(sid)]
   rows.append((city,sid,t,counts.get((sid,t),0) if ok else -1,bool(ok)))
 con.close();out=core.repo_path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 pending=out.with_name(out.name+'.pending-'+uuid.uuid4().hex+'.npz');np.savez_compressed(pending,city_id=np.array([r[0] for r in rows],np.int64),station_id=np.array([r[1] for r in rows],np.int64),forecast_origin_us=np.array([r[2] for r in rows],np.int64),count=np.array([r[3] for r in rows],np.int64),observed=np.array([r[4] for r in rows],bool));pending.replace(out)
 return {'status':'PHASE_B_TARGETS_MATERIALIZED_AFTER_COMMITMENT','path':args.output,'sha256':core.sha256_path(out),'rows':len(rows),'scientific_summaries_emitted':False}

def main():
 p=argparse.ArgumentParser()
 for x in ('authorization','commitment','label-binding','package-sha','job-sha','data-sha','implementation-sha','output'):p.add_argument('--'+x,required=True)
 print(json.dumps(materialize(p.parse_args()),indent=2))
if __name__=='__main__':main()
