"""NON_SCIENTIFIC_TEST_ONLY: one development city, real status, synthetic counts.

No scientific panel/fit artifact is read or written, and no final labels exist
on the permitted input path. Raw status reads are explicitly station-filtered.
"""
import unittest
from pathlib import Path

from research.v2_2.contract import Contract, ROOT, HOUR, canonical_bytes, sha256_bytes
from research.v2_2.history import StatusEvent, StatusIndex, IdealizedCountFeed, CausalHistory
from research.v2_2.snapshots import FitRequest, build_fit_snapshot, seal_training_keys


class DevelopmentStatusFixtureTests(unittest.TestCase):
    provenance = {}

    @classmethod
    def setUpClass(cls):
        import duckdb
        cls.contract = Contract()
        cls.city = 129
        cls.t = cls.contract.boundaries["HD"]
        ids = [s.station_id for s in cls.contract.city(cls.city)]
        low, high = cls.t-182*HOUR, cls.t+24*HOUR
        # Two old extrema retain lifetime evidence without loading complete bike/status histories.
        glob = (ROOT / "feasibility_test/full_audit/coverage/data/station_status/*.parquet").as_posix()
        query = f"""WITH scoped AS (
          SELECT station_id::BIGINT station_id, epoch_us(to_timestamp(time)) ts
          FROM read_parquet('{glob}')
          WHERE station_id IN ({','.join(map(str,ids))}) AND time IS NOT NULL
            AND isfinite(time) AND time <= {high}/1000000.0
        ), selected AS (
          SELECT station_id,ts FROM scoped WHERE ts >= {low}
          UNION SELECT station_id,min(ts) FROM scoped WHERE ts < {low} GROUP BY station_id
          UNION SELECT station_id,max(ts) FROM scoped WHERE ts < {low} GROUP BY station_id
        ) SELECT station_id,ts FROM selected WHERE ts IS NOT NULL ORDER BY station_id,ts"""
        con=duckdb.connect(":memory:",config={"threads":"1","temp_directory":""})
        try:
            rows=con.execute(query).fetchall()
        finally:
            con.close()
        if not rows:
            raise RuntimeError("Required development status fixture is missing; do not skip")
        assert all(i in ids for i,t in rows)
        cls.events=tuple(StatusEvent(i,t) for i,t in rows)
        cls.feed=IdealizedCountFeed(cls.contract, cls.contract.by_id,
                                   cls.contract.boundaries['H0'],cls.contract.boundaries['HF'],
                                   {(ids[0],cls.t-2*HOUR):3})
        cls.engine=CausalHistory(cls.contract,StatusIndex(cls.contract,cls.events),cls.feed)
        cls.provenance={"purpose":"NON_SCIENTIFIC_TEST_ONLY", "city_ids":[129],
            "station_count":len(ids),"status_event_count":len(rows),
            "recent_window_us":[low,high],"earlier_evidence":"per-station first/last before window",
            "status_rows_sha256":sha256_bytes(canonical_bytes(rows)),
            "source_manifest":"feasibility_test/full_audit/coverage/station_status_source_manifest.json",
            "source_manifest_sha256":sha256_bytes((ROOT/'feasibility_test/full_audit/coverage/station_status_source_manifest.json').read_bytes()),
            "query":query,"cutoffs_us":[cls.t,cls.t+HOUR],
            "count_values":"SYNTHETIC: one value=3; all other declared station-hours=0",
            "scientific_panel_accessed":False,"final_target_evaluation_labels_accessed":False,
            "model_calls":0,"snapshot_fixture_candidate_hours":1}

    def test_development_physical_prefix_and_future_append(self):
        fingerprints=[]
        for t in (self.t,self.t+HOUR):
            prefix=tuple(e for e in self.events if e.timestamp<=t)
            cut=CausalHistory(self.contract,StatusIndex(self.contract,prefix),self.feed)
            a,b=self.engine.at_origin(self.city,t),cut.at_origin(self.city,t)
            self.assertEqual(a,b)
            self.assertEqual(a.tensor_bytes(),b.tensor_bytes())
            self.assertGreater(sum(sum(m) for m in a.m_hist),0)
            self.assertLess(sum(sum(m) for m in a.m_hist),24*len(a.station_ids))
            fingerprints.append(a.fingerprint())
        self.provenance['history_fixture_sha256']=fingerprints

    def test_development_training_inference_array_schema(self):
        import numpy as np
        a=self.engine.for_training(self.city,self.t).arrays()
        b=self.engine.for_inference(self.city,self.t).arrays()
        self.assertEqual(a['x_hist'].shape,(24,76,2))
        self.assertEqual(a['x_week'].shape,(76,2))
        for k in a:self.assertEqual(a[k].tobytes(),b[k].tobytes())
        self.assertTrue(np.array_equal(a['m_hist'],a['x_hist'][...,1].astype(bool)))

    def test_development_one_hour_fit_snapshot_prefix(self):
        req=FitRequest.registered(self.contract,phase='development',kind='source',target_city=476)
        prefix=tuple(e for e in self.events if e.timestamp<=self.t)
        cut=CausalHistory(self.contract,StatusIndex(self.contract,prefix),self.feed)
        a=build_fit_snapshot(self.engine,req,candidate_origins=[self.t-3*HOUR])
        b=build_fit_snapshot(cut,req,candidate_origins=[self.t-3*HOUR])
        self.assertEqual(a,b)
        self.assertGreater(len(a.rows),0)
        self.assertTrue(all(r.city_id==129 for r in a.rows))
        self.assertEqual(seal_training_keys(a),seal_training_keys(b))
        self.provenance['snapshot_fixture_rows']=len(a.rows)
        self.provenance['snapshot_fixture_key_manifest_sha256']=seal_training_keys(a)[1]
