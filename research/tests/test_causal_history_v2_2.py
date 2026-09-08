"""NON_SCIENTIFIC_TEST_ONLY: synthetic adversarial fixtures, no model calls."""
from dataclasses import asdict, replace
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from research.v2_2.contract import (Contract, HOUR, BRACKET, ROOT, BASE, COHORT_SHA256,
                                   SEAL_SHA256, canonical_bytes, hour_us, sha256_bytes, utc_us)
from research.v2_2.history import (StatusEvent, StatusIndex, IdealizedCountFeed, CausalHistory,
                                  coverage, city_gate, count_available)
from research.v2_2.artifacts import ArtifactIdentity, ArtifactLoader, envelope, validate_envelope
from research.v2_2.snapshots import (FitRequest, build_fit_snapshot, seal_training_keys,
                                    require_training_keys, historical_average_rows)


class CausalHistoryV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = Contract()
        cls.t = cls.contract.boundaries["HD"]
        cls.ids = [s.station_id for s in cls.contract.city(129)]

    def engine(self, events=None, counts=None, *, t=None):
        t = self.t if t is None else t
        if events is None:
            events = [StatusEvent(self.ids[0], t + k*HOUR) for k in range(-181, 15)]
        feed = IdealizedCountFeed(self.contract, self.contract.by_id,
                                 self.contract.boundaries["H0"], self.contract.boundaries["HF"],
                                 counts or {})
        return CausalHistory(self.contract, StatusIndex(self.contract, events), feed)

    def test_01_future_append_own_station(self):
        h = self.t - HOUR
        events = [StatusEvent(self.ids[0], h-HOUR)]
        before, after = self.engine(events), self.engine(events+[StatusEvent(self.ids[0], self.t+1)])
        self.assertEqual(before.at_origin(129,self.t), after.at_origin(129,self.t))
        self.assertFalse(after.at_origin(129,self.t).m_hist[0][0])
        # An oracle that uses later evidence would accept the same historical hour.
        self.assertTrue(coverage(self.contract,after.status,129,h,self.t+1).observed[0])

    def test_01b_future_append_other_station_denominator(self):
        h = self.t-2*HOUR
        ev = [StatusEvent(self.ids[0],h-HOUR),StatusEvent(self.ids[0],h+HOUR)]
        ev += [StatusEvent(i,h-2*HOUR) for i in self.ids[1:4]]
        a = self.engine(ev)
        later = self.t+20*HOUR
        b = self.engine(ev+[StatusEvent(i,later) for i in self.ids[1:4]])
        self.assertEqual(a.at_origin(129,self.t),b.at_origin(129,self.t))
        self.assertTrue(coverage(self.contract,a.status,129,h,self.t).observed[0])
        self.assertFalse(coverage(self.contract,b.status,129,h,later).observed[0])

    def test_02_status_prefix_equivalence(self):
        events=[StatusEvent(self.ids[0],self.t+k*HOUR) for k in range(-181,30)]
        full=self.engine(events).at_origin(129,self.t)
        prefix=self.engine([e for e in events if e.timestamp<=self.t]).at_origin(129,self.t)
        self.assertEqual(full,prefix)
        self.assertEqual(full.tensor_bytes(),prefix.tensor_bytes())

    def test_03_all_status_lineage_at_or_before_origin(self):
        history=self.engine().at_origin(129,self.t)
        for cv in history.lineage:
            for row in cv.evidence:
                for value in (row.previous,row.next,row.first,row.last):
                    self.assertTrue(value is None or value<=self.t)

    def test_04_exact_backward_forward_boundaries(self):
        h=self.t-13*HOUR
        for previous_excess,next_excess,expected in [(0,0,True),(1,0,False),(0,1,False)]:
            with self.subTest(previous_excess=previous_excess,next_excess=next_excess):
                ev=[StatusEvent(self.ids[0],h-BRACKET-previous_excess),
                    StatusEvent(self.ids[0],h+HOUR+BRACKET+next_excess)]
                cv=coverage(self.contract,self.engine(ev).status,129,h,self.t+1)
                self.assertEqual(cv.observed[0],expected)
        at=self.engine([StatusEvent(self.ids[0],h),StatusEvent(self.ids[0],h+HOUR)])
        self.assertTrue(coverage(self.contract,at.status,129,h,h+HOUR).observed[0])
        self.assertFalse(coverage(self.contract,at.status,129,h,h+HOUR-1).observed[0])

    def test_05_train_inference_and_chronological_parity(self):
        counts={(self.ids[0],self.t-k*HOUR):k for k in range(1,25)}
        engine=self.engine(counts=counts)
        training,inference=engine.for_training(129,self.t),engine.for_inference(129,self.t)
        self.assertEqual(training.tensor_bytes(),inference.tensor_bytes())
        self.assertEqual(training.chronological[0],training.x_hist[23])
        self.assertEqual(training.chronological[-1],training.x_hist[0])
        self.assertGreater(training.chronological[0][0][0],training.chronological[-1][0][0])
        for row,mask in zip(training.x_hist,training.m_hist):
            self.assertEqual(tuple(bool(pair[1]) for pair in row),mask)

    def test_06_fit_cutoff_append_invariance(self):
        request=FitRequest.registered(self.contract,phase="development",kind="source",target_city=476)
        ev=[StatusEvent(self.ids[0],self.t+k*HOUR) for k in range(-181,1)]
        late=ev+[StatusEvent(i,self.t+20*HOUR) for i in self.ids]
        a=build_fit_snapshot(self.engine(ev),request,candidate_origins=[self.t-3*HOUR])
        b=build_fit_snapshot(self.engine(late),request,candidate_origins=[self.t-3*HOUR])
        self.assertEqual(a,b)
        self.assertEqual(seal_training_keys(a),seal_training_keys(b))

    def test_07_fit_label_changes_do_not_rewrite_predictors(self):
        origin=self.t-2*HOUR
        ev=[StatusEvent(self.ids[0],origin-HOUR)]
        a=self.engine(ev); b=self.engine(ev+[StatusEvent(self.ids[0],origin+HOUR)])
        request=FitRequest.registered(self.contract,phase="development",kind="source",target_city=476)
        before=a.at_origin(129,origin)
        self.assertEqual(before,b.at_origin(129,origin))
        self.assertEqual(len(build_fit_snapshot(a,request,candidate_origins=[origin]).rows),0)
        self.assertEqual(len(build_fit_snapshot(b,request,candidate_origins=[origin]).rows),1)
        self.assertEqual(before,a.at_origin(129,origin))

    def test_07b_retrospective_adjudication_is_a_separate_type(self):
        from research.v2_2.labels import RetrospectiveLabel, RetrospectiveLabelSnapshot
        engine=self.engine(); before=engine.at_origin(129,self.t)
        unknown=RetrospectiveLabelSnapshot(self.t,(RetrospectiveLabel(129,self.ids[0],self.t,None,False),))
        observed=RetrospectiveLabelSnapshot(self.t+20*HOUR,(RetrospectiveLabel(129,self.ids[0],self.t,17,True),))
        self.assertNotEqual(unknown,observed)
        self.assertEqual(before,engine.at_origin(129,self.t))
        with self.assertRaises(TypeError):CausalHistory(self.contract,engine.status,observed)

    def loader(self,path,identity=None):
        identity=identity or ArtifactIdentity.create(self.contract,"predictor_cache",self.t,[129])
        index={"protocol_version":"2.2","specification_seal_sha256":SEAL_SHA256,
               "artifacts":{"fixture":{"identity":identity.as_json(),"path":path,"sha256":"0"*64}}}
        return ArtifactLoader(ROOT,index,sha256_bytes(canonical_bytes(index))),identity

    def test_08_final_label_firewall_before_open(self):
        for path in ["processed/protocol_v2_1/final_labels/SEALED_final_evaluation_labels.parquet",
                     "final_evaluation_labels.json","../outside.json"]:
            with self.subTest(path=path),patch.object(Path,"read_bytes",side_effect=AssertionError("opened")):
                loader,identity=self.loader(path)
                with self.assertRaises((PermissionError,ValueError)):
                    loader.load("fixture",identity)
        with self.assertRaises(ValueError):
            ArtifactIdentity.create(self.contract,"evaluation_labels",self.t,[129])

    def test_09_zero_denominator_is_unknown(self):
        engine=self.engine([])
        cv=coverage(self.contract,engine.status,129,self.t-HOUR,self.t)
        self.assertEqual((cv.denominator,cv.numerator,cv.city_observed),(0,0,False))
        self.assertFalse(any(cv.observed))
        self.assertEqual(engine.at_origin(129,self.t).x_hist[0][0],(0,0))

    def test_10_integer_odd_even_thresholds(self):
        for d,n in [(1,1),(2,1),(3,2),(4,2),(5,3)]:
            self.assertTrue(city_gate(n,d));self.assertFalse(city_gate(n-1,d))
        self.assertFalse(city_gate(0,0))
        with self.assertRaises(ValueError):city_gate(2,1)

    def test_11_positive_zero_unknown_and_no_exception(self):
        engine=self.engine(counts={(self.ids[0],self.t-2*HOUR):7,(self.ids[1],self.t-HOUR):12})
        history=engine.at_origin(129,self.t)
        self.assertGreater(history.x_hist[1][0][0],0)
        self.assertEqual(history.x_hist[1][0][1],1)
        self.assertEqual(history.x_hist[0][0],(0,1))
        self.assertEqual(history.x_hist[0][1],(0,0))

    def test_12_duplicate_missing_and_order_determinism(self):
        events=[StatusEvent(self.ids[0],self.t+k*HOUR) for k in range(-181,1)]
        dirty=list(reversed(events+events))+[StatusEvent(self.ids[0],None),StatusEvent(-999,self.t)]
        self.assertEqual(self.engine(events).at_origin(129,self.t),self.engine(dirty).at_origin(129,self.t))
        with self.assertRaises(TypeError):StatusIndex(self.contract,[{"target_12h":4}])

    def test_13_dst_elapsed_hour_and_week(self):
        for date in ["2023-03-26T00:00:00Z","2022-10-30T00:00:00Z"]:
            t=hour_us(date)
            engine=self.engine(t=t)
            a,b=engine.at_origin(129,t),engine.at_origin(129,t+HOUR)
            self.assertEqual(a.lineage[24].hour_start,t-168*HOUR)
            self.assertEqual(b.lineage[24].hour_start,a.lineage[24].hour_start+HOUR)
            self.assertNotEqual(a.x_calendar[-1],b.x_calendar[-1])
            if date.startswith("2022"):
                self.assertEqual(a.x_calendar[:2],b.x_calendar[:2])

    def test_14_fixed_799_cohort_and_foreign_station_invariance(self):
        self.assertEqual(len(self.contract.stations),799)
        a=self.engine([]).at_origin(129,self.t)
        b=self.engine([StatusEvent(999999999,self.t)]).at_origin(129,self.t)
        self.assertEqual(a,b)
        rows=[asdict(s) for s in self.contract.city(129)]
        with self.assertRaises(ValueError):self.contract.validate_static_rows(129,rows[:-1])

    def test_15_static_metadata_hash_and_fields(self):
        self.contract.validate_static_rows(129,self.contract.city(129))
        for field,value in [("bike_racks",999),("timezone","UTC"),("latitude",0.0),("city_id",194)]:
            rows=[asdict(s) for s in self.contract.city(129)];rows[0][field]=value
            with self.assertRaises(ValueError):self.contract.validate_static_rows(129,rows)
        read=Path.read_bytes
        def altered(path):
            data=read(path)
            return data+b" " if path.name=="fixed_cohort_static_manifest.json" else data
        with patch.object(Path,"read_bytes",altered),self.assertRaises(ValueError):Contract()

    def test_16_completed_count_feed_no_eight_hour_delay(self):
        h=self.t-HOUR
        self.assertTrue(count_available(h,self.t))
        self.assertFalse(count_available(h,self.t-1))
        engine=self.engine()
        self.assertEqual(engine.at_origin(129,self.t).x_hist[0][0],(0,1))
        with self.assertRaises(ValueError):engine.counts.count(self.ids[0],h,self.t-1)

    def test_17_no_retrospective_y_dependency(self):
        from research.v2_2 import history
        tree=ast.parse(Path(history.__file__).read_text(encoding="utf-8"))
        forbidden={"target_12h","target_24h","coverage_observed_12h","retrospective_Y","y","m_target"}
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute):self.assertNotIn(node.attr,forbidden)
            if isinstance(node,ast.Constant) and isinstance(node.value,str):self.assertNotIn(node.value,forbidden)
            if isinstance(node,ast.ImportFrom):
                self.assertNotIn(node.module,{"snapshots","research.data","research.data.window_dataset"})
        self.assertNotIn("read_parquet",Path(history.__file__).read_text(encoding="utf-8"))

    def test_18_sealed_keys_required_for_neural_and_ha(self):
        request=FitRequest.registered(self.contract,phase="development",kind="source",target_city=476)
        snapshot=build_fit_snapshot(self.engine(),request,candidate_origins=[self.t-3*HOUR])
        with self.assertRaises(PermissionError):require_training_keys(snapshot,None,"",scientific=False)
        manifest,digest=seal_training_keys(snapshot)
        self.assertEqual(require_training_keys(snapshot,manifest,digest,scientific=False),snapshot.rows)
        self.assertEqual(historical_average_rows(snapshot,manifest,digest,scientific=False),snapshot.rows)
        with self.assertRaises(PermissionError):require_training_keys(snapshot,manifest,digest,scientific=True)
        changed=replace(snapshot,rows=(replace(snapshot.rows[0],count=99),))
        with self.assertRaises(PermissionError):require_training_keys(changed,manifest,digest,scientific=False)

    def test_extra_all_registered_budgets(self):
        target=self.contract.city(476)[0].station_id
        engine=self.engine([StatusEvent(target,self.t+k*HOUR) for k in range(-181,1)])
        for budget in ["zero","1","7","30","full"]:
            request=FitRequest.registered(self.contract,phase="development",kind="adaptation",budget=budget,target_city=476)
            snap=build_fit_snapshot(engine,request,candidate_origins=[self.t-3*HOUR])
            self.assertEqual(len(snap.rows),0 if budget=="zero" else 1)
        request=FitRequest.registered(self.contract,phase="development",kind="adaptation",budget="1",target_city=476)
        with self.assertRaises(ValueError):build_fit_snapshot(engine,request,candidate_origins=[self.t-25*HOUR])
        with self.assertRaises(ValueError):build_fit_snapshot(engine,replace(request,start=request.start-HOUR),candidate_origins=[])

    def test_extra_identity_cross_loading_and_graph(self):
        identity=ArtifactIdentity.create(self.contract,"checkpoint",self.t,[129],model_family="graph_gru",graph_binding_sha256="a"*64)
        value=envelope(identity,{"fixture":"no model weights"},purpose="NON_SCIENTIFIC_TEST_ONLY")
        self.assertEqual(validate_envelope(value,identity),value["payload"])
        for field,bad in [("protocol_version","2.1"),("specification_sha256","0"*64),
                          ("cohort_static_sha256","0"*64),("feature_function_version","v2.1"),
                          ("cutoff_us",self.t+1),("roster_sha256","0"*64),("graph_binding_sha256","b"*64)]:
            mutated=copy.deepcopy(value);mutated["identity"][field]=bad
            with self.assertRaises(ValueError):validate_envelope(mutated,identity)
        with self.assertRaises(ValueError):ArtifactIdentity.create(self.contract,"checkpoint",self.t,[129],model_family="graph_gru")
        loader,identity=self.loader("processed/protocol_v2_1/cache.json")
        with self.assertRaises(PermissionError):loader.path("fixture",identity)

    def test_extra_count_feed_domain_and_copy(self):
        counts={(self.ids[0],self.t-HOUR):3}
        engine=self.engine(counts=counts);before=engine.at_origin(129,self.t)
        counts[(self.ids[0],self.t-HOUR)]=999
        self.assertEqual(before,engine.at_origin(129,self.t))
        with self.assertRaises(ValueError):engine.counts.count(self.ids[0],self.contract.boundaries['H0']-HOUR,self.t)
        with self.assertRaises(ValueError):self.engine(counts={(self.ids[0],self.t-HOUR):-1})

    def test_extra_timestamp_precision_and_naive_rejection(self):
        self.assertEqual(utc_us("2023-01-01T00:00:00.000001Z")-utc_us("2023-01-01T00:00:00Z"),1)
        with self.assertRaises(ValueError):utc_us("2023-01-01T00:00:00")
        with self.assertRaises(ValueError):utc_us(1234.5)
        with self.assertRaises(ValueError):hour_us(self.t+1)
        with self.assertRaises(ValueError):utc_us("2023-01-01T00:00:00.0000001Z")

    def test_extra_artifact_loader_content_hash_and_schema(self):
        identity=ArtifactIdentity.create(self.contract,"predictor_cache",self.t,[129])
        data=canonical_bytes(envelope(identity,{"fixture":"no predictions"},purpose="NON_SCIENTIFIC_TEST_ONLY"))
        index={"protocol_version":"2.2","specification_seal_sha256":SEAL_SHA256,
               "artifacts":{"fixture":{"identity":identity.as_json(),"path":"fixture.json","sha256":sha256_bytes(data)}}}
        loader=ArtifactLoader(ROOT,index,sha256_bytes(canonical_bytes(index)))
        with patch.object(Path,"read_bytes",return_value=data):
            self.assertEqual(loader.load("fixture",identity),{"fixture":"no predictions"})
        with patch.object(Path,"read_bytes",return_value=data+b" "),self.assertRaises(ValueError):
            loader.load("fixture",identity)
        with self.assertRaises(ValueError):replace(identity,protocol_version="2.1")
        legacy=copy.deepcopy(index);legacy['protocol_version']='2.1'
        with self.assertRaises(ValueError):ArtifactLoader(ROOT,legacy,sha256_bytes(canonical_bytes(legacy)))


if __name__=="__main__":
    unittest.main()
