from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from research.v2_2.contract import Contract,HOUR,BRACKET
from research.v2_2.history import StatusEvent,StatusIndex,IdealizedCountFeed,CausalHistory,coverage
from research.v2_2.snapshots import FitRequest
from . import core,firewall,phase_b

class DataGateLeakageTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=Contract();cls.city=195;cls.ids=[s.station_id for s in cls.c.city(cls.city)];cls.sid=cls.ids[0];cls.t=cls.c.boundaries['HT']
 def status(self,events):return StatusIndex(self.c,[StatusEvent(self.sid,x) for x in events])
 def test_next_after_origin_unavailable(self):
  s=self.status([self.t,self.t+HOUR+1]);self.assertFalse(coverage(self.c,s,self.city,self.t,self.t+HOUR).evidence[0].bracket)
 def test_next_exactly_asof_allowed(self):
  s=self.status([self.t,self.t+HOUR]);self.assertTrue(coverage(self.c,s,self.city,self.t,self.t+HOUR).evidence[0].bracket)
 def test_exact_12h_allowed(self):
  s=self.status([self.t-BRACKET,self.t+HOUR+BRACKET]);self.assertTrue(coverage(self.c,s,self.city,self.t,self.t+HOUR+BRACKET).evidence[0].bracket)
 def test_12h_plus_microsecond_rejected(self):
  s=self.status([self.t-BRACKET-1,self.t+HOUR]);self.assertFalse(coverage(self.c,s,self.city,self.t,self.t+HOUR).evidence[0].bracket)
 def test_lag_retrospective_future_status_unavailable(self):
  events=[]
  for sid in self.ids:events += [StatusEvent(sid,self.t-HOUR),StatusEvent(sid,self.t+1)]
  feed=IdealizedCountFeed(self.c,self.ids,self.t-2*HOUR,self.t+HOUR,{(self.sid,self.t-HOUR):3})
  h=CausalHistory(self.c,StatusIndex(self.c,events),feed).at_origin(self.city,self.t)
  self.assertFalse(h.m_hist[0][0])
 def test_previous_hour_available_after_end(self):
  events=[]
  for sid in self.ids:events += [StatusEvent(sid,self.t-HOUR),StatusEvent(sid,self.t)]
  feed=IdealizedCountFeed(self.c,self.ids,self.t-2*HOUR,self.t+HOUR,{(self.sid,self.t-HOUR):2})
  h=CausalHistory(self.c,StatusIndex(self.c,events),feed).at_origin(self.city,self.t);self.assertEqual(h.x_hist[0][0][1],1.0)
 def test_current_and_future_not_in_history(self):
  events=[]
  for sid in self.ids:events += [StatusEvent(sid,self.t-HOUR),StatusEvent(sid,self.t)]
  feed=IdealizedCountFeed(self.c,self.ids,self.t-2*HOUR,self.t+2*HOUR,{(self.sid,self.t-HOUR):1,(self.sid,self.t):99,(self.sid,self.t+HOUR):88})
  h=CausalHistory(self.c,StatusIndex(self.c,events),feed).at_origin(self.city,self.t);self.assertAlmostEqual(h.x_hist[0][0][0],np.log1p(1),places=6);self.assertNotIn(np.log1p(99),[p[0] for row in h.x_hist for p in row])
 def test_zero_vs_unknown(self):
  events=[]
  for sid in self.ids:events += [StatusEvent(sid,self.t-HOUR),StatusEvent(sid,self.t)]
  h=CausalHistory(self.c,StatusIndex(self.c,events),IdealizedCountFeed(self.c,self.ids,self.t-2*HOUR,self.t+HOUR,{})).at_origin(self.city,self.t)
  self.assertEqual(h.x_hist[0][0],(0.0,1.0));self.assertEqual(h.x_hist[1][0],(0.0,0.0))
 def test_budget_not_extended(self):
  r=FitRequest.registered(self.c,phase='final',kind='adaptation',budget='7',target_city=195);self.assertEqual(r.cutoff-r.start,7*24*HOUR)
 def test_fixed_absent_station_kept_in_denominator(self):
  cv=coverage(self.c,self.status([]),self.city,self.t,self.t);self.assertEqual(len(cv.evidence),len(self.ids))
 def test_phase_a_import_no_label_reader(self):
  import research.final_v2_2.data_gate as d;self.assertFalse(hasattr(d,'materialize_labels'))
 def test_phase_b_missing_capability_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for n in ('a','c','l'):(p/n).write_text('{}')
   with self.assertRaises(PermissionError):phase_b.verify_gate(p/'a',p/'c',p/'l','x','y','z','q')
 def test_phase_a_path_rejects_labels(self):self.assertRaises(PermissionError,firewall.phase_a_path,'processed/protocol_v2_2/final_labels/x')

if __name__=='__main__':unittest.main(verbosity=2)
