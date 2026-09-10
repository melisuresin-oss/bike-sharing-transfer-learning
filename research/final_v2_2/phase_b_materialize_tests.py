import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from . import phase_b_materialize
class Tests(unittest.TestCase):
 def test_absent_authorization_refuses_before_raw_reader(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);[(p/n).write_text('{}') for n in ('a','c','l')]
   a=SimpleNamespace(authorization=p/'a',commitment=p/'c',label_binding=p/'l',package_sha='p',job_sha='j',data_sha='d',implementation_sha='i',output='x')
   with self.assertRaises(PermissionError):phase_b_materialize.materialize(a)
if __name__=='__main__':unittest.main(verbosity=2)
