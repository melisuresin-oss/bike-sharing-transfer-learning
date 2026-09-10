"""Windows-safe Phase-A builder wrapper; scientific semantics are unchanged."""
from pathlib import Path
import os,sys,uuid
ROOT=Path(__file__).resolve().parents[2];sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'tmp/neural_build/pydeps'),str(ROOT)]
import numpy as np
from research.final_v2_2 import core,data_gate

def atomic_npz(relative,**arrays):
 p=core.repo_path(relative);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(relative)
 pending=p.with_name(p.name+'.pending-'+uuid.uuid4().hex+'.npz')
 np.savez_compressed(pending,**arrays)
 with pending.open('r+b') as f:f.flush();os.fsync(f.fileno())
 pending.replace(p)
 return {'path':relative,'bytes':p.stat().st_size,'sha256':core.sha256_path(p)}

data_gate.atomic_npz=atomic_npz
if __name__=='__main__':print(data_gate.build())
