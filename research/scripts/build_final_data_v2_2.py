"""Build Phase-A final V2.2 data only; no current-target artifact is created."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'tmp/neural_build/pydeps'),str(ROOT)]
from research.final_v2_2.data_gate import build
if __name__=='__main__':print(build())
