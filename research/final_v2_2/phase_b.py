"""Quarantined Phase-B gate; never imported by Phase-A execution modules."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import core,firewall,data_gate

def verify_gate(authorization,commitment,label_binding,package_sha,job_sha,data_sha,implementation_sha):
 cap=firewall.authorize_phase_b(Path(authorization),Path(commitment),Path(label_binding));auth=json.loads(Path(authorization).read_text(encoding='utf-8'))
 required={'package_manifest_sha256':package_sha,'job_manifest_sha256':job_sha,'final_data_manifest_sha256':data_sha,'implementation_contract_sha256':implementation_sha,'final_protocol_markdown_sha256':core.AUTHORITY['FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md'],'final_protocol_json_sha256':core.AUTHORITY['final_evaluation_protocol_v2_2.json']}
 for k,v in required.items():
  if auth.get(k)!=v:raise PermissionError('Phase-B binding mismatch: '+k)
 if core.sha(data_gate.FINAL_MANIFEST)!=data_sha or core.sha(core.SEAL+'/final_job_manifest.json')!=job_sha:raise PermissionError('bound artifact changed')
 return cap

def main():
 p=argparse.ArgumentParser()
 for x in ('authorization','commitment','label-binding','package-sha','job-sha','data-sha','implementation-sha'):p.add_argument('--'+x,required=True)
 a=p.parse_args();verify_gate(a.authorization,a.commitment,a.label_binding,a.package_sha,a.job_sha,a.data_sha,a.implementation_sha)
 raise RuntimeError('Gate verified; production materialization remains separately sealed and was not executed')

if __name__=='__main__':main()
