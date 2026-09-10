"""Dormant R7 Phase-B capability gate. Not exposed by the Phase-A CLI."""
from __future__ import annotations
import hashlib,json,os
from dataclasses import dataclass
from pathlib import Path
from . import commitment,core

@dataclass(frozen=True)
class PhaseBCapability:
    authorization_sha256:str
    prediction_commitment_sha256:str
    opaque_target_binding_sha256:str
    opaque_target_canonical_path:str=""

def _regular_no_symlink(value)->Path:
    supplied=Path(value).absolute()
    if not supplied.exists() or not supplied.is_file(): raise PermissionError("bound Phase-B file missing/not regular")
    for part in (supplied,*supplied.parents):
        if part.exists() and part.is_symlink(): raise PermissionError("symlinked Phase-B path forbidden")
    resolved=supplied.resolve(strict=True)
    if resolved!=supplied: raise PermissionError("Phase-B path substitution forbidden")
    return resolved

def _hash_bytes(payload:bytes)->str: return hashlib.sha256(payload).hexdigest()

def _read_bound_json(value):
    path=_regular_no_symlink(value); payload=path.read_bytes(); return path,payload,json.loads(payload.decode("utf-8"))

def required_bindings(package_sha,commitment_path,opaque_target_binding):
    commitment_file=_regular_no_symlink(commitment_path); target_file=_regular_no_symlink(opaque_target_binding)
    return {"prediction_commitment_sha256":core.sha256_path(commitment_file),"package_manifest_sha256":package_sha,
        "job_manifest_sha256":core.FROZEN[core.JOB_MANIFEST],"evaluation_manifest_sha256":core.FROZEN[core.EVALUATION_MANIFEST],
        "final_data_manifest_sha256":core.FROZEN[core.DATA_MANIFEST],"implementation_contract_sha256":core.sha(core.IMPLEMENTATION_CONTRACT),
        "package_member_manifest_sha256":core.sha(core.MEMBER_MANIFEST),"final_protocol_markdown_sha256":core.FROZEN["FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md"],
        "final_protocol_json_sha256":core.FROZEN["final_evaluation_protocol_v2_2.json"],
        "grl_clarification_markdown_sha256":core.FROZEN["FINAL_GRL_SOURCE_DOMAIN_CARDINALITY_CLARIFICATION_V2_2.md"],
        "grl_clarification_json_sha256":core.FROZEN["final_grl_source_domain_cardinality_v2_2.json"],
        "executable_code_manifest_sha256":core.sha(core.CODE_MANIFEST),
        "opaque_held_out_target_binding_sha256":core.sha256_path(target_file),
        "opaque_held_out_target_canonical_path":str(target_file)}

def authorize(authorization_path,commitment_path,opaque_target_binding,package_sha):
    commitment.verify(str(_regular_no_symlink(commitment_path)),package_sha)
    auth_path,payload,auth=_read_bound_json(authorization_path)
    if auth.get("status")!="FINAL_V2_2_R7_R1_PHASE_B_EXPLICITLY_AUTHORIZED": raise PermissionError("explicit Phase-B authorization absent")
    required=required_bindings(package_sha,commitment_path,opaque_target_binding)
    for key,value in required.items():
        if key not in auth or auth[key]!=value: raise PermissionError("Phase-B binding mismatch: "+key)
    return PhaseBCapability(_hash_bytes(payload),required["prediction_commitment_sha256"],
        required["opaque_held_out_target_binding_sha256"],required["opaque_held_out_target_canonical_path"])

def invoke_target_reader(capability,authorization_path,commitment_path,opaque_target_binding,package_sha,reader):
    first=authorize(authorization_path,commitment_path,opaque_target_binding,package_sha)
    if first!=capability: raise PermissionError("capability mismatch")
    second=authorize(authorization_path,commitment_path,opaque_target_binding,package_sha)
    if second!=first: raise PermissionError("TOCTOU binding changed")
    target=_regular_no_symlink(opaque_target_binding)
    with target.open("rb") as handle:
        before=os.fstat(handle.fileno()); digest=hashlib.sha256()
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
        after=os.fstat(handle.fileno()); handle.seek(0)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise PermissionError("held-out target changed while binding")
        if digest.hexdigest()!=second.opaque_target_binding_sha256 or str(target)!=second.opaque_target_canonical_path:
            raise PermissionError("held-out target descriptor binding mismatch")
        return reader(second,handle)