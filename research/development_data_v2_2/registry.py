"""Read-only development panel access with separate predictor and label roles."""
from pathlib import Path
import hashlib
import json
import numpy as np
from research.v2_2.contract import Contract, DEVELOPMENT, PSEUDO_TARGETS, SPEC_SHA256, SEAL_SHA256, COHORT_SHA256, canonical_bytes
from research.v2_2.artifacts import ArtifactIdentity
from research.v2_2.snapshots import FitRequest, FitRow, FitSnapshot, require_training_keys, seal_training_keys

IMPLEMENTATION_SHA256="a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d"
PREDICTOR_FIELDS={"origin_us","station_ids","x_hist","m_hist","x_week","x_static","x_calendar","origin_feature_sha256"}


class DevelopmentRegistry:
    def __init__(self,root,manifest_path,expected_manifest_sha256):
        self.root=Path(root).resolve()
        data=self.path(manifest_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected_manifest_sha256:
            raise ValueError("Unbound development manifest")
        self.manifest=json.loads(data)
        m=self.manifest
        if (m.get("protocol_version")!="2.2" or m.get("scope")!="DEVELOPMENT_ONLY"
            or m.get("specification_sha256")!=SPEC_SHA256 or m.get("specification_seal_sha256")!=SEAL_SHA256
            or m.get("cohort_static_sha256")!=COHORT_SHA256
            or m.get("implementation_manifest_sha256")!=IMPLEMENTATION_SHA256
            or m.get("development_cities")!=list(DEVELOPMENT)):
            raise ValueError("Legacy or mismatched development authority")
        self.contract=Contract(self.root)
        self.panels={r["city_id"]:r for r in m["panels"]}
        if set(self.panels)!=set(DEVELOPMENT): raise ValueError("Development membership differs")
        self.artifacts={r["path"]:r for r in m["artifacts"]}

    def path(self,relative):
        if not isinstance(relative,str) or "\\" in relative or ":" in relative or ".." in Path(relative).parts:
            raise PermissionError("Only repository-relative paths allowed")
        path=(self.root/relative).resolve()
        rel=path.relative_to(self.root).as_posix().lower()
        if not rel.startswith(("processed/protocol_v2_2/","research/results/causal_history_audit_v2_2/")):
            raise PermissionError("Unapproved data authority or legacy path")
        if any(x in rel for x in ("final_label","evaluation_label","final_adaptation","final_features","sealed_final")):
            raise PermissionError("Final-label firewall")
        return path

    def bound(self,entry):
        path=self.path(entry["path"])
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
        if h.hexdigest()!=entry["sha256"]: raise ValueError("Artifact bytes changed")
        return path

    def arrays(self,entry):
        with np.load(self.bound(entry),allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def predictors(self,city):
        if city not in DEVELOPMENT: raise PermissionError("Development cities only")
        entry=self.panels[city]
        if entry["path"]!=f"processed/protocol_v2_2/development/city_{city}_predictors.npz":
            raise PermissionError("Predictor reader cannot open a label or fitting partition")
        expected=ArtifactIdentity.create(self.contract,"predictor_cache",self.contract.boundaries["HF"],[city])
        expected.validate(entry["identity"])
        data=self.arrays(entry)
        if set(data)!=PREDICTOR_FIELDS: raise ValueError("Predictor schema includes unauthorized/missing fields")
        if data["station_ids"].tolist()!=[s.station_id for s in self.contract.city(city)]:
            raise ValueError("Frozen node order mismatch")
        return data

    def json_artifact(self,path):
        return json.loads(self.bound(self.artifacts[path]).read_text(encoding="utf-8"))

    def graph(self,city,k):
        """The sole legacy-data exception is explicitly reconciled static adjacency."""
        if city not in DEVELOPMENT or k not in (4,8,16):
            raise PermissionError("Unregistered development graph")
        authority=self.json_artifact("research/results/causal_history_audit_v2_2/graph_static_reconciliation.json")
        entry=next(r for r in authority["graphs"] if r["city_id"]==city and r["k"]==k)
        expected=f"tmp/stage2_graph_grid_cache_v2_1/k_{k:02d}/city_{city}/adjacency.npy"
        if entry["path"]!=expected or entry["v2_2_roster_sha256"]!=self.contract.city_roster_hash(city):
            raise ValueError("Static graph exception binding mismatch")
        path=(self.root/expected).resolve(); path.relative_to(self.root)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("Reused static graph bytes changed")
        identity=ArtifactIdentity.create(self.contract,"graph",self.contract.boundaries["H0"],[city],
                    graph_binding_sha256=entry["graph_config"]["adjacency_hash_sha256"])
        return np.load(path,allow_pickle=False),identity

    def retrospective_labels(self,city):
        if city not in DEVELOPMENT: raise PermissionError("Development labels only")
        authority=self.json_artifact("processed/protocol_v2_2/development_labels/retrospective_label_manifest.json")
        entry=next(r for r in authority["labels"] if r["city_id"]==city)
        if entry["role"]!="DEVELOPMENT_RETROSPECTIVE_LABELS": raise ValueError("Label role mismatch")
        return self.arrays(entry)

    def sealed_fit(self,*,target,budget=None):
        """Return only rows admitted by the sealed implementation fitting gate.

        budget=None: seven-source fold; otherwise a registered adaptation budget.
        The zero-budget object cannot authorize a fitting operation.
        """
        if target not in PSEUDO_TARGETS: raise PermissionError("Pseudo-target development only")
        authority=self.json_artifact("processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json")
        if budget is None:
            entry=next(r for r in authority["source_folds"] if r["target_city"]==target)
        else:
            entry=next(r for r in authority["adaptations"] if r["city_id"]==target and r["budget"]==budget)
        info=json.loads(self.bound(entry).read_text(encoding="utf-8"))
        q=dict(info["request"]); q["city_ids"]=tuple(q["city_ids"]); req=FitRequest(**q); req.validate(self.contract)
        expected=FitRequest.registered(self.contract,phase="development",kind="source" if budget is None else "adaptation",
                                      budget="full" if budget is None else budget,target_city=target)
        if req!=expected: raise ValueError("Fit scope/cutoff mismatch")
        parts=info["source_partitions"] if budget is None else [info["data"]]
        rows=[]
        for part in parts:
            a=self.arrays(part)
            if not np.all(a["observed"]): raise ValueError("Sealed eligible rows must be observed")
            rows.extend(FitRow(int(i),int(s),int(t),int(y),h.decode()) for i,s,t,y,h in zip(
                a["city_id"],a["station_id"],a["origin_us"],a["count"],a["origin_feature_sha256"]))
        snap=FitSnapshot(req,ArtifactIdentity.create(self.contract,"fit_snapshot",req.cutoff,req.city_ids),tuple(rows),"SCIENTIFIC_ARTIFACT")
        if budget=="zero":
            if rows: raise ValueError("Zero-budget contamination")
            seal,digest=seal_training_keys(snap)
            if seal!=info["training_key_seal"] or digest!=info["training_key_seal_sha256"]:
                raise ValueError("Zero-budget seal mismatch")
            return snap
        require_training_keys(snap,info["training_key_seal"],info["training_key_seal_sha256"],scientific=True)
        return snap
