# V2.1 protocol dataset builder

This folder contains the pre-model data-construction gate governed by
`RESEARCH_PROTOCOL_V2_1.md`. It builds no forecasting model and performs no
model selection.

## Build

From the repository root:

```powershell
python -m pip install -r research/requirements.txt
python research/scripts/build_protocol_dataset.py
```

The default output is `processed/protocol_v2_1/`. Existing output is preserved
unless `--overwrite` is supplied explicitly.

## Validate

Run an independent second build and compare deterministic hashes:

```powershell
python research/scripts/build_protocol_dataset.py `
  --output tmp/protocol_v2_1_repro --overwrite
python research/scripts/compare_protocol_builds.py `
  processed/protocol_v2_1 tmp/protocol_v2_1_repro
python -m unittest discover -s research/tests -v
```

The required build timestamp is intentionally different between runs. The
comparison requires every deterministic artifact hash, the normalized protocol
manifest, and the logical final-key hash to match.

## Seal boundaries

- `development/` contains only the eight development cities before `HF`.
- `final_adaptation/` contains final-target labels only before `HF`.
- `final_features/` contains label-free final-evaluation inputs and prediction
  keys. Dynamic values use strictly earlier causal lags.
- `final_labels/SEALED_final_evaluation_labels.parquet` is evaluation-only and
  must not be imported by training or model-selection code.

Model loaders must select predictors exclusively from
`manifests/feature_manifest.json`; audit and coverage columns are not learned
inputs.
