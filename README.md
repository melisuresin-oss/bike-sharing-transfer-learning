# When Does Cross-City Graph Transfer Help?

**Few-Shot Bike-Sharing Demand Forecasting Across Heterogeneous European Systems**

Technical University of Munich — Deep Learning and Decision Making

Authors:

- Sefa Kosova
- Melis Üresin
- Nil Yılmazcan

## Overview

This repository contains the final V2.2 scientific implementation and compact,
frozen result summaries for a 12-city European bike-sharing benchmark. The task
is **next-hour, station-level departure forecasting**: at each eligible forecast
hour, a method predicts departure counts for the stations in a target system.

This is not a history-free cold-start benchmark. Strictly causal target demand
histories may be used as forecasting inputs. **Parameter zero-shot** means that
no target labels are used for fitting, adaptation, or model selection.

## Research Question

When do graph structure and cross-city transfer improve demand forecasts under
limited target-city supervision, and how consistently do those effects hold
across heterogeneous bike-sharing systems and supervision budgets?

## Dataset and City Split

The benchmark uses 12 systems from the [European Bike-Sharing Dataset](https://github.com/TUMFTM/european-bike-sharing-dataset).
The canonical source is maintained by TUM's Chair of Automotive Technology on
GitHub; a [Hugging Face mirror](https://huggingface.co/datasets/PellelNitram/european-bike-sharing-dataset)
is also available.

| Role | Cities |
|---|---|
| Development/source | Dortmund, Heidelberg, Marburg, Gießen, Cardiff, Bilbao, Freiburg, Göteborg |
| Held-out final targets | Mannheim, Innsbruck, Glasgow, Split |

Target-supervision budgets are **0d, 1d, 7d, 30d, and Full**. The held-out final
targets were not used for development selection.

## Methods

The final comparison covers historical-average, persistence, and seasonal-naive
baselines; target-only Vanilla-GRU and Graph-GRU; pooled Graph-GRU; ordinary
source-pretrained Graph-GRU; and a gradient-reversal-layer (GRL) source-pretrained
Graph-GRU. The selected graph uses four geographic neighbors, and learned models
use a hidden width of 32.

## Experimental Protocol

The protocol fixes the city split, causal preprocessing, temporal windows,
architecture choices, optimization budgets, evaluation cohorts, and paired
temporal bootstrap before final scoring. Learned-model results aggregate five
seeds: 17, 29, 43, 71, and 101. The primary metric is count-space MAE, with
RMSE, WAPE, and station-macro MAE as secondary metrics.

The authoritative specification is
[`FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md`](FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md),
with the machine-readable resolution in
[`final_evaluation_protocol_v2_2.json`](final_evaluation_protocol_v2_2.json).

## Main Results

Standalone equal-city macro MAE (lower is better):

| Method | 0d | 1d | 7d | 30d | Full |
|---|---:|---:|---:|---:|---:|
| Source historical average | 1.072289 | — | — | — | — |
| Target historical average | — | 0.976402 | 0.941287 | **0.835273** | **0.807435** |
| Seasonal naive | 0.878195 | 0.878195 | 0.878195 | 0.878195 | 0.878195 |
| Persistence | unavailable | unavailable | unavailable | unavailable | unavailable |
| Target-only Vanilla-GRU | — | 1.111835 | 0.996551 | 0.970196 | 0.956180 |
| Target-only Graph-GRU | — | 1.084970 | 0.996926 | 0.992346 | 0.991063 |
| Pooled Graph-GRU | — | 0.888162 | **0.869285** | 0.879580 | 0.906724 |
| Ordinary source-pretrained Graph-GRU | **0.874405** | **0.889892** | 0.873627 | 0.872275 | 0.871797 |
| GRL source-pretrained Graph-GRU | 0.940250 | 0.961752 | 0.943779 | 0.936276 | 0.936954 |

The controlled ordinary-transfer gains over target-only Graph-GRU are
**+0.195077, +0.123300, +0.120071, and +0.119266** at 1d, 7d, 30d, and Full,
respectively. A positive controlled gain favors the evaluated method.

The main interpretation is:

- Ordinary source transfer beats target-only Graph-GRU at every nonzero target-supervision budget.
- Graph propagation helps at 1d but not at larger budgets.
- Pooled Graph-GRU is strong under scarce supervision and is the best requested learned/table method at 7d.
- Target historical averaging is the strongest registered standalone method at 30d and Full.
- The evaluated GRL specification underperforms ordinary pretraining.
- Effects differ across cities and budgets; the results do not establish universal significance across European systems.

See [`docs/RESULTS.md`](docs/RESULTS.md) for controlled comparisons, city-level
examples, and the distinction between standalone and matched paired cohorts.

## Repository Structure

- `research/` — scientific code, development-stage implementations, final execution revisions, tests, and compact protocol-bound manifests.
- `results/final_v2_2/` — the three byte-preserved frozen result summaries and their public index.
- `docs/` — results interpretation, reproducibility guidance, and protocol provenance.
- `processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json` — frozen final dataset identity and cohort manifest; no raw data are included.
- Root protocol files — the final evaluation and GRL-cardinality resolutions plus their inherited specifications.

Large raw data, snapshots, label arrays, predictions, bootstrap replicates,
checkpoints, execution archives, and machine-specific workspaces are not
published in this repository.

## Reproducibility

[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) maps the scientific workflow
to the actual source modules and records the final settings. The public tree can
reproduce and inspect the **scientific workflow**, but not the exact historical
governed execution: immutable raw snapshots, fitted checkpoints, committed
predictions, private label-access state, and large execution bundles are
intentionally absent.

Install the core dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Dataset

Dataset resources:

- [Canonical TUMFTM GitHub repository](https://github.com/TUMFTM/european-bike-sharing-dataset)
- [Hugging Face mirror](https://huggingface.co/datasets/PellelNitram/european-bike-sharing-dataset)

Dataset files are not redistributed here. Their final protocol identity is recorded
in [`processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json`](processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json).

## Report

[Final submitted paper](paper/DLDM_Final_Project_Kosova_Uresin_Yilmazcan.pdf):
**“When Does Cross-City Graph Transfer Help? Few-Shot
Bike-Sharing Demand Forecasting Across Heterogeneous European Systems”**, by
Sefa Kosova, Melis Üresin, and Nil Yılmazcan, submitted for TUM Deep Learning
and Decision Making.

[Project presentation](https://drive.google.com/drive/folders/13au-S79TYy3v8xBDL3Z2-2A652nBlmRS?usp=drive_link)
