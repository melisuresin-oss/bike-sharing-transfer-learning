# bike-sharing-transfer-learning

Cross-city transfer learning (GRU) for data-scarce bike-sharing demand forecasting — pretrained on Bilbao, Vienna & Glasgow, fine-tuned on Freiburg. TUM Deep Learning & Decision Making course project.

**Authors:** Melis Üresin, Nil Yılmazcan, Sefa Kosova — `{melis.ueresin, nil.yilmazcan, sefa.kosova}@tum.de`

---

## What this project does

Bike-sharing operators need short-term demand forecasts to rebalance bikes across stations. New systems don't have enough local history to train reliable models. This project asks: **does pretraining on data from other European cities help forecast demand in Freiburg when only limited Freiburg data is available?**

We compare 5 methods across 4 target-history budgets (1 day, 7 days, 30 days, full training split):

| Method | Description |
|--------|-------------|
| Historical average | Per-station, per-hour-of-week mean from training data |
| Persistence | Predict the same value as the last observed hour |
| Target-only GRU | GRU trained from scratch on Freiburg data only |
| Pooled GRU | GRU trained on Bilbao + Vienna + Glasgow + Freiburg budget |
| Pretrained + fine-tuned GRU | Pretrain on source cities, fine-tune on Freiburg budget |

The main result is a **transfer-gain curve**: how much does pretraining help as more Freiburg data becomes available?

---

## Data

We use the [European Bike-Sharing Dataset](https://github.com/TUMFTM/european-bike-sharing-dataset) (Waldner et al., 2025), CC BY-NC 4.0.

| Role | City | Stations | Departures |
|------|------|----------|------------|
| Source | Bilbao | 44 | 1,438,542 |
| Source | Vienna | 250 | 356,447 |
| Source | Glasgow | 105 | 348,612 |
| Target | Freiburg | 100 | 496,145 |

Observation window: 29 Aug 2022 – 15 Jul 2023.

---

## Setup

```bash
git clone https://github.com/melisuresin-oss/bike-sharing-transfer-learning.git
cd bike-sharing-transfer-learning
git checkout feature/data-pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the pipeline

### Step 1 — Filter the raw dataset
Download the full dataset from the [European Bike-Sharing Dataset repo](https://github.com/TUMFTM/european-bike-sharing-dataset) and place the CSVs in `data/raw/`. Then run:

```bash
python src/filter_cities.py --data-dir data/raw --out-dir data/interim_csv
```

### Step 2 — Build the hourly panel
```bash
python src/data/build_panel.py
```
Outputs per-city panel parquets to `data/processed/`. Expected total eligible station-hours: ~2,731,800.

### Step 3 — Build supervised samples
```bash
python src/features/windowing.py
```
Outputs windowed training/validation/test parquets and scalers to `data/processed/samples/`.

### Step 4 — Run the experiment
```bash
python src/run_experiment.py
```
Runs all 5 methods across all 4 budgets. Saves results to `results/all_metrics.csv` and model checkpoints to `results/checkpoints/`.

### Step 5 — Plot the transfer-gain curve
```bash
python src/plot_results.py
```
Saves `results/transfer_gain_curve.png`.

### Running on Google Colab
Use `--config configs/colab.yaml` with any of the above scripts. The Colab config points data paths at Google Drive (`MyDrive/bikeshare-transfer/`).

---

## Running tests

```bash
pytest
```

---

## Project structure

```
configs/          Config files (default.yaml, colab.yaml)
data/
  raw/            Raw dataset CSVs (not committed)
  interim/        Filtered 4-city parquets (not committed)
  processed/      Model-ready windowed samples (not committed)
src/
  data/           build_panel.py — hourly panel builder
  features/       windowing.py — supervised sample builder
  models/         gru.py, baselines.py
  training/       train.py — all 3 GRU training regimes
  eval/           metrics.py
  run_experiment.py   Run all 5 methods
  plot_results.py     Transfer-gain curve
tests/            Pytest test suite
results/          Model outputs (not committed)
proposal.pdf      Project proposal
```

---

## Data pipeline notes

`src/data/build_panel.py` builds the hourly per-station departure panel and coverage mask described in the proposal. Validated against real data: total eligible station-hours across the four cities come to 2,731,800, within 0.35% of the proposal's reported 2,741,276.

That validation required disabling the maintenance filter for Bilbao only (`trust_maintenance_flag: false` in `configs/default.yaml`). Bilbao's raw `maintenance` field is not trustworthy: 63.5% of its station_status rows read `maintenance=True`, versus 1.5–15.8% for the other cities. With the filter disabled, Bilbao's eligible station-hours rose from 28.50% to 92.53%.
