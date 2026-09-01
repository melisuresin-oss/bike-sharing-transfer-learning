# bike-sharing-transfer-learning
Cross-city transfer learning (GRU) for data-scarce bike-sharing demand forecasting — pretrained on Bilbao, Vienna &amp; Glasgow, fine-tuned on Freiburg. TUM Deep Learning &amp; Decision Making course project.

## Data pipeline notes

`src/data/build_panel.py` builds the hourly per-station departure panel and
coverage mask described in the proposal. Validated against real data: total
eligible station-hours across the four cities come to 2,731,800, within
0.35% of the proposal's reported 2,741,276.

That validation required disabling the maintenance filter for Bilbao only
(`coverage_mask.trust_maintenance_flag: false` in `configs/default.yaml` and
`configs/colab.yaml`). Bilbao's raw `maintenance` field is not trustworthy:
63.5% of its station_status rows read `maintenance=True`, versus 1.5% for
Vienna and comparably low rates for Glasgow and Freiburg. With the filter
applied, Bilbao's eligible station-hours sat at 28.50%; with it disabled,
92.53% — the other three cities are unaffected and keep the filter on.
