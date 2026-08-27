# Freight Rate Prediction

This project predicts `posted_rate` for 12,000 November-December 2025 freight loads and a fixed Lexington-to-Fort Wayne lane for every day in December. It uses a leakage-aware temporal holdout, a transparent equipment-rate baseline, and a LightGBM model trained on a log-transformed target.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the supplied CSV files in `data/`. Keep the provided `score.py` unchanged.

## Run

Run these commands from the repository root, in this order:

```bash
python src/train.py
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

To execute the EDA and regenerate its charts and PDF report:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

Training prints baseline and LightGBM MAE, MAPE, and RMSE, writes `report/metrics.txt` and `report/feature_importance.png`, and saves the final artifact at `models/model.pkl`. Prediction fills `validation_predictions.csv` by an exact `load_id` merge and updates only `predicted_rate` in the seven-column December file.

## Approach

- **Temporal validation:** January-August is used for fitting and September-October is held out, matching the two-month forecasting horizon. The code asserts that the training dates end before the holdout begins.
- **Data quality:** weights are made absolute and missing weights use training equipment medians. Missing market index values use same-date and same-month medians, then a training-derived global fallback. No validation rows are dropped.
- **Features:** calendar fields, native categorical lane/equipment fields, Haversine distance, weight per mile, market index, and quote signal are included. Missing December coordinates are looked up from the training data; its market index uses the observed January-February winter median and quote signal uses comparable Dry Van loads around 360 miles. A tightly capped weekday factor from those training loads provides realistic daily variation beyond the labeled date range.
- **Model:** the equipment median rate-per-mile baseline is compared with LightGBM. LightGBM learns `log1p(posted_rate)`, reducing sensitivity to extreme rate-per-mile observations; outputs are transformed with `expm1` and floored at $50.

## Repository structure

```text
data/                                   supplied inputs and filled December output
notebooks/01_eda.ipynb                  reproducible EDA and report charts
src/clean.py                            cleaning and imputation
src/features.py                         feature engineering
src/train.py                            temporal evaluation and final training
src/predict.py                          validation and December inference
models/model.pkl                        generated model artifact
report/                                 generated metrics, charts, and report.pdf
validation_predictions.csv              generated 12,000-row submission
score.py                                provided validation script (unchanged)
requirements.txt                        Python dependencies
```
