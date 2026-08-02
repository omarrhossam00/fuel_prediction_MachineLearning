# Fuel Consumption Prediction vs. Onboard OBD Estimate

Predicting real (sensor-measured) vehicle fuel consumption from OBD-II telemetry, and testing whether a machine learning model beats the vehicle's own onboard fuel estimate.

## Problem

Vehicles estimate fuel consumption in real time using a simple formula based on Mass Air Flow (MAF) and a fixed air-fuel ratio assumption. This project asks: can a model that sees the full sensor picture (RPM, load, throttle, temperature, gear) produce a more accurate estimate than that single-formula baseline, when checked against an actual fuel-flow sensor?

## Data

~789,000 rows of OBD-II telemetry logged from a real vehicle across 22 driving segments, spanning April 2025–June 2026. After dropping rows without ground-truth fuel measurements, ~557,000 rows across 16 segments remained for training/evaluation.

| Role | Column | Notes |
|---|---|---|
| Target | `REAL_FUEL_USAGE_ML_MIN` | Measured fuel-flow sensor reading — ground truth |
| Baseline | `FUEL_USAGE_ML_MIN` | Onboard MAF-based estimate — the number the model needs to beat |
| Features | `RPM`, `SPEED`, `THROTTLE_POS`, `MAF`, `ACCELERATOR_POS_D`, `ENGINE_LOAD`, `COOLANT_TEMP`, `INTAKE_TEMP`, `GEAR`, `ABSOLUTE_LOAD`, `INTAKE_PRESSURE` + engineered rolling averages/deltas | |
| Excluded | `PREDICTED_FUEL_USAGE`, `FUEL_USAGE_DIFF` | Unknown provenance in the source dataset — not defensible as a feature or baseline |

## Approach

1. **Group-aware train/test split** — split by `segment_id` (whole drives), not by row. A random row-level split would leak near-duplicate 1-second-apart samples across train/test and inflate the score.
2. **Feature engineering** — rolling means and deltas per segment (fuel behavior depends on recent driving, not just the instantaneous reading), plus an accelerator-vs-throttle gap feature.
3. **Two models** — Linear Regression as a simple baseline, Random Forest (depth/leaf-constrained on purpose, to keep the train/test gap honest and avoid trivially overfitting on 630k+ rows).
4. **Evaluated on train *and* test** to make overfitting visible, not just report test scores.
5. **MAF ablation** — since the baseline formula is itself MAF-derived, the model was also tested with MAF-based features removed, to check whether it's learning genuinely new signal from the other sensors or mostly re-deriving MAF's role.

## Results

| Method | MAE (ml/min) | RMSE (ml/min) | R² |
|---|---|---|---|
| Onboard MAF formula (baseline) | 5.675 | 6.812 | 0.957 |
| Linear Regression | 2.430 | 4.316 | 0.984 |
| **Random Forest** | **1.772** | **3.543** | **0.989** |

**Random Forest reduces error by 68.8% (MAE) compared to the onboard estimate.**

Train/test R² stayed close (0.995 vs. 0.989), so the model isn't badly overfit despite the row count.

*MAF-ablation result: [fill in from notebook output — MAE without MAF features vs. the 5.675 ml/min baseline]. This determines whether the honest framing is "the model fuses multiple sensors" or "the model corrects the onboard formula's fixed air-fuel-ratio assumption" — worth stating plainly either way.*

## Limitations

- Only 16 of 22 segments had ground-truth fuel readings, and the test set is 4 held-out segments — a real constraint on how confidently these results generalize to unseen driving conditions.
- `TORQUE`, `POWER`, `ABSOLUTE_LOAD`, `INTAKE_PRESSURE`, and related PIDs are missing for entire segments (not scattered dropouts), consistent with a hardware/logging change partway through the collection period — not treated as random missingness.
- Model performance is reported against a single vehicle's data; it isn't validated across different vehicles or engine types.

## What I'd do differently

- Try a Gradient Boosting model (XGBoost/LightGBM) as a stronger comparison point against Random Forest.
- Validate on more held-out segments if more labeled data becomes available.
- Investigate whether the MAF sensor itself has periods of reduced accuracy that both the baseline and the model inherit.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook fuel_consumption_prediction.ipynb
```

Update the CSV filename in the "Load data" cell if your file isn't named `dataset.csv`, then run all cells.

## Repo contents

- `fuel_consumption_prediction.ipynb` — full pipeline: baseline, feature engineering, models, evaluation, ablation, plots
- `fuel_prediction_results.png` — output plots (actual vs. predicted, feature importance, model vs. baseline error)
