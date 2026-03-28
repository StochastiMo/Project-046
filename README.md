# Project-046: Minute-Level HFT Alpha Research on XAG/USD

## Overview
This project studies short-horizon return predictability for XAG/USD using high-frequency market microstructure factors and rolling time-series backtests.

Main workflow implemented in the notebooks:
1. Data cleaning and preprocessing of minute-level quote/tick data.
2. Feature engineering of 20 HFT-inspired factors.
3. Factor testing with Pearson IC and RankIC.
4. Three-class classification backtests across multiple horizons.
5. Model comparison across Logistic, Generative, and Tree-based families.

## Repository Structure
- `Quantitative strategy.ipynb`: main research notebook (factor engineering, IC tests, classification, model comparisons).
- `Backtesting.ipynb`: strategy/backtesting notebook.
- `data/`: raw datasets (COMMODITY, CRYPTO, CURRENCY, EQUITY).

## Data and Targets
The main notebook currently focuses on `data/COMMODITY/XAG_USD.csv`.

Forward returns are defined as:
- `ret_5m`
- `ret_10m`
- `ret_20m`
- `ret_50m`

Classification target is generated per rolling window using train-set terciles of forward return:
- lower tercile: down
- middle tercile: noise
- upper tercile: up

## Feature Set (20 Factors)
The project engineers 20 microstructure and HFT factors:

`QS, RS, SCZ, ISA, MPR, IPS, OCG, RER, TDR, PTC, TWMD, TACC_R, PV, GKV, VRR, BVA, ES, RIS, KL, AIR`

These factors cover:
- spread and order-book pressure
- momentum and mean reversion
- activity and tick-flow information
- volatility and regime effects
- microstructure friction and price impact

## Modeling and Backtest Design
### Rolling backtest setup
- Training window: `500`
- Validation window: `100`
- Step size: `500`
- Embargo: aligned with horizon (`5/10/20/50`) to reduce leakage

### Preprocessing pipeline in each rolling split
1. Replace inf with nan.
2. Clip features using train 1%/99% quantiles.
3. Fill missing values with train median.
4. Standardize with train-fit `StandardScaler`.
5. Optional PCA (`pca_keep=0.8`).

### Metrics
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Multiclass Log Loss

## Implemented Models
### Logistic Regression family
- Plain
- Lasso
- Ridge
- Elastic Net

### Generative family
- Gaussian Naive Bayes
- LDA
- QDA
- KNN

### Tree-based family
- Bagging
- Random Forest
- Gradient Boosting
- XGBoost

## Existing Results Summary (from notebook outputs)
Note: The following numbers are aggregated means from current notebook output cells.

### Best Accuracy by model family and horizon
| Horizon | Logistic | Generative | Tree-based |
|---|---:|---:|---:|
| 5m  | 0.3779 (LR_lasso + PCA) | 0.3870 (Naive Bayes, no PCA) | 0.3790 (Random Forest) |
| 10m | 0.3711 (LR_lasso, no PCA) | 0.3781 (Naive Bayes, no PCA) | 0.3732 (Random Forest) |
| 20m | 0.3615 (LR_lasso, no PCA) | 0.3749 (LDA, no PCA) | 0.3665 (Random Forest) |
| 50m | 0.3563 (LR_lasso, no PCA) | 0.3817 (LDA, no PCA) | 0.3572 (Bagging) |

### Best Log Loss by model family and horizon
| Horizon | Logistic | Generative | Tree-based |
|---|---:|---:|---:|
| 5m  | 1.1143 (LR_lasso + PCA) | 1.1167 (LDA + PCA) | 1.0936 (Random Forest) |
| 10m | 1.1355 (LR_lasso + PCA) | 1.1372 (LDA + PCA) | 1.1034 (Random Forest) |
| 20m | 1.1683 (LR_lasso + PCA) | 1.1685 (LDA + PCA) | 1.1209 (Random Forest) |
| 50m | 1.2239 (LR_lasso + PCA) | 1.2198 (LDA + PCA) | 1.1485 (Random Forest) |

### Baseline reference
For uniform random guess in 3-class classification:
- Accuracy baseline: `1/3 = 0.3333`
- Log loss baseline: `log(3) = 1.0986`

Interpretation:
- Accuracy values are above random baseline across most settings.
- Probability calibration (log loss) is more mixed; tree models, especially Random Forest, are currently strongest in this notebook snapshot.

## How to Run
1. Open `Quantitative strategy.ipynb`.
2. Run all cells from top to bottom.
3. Re-check model output blocks under section `1.4.1`.
4. Optionally run `Backtesting.ipynb` for downstream strategy evaluation.

## Dependencies
Core packages used in the notebook:
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- vectorbt
- ta-lib

Example installation:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost vectorbt ta-lib
```

## Known Issue and Fix
If you see this XGBoost error:

`ValueError: Invalid classes inferred from unique values of y. Expected: [0 1 2], got [-1 0 1]`

Cause:
- XGBoost multiclass requires labels encoded as consecutive integers starting from 0.

Fix:
- Use `{0, 1, 2}` internally for training (`0=down, 1=noise, 2=up`) or map `{-1,0,1}` to `{0,1,2}` before `fit`.

## Final Recommended Model Config
To keep one robust default setting for future experiments, use the following configuration.

### Recommended default (balanced and stable)
- Model: `RandomForestClassifier`
- Why: among current outputs, it is consistently strong on log loss across all horizons and remains competitive on accuracy.

Suggested parameters:

```python
RandomForestClassifier(
	n_estimators=100,
	max_depth=4,
	min_samples_leaf=8,
	class_weight="balanced",
	random_state=42,
	n_jobs=-1
)
```

Backtest settings:
- `training_size = 500`
- `validation_size = 100`
- `step_size = 500`
- `embargo = horizon` (5/10/20/50)
- `use_pca = False` for tree models
- Label encoding for XGBoost compatibility: `0=down, 1=noise, 2=up`

### If your priority is top Accuracy
- 5m/10m: `Naive Bayes` (without PCA)
- 20m/50m: `LDA` (without PCA)

### If your priority is best Log Loss
- Use `Random Forest` (without PCA) as first choice.
- For linear benchmark and easier interpretability, use `LR_lasso + PCA`.

### Practical deployment suggestion
- Primary model: `Random Forest`
- Benchmark model: `LR_lasso + PCA`
- Report both metrics for each horizon: Accuracy and Log Loss
- Keep the same rolling windows and embargo across all model families for fair comparison

## Next Steps
- Add transaction-cost-aware PnL evaluation and turnover constraints.
- Add probability calibration (Platt/Isotonic) and compare post-calibration log loss.
- Extend the same pipeline to other assets in `data/` for cross-asset robustness.