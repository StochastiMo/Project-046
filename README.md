# Project-046: Minute-Level HFT Alpha Research on XAG/USD

## Overview
This project studies short-horizon return predictability for XAG/USD using high-frequency market microstructure factors and rolling time-series backtests. The research involves comprehensive factor testing, feature engineering with expanded feature sets, and systematic model comparison across three families of classifiers.

**Key Research Goals:**
- Identify reliable predictive factors for minute-level price movements in precious metals (XAG/USD)
- Develop robust classification models with proper handling of time-series leakage
- Evaluate model performance across multiple prediction horizons (5m, 10m, 20m, 50m)
- Compare statistical significance and practical trading utility of different models

## Workflow
The research follows a structured methodology:
1. **Data preparation**: Cleaning and preprocessing minute-level OHLC and tick data
2. **Factor engineering**: Construction of 22 HFT-inspired microstructure factors (expanded to 162 features)
3. **Factor testing**: IC (Information Coefficient) and RankIC analysis across prediction horizons
4. **Rolling backtesting**: Walk-forward validation with proper embargo handling to prevent leakage
5. **Model comparison**: Systematic evaluation across Logistic, Generative, and Tree-based model families

## Repository Structure
- [Quantitative strategy.ipynb](Quantitative%20strategy.ipynb): Main research notebook containing:
  - Data preprocessing and feature engineering
  - IC testing and factor significance analysis
  - Classification model development and validation
  - Comprehensive model performance comparison
- [Backtesting.ipynb](Backtesting.ipynb): Strategy implementation and performance analysis notebook
- [Improvement ideas.txt](Improvement%20ideas.txt): Documentation of iterative improvements and research directions
- `data/`: Raw datasets organized by asset class
  - `COMMODITY/`: XAG_USD.csv (Silver/USD), XAU_USD.csv (Gold/USD)
  - `CRYPTO/`: BTC (Bitcoin data)
  - `CURRENCY/`: EUR_USD, GBP_USD, AUD_USD, USD_JPY, USD_CAD pairs
  - `EQUITY/`: AAPL, QQQ stock data

## Data and Targets
**Primary Asset**: XAG/USD (Silver futures) minute-level data

**Forward Return Definitions** (calculated with embargo periods):
- `ret_5m`: 5-minute forward return
- `ret_10m`: 10-minute forward return
- `ret_20m`: 20-minute forward return
- `ret_50m`: 50-minute forward return

**Classification Targets**: Three-class labels generated using train-set terciles:
- **Down**: Lower tercile (negative return signals)
- **Noise**: Middle tercile (neutral/indecisive signals)
- **Up**: Upper tercile (positive return signals)

## Key Improvements Implemented

### 1. **Time-Series Leakage Prevention** (Critical)
- **Issue**: Previous label construction allowed training set tail samples to use validation-period future prices
- **Solution**: Added embargo gaps between training and validation sets, aligned with prediction horizons
- **Impact**: Eliminates look-ahead bias and ensures realistic backtesting

### 2. **Enhanced Feature Set** (162 Features from Original 20)
Original 20 base factors:
- **Spread/Pressure**: QS, RS, SCZ, ISA, MPR, IPS, OCG
- **Momentum**: RER, TDR, PTC
- **Volume**: TWMD, TACC_R, PV (Parkinson Volatility)
- **Volatility**: GKV (Garman-Klass), VRR (Volatility Regime Ratio)
- **Microstructure**: BVA, ES, RIS, KL, AIR

**Feature Expansion** (adds ~142 features):
- **Lag terms** (lags: 1, 5, 20): Captures factor persistence and time-series relationships
- **Change rates** (returns): Captures factor momentum and acceleration
- **Intraday time features** (cyclical encoding):
  - Hour and minute sine/cosine transforms (captures circadian trading patterns)
  - Session flags: Asia, Europe, US trading sessions
- **Regime features**:
  - 20-period rolling volatility quartiles
  - Regime indicators: low volatility, mid volatility, high volatility states
  - Critical for model adaptation to different market conditions

### 3. **Improved Rolling Window Configuration**
| Parameter | Previous | Improved | Rationale |
|-----------|----------|----------|-----------|
| Training window | 500 samples | 1800 samples | Increases stable period for model fitting |
| Validation window | 100 samples | 200 samples | Reduces validation metric variance |
| Step size | 500 samples | 1800 samples | Allows model retraining every 1800 periods |
| Embargo | Per horizon | Per horizon (5/10/20/50) | Prevents look-ahead bias |

**Impact**: Larger training windows provide more stable factor estimates; larger step sizes maintain independence between validation folds.

### 4. **Model Evaluation Focus**
- **Expanded metrics**: Accuracy, Precision, Recall, F1, **Log Loss**
- **Emphasis on Log Loss**: Better measures probability calibration than accuracy for imbalanced multi-class problems
- **Note**: Practical trading viability requires consideration of:
  - Probability calibration (Platt/Isotonic scaling recommended for future work)
  - Transaction costs and slippage
  - Strategy Sharpe ratio and maximum drawdown (implemented in Backtesting.ipynb)

## Feature Set Details

### Base Factors (20)
The factors cover critical market microstructure dimensions:

**Spread & Order Book (Depth):**
- `QS` (Quoted Spread): Bid-ask spread normalized
- `RS` (Realized Spread): Actual execution spread indicator
- `SCZ` (Spread-Cost Proxy): Normalized cost measure

**Volume & Activity:**
- `TDR` (Tick Density Ratio): Current tick activity vs. rolling average
- `PTC` (Price-Tick Correlation): Rolling correlation of price changes and tick counts
- `TWMD` (Tick-Weighted Mid Deviation): Deviation from activity-weighted average price
- `TACC_R` (Tick Acceleration): First difference of tick activity

**Volatility Estimation:**
- `PV` (Parkinson Volatility): ~5× more efficient than close-close estimator
- `GKV` (Garman-Klass Volatility): Unbiased estimator using full OHLC
- `VRR` (Volatility Regime Ratio): Short-to-long volatility ratio for regime detection

**Additional Factors:**
- `ISA`, `MPR`, `IPS`: Order-book imbalance and pressure indicators
- `OCG`: Order cluster gravity
- `RER`: Realized excess returns
- `BVA`, `ES`, `RIS`, `KL`, `AIR`: Additional microstructure signals

## Factor Testing Results

### Information Coefficient (IC) Analysis
IC and RankIC are computed over rolling 10-hour (600-minute) windows for each factor against forward returns.

**Key Metrics**:
- **IC (Pearson)**: Linear correlation between factor and forward return
- **RankIC (Spearman)**: Rank correlation, robust to outliers
- **IC-IR (Information Ratio)**: IC mean divided by IC standard deviation (signal stability)
- **Positive IC %**: Percentage of periods with positive IC (directional consistency)

**Methodology**:
- Rolling period: 600 minutes (~10 trading hours)
- Computed against each horizon (ret_5m, ret_10m, ret_20m, ret_50m)

### Top Base Factor IC Results

| Factor | IC (5m) | IC_IR (5m) | IC (50m) | IC_IR (50m) | Direction |
|--------|---------|-----------|---------|-----------|-----------|
| **MR** (Mid-Return) | -0.0486 | -0.443 | -0.0963 | -0.631 | Strong mean-reversion |
| **MS** (Momentum Score) | -0.0418 | -0.389 | -0.0919 | -0.599 | Strong mean-reversion |
| **MPR** (Mid-Price Return) | -0.0403 | -0.371 | -0.0807 | -0.621 | Strong mean-reversion |
| **TWMD** (Tick-Weighted Mid Dev) | -0.0350 | -0.336 | -0.0674 | -0.617 | Mean-reversion |
| **RS** (Realized Spread) | +0.0101 | +0.138 | +0.0139 | +0.095 | Positive (spread signal) |
| **QS** (Quoted Spread) | +0.0070 | +0.096 | +0.0052 | +0.035 | Positive |
| **RER** (Realized Excess Return) | +0.0044 | +0.081 | +0.0033 | +0.099 | Positive |

**Key Findings**:
- Mean-reversion factors (MR, MS, MPR) dominate with the strongest IC_IR, signal strengthens with longer horizons (50m IC_IR > -0.60)
- Spread factors (RS, QS) show weaker but consistently positive predictability
- Most factors show IC_IR < |0.2| for short horizons, indicating weak but exploitable alpha

## Modeling and Backtest Design

### Rolling Backtest Architecture
The rolling window approach prevents look-ahead bias while maintaining temporal structure:

```
Time →
|----Training (1800)----|-Embargo-|--Validation (200)--|----Training (1800)----|-Embargo-|--Validation (200)--|
                        ↑embargo size = prediction horizon
```

### Preprocessing Pipeline (per rolling split)
Applied identically to training and validation:
1. **Handling infinity**: Replace inf/nan values
2. **Clipping**: Use training set 1%/99% quantiles to clip extreme values
3. **Imputation**: Fill remaining NaN with training set median
4. **Standardization**: Transform via training set mean/std using `StandardScaler`
5. **Dimensionality reduction** (optional):
   - PCA with 80% variance retention (`pca_keep=0.8`)
   - Applied only to Logistic/Generative models (not trees)

### Performance Metrics
- **Accuracy**: Mean proportion of correct predictions (baseline ~33% for random 3-class)
- **Precision/Recall**: Per-class evaluations (macro-averaged across classes)
- **F1-Score**: Harmonic mean of precision and recall
- **Log Loss**: $-\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{3}y_{i,c}\log(p_{i,c})$ (penalizes confidence in wrong predictions)

## Model Families Tested

### 1. Logistic Regression Family
Interpretable linear models with different regularization strategies:
- **LR_plain**: Unregularized baseline
- **LR_lasso**: L1 regularization for feature selection
- **LR_ridge**: L2 regularization for stability
- **LR_elasticnet**: Combined L1/L2 with 0.5 ratio

**Best use case**: Interpretability, risk-limited deployments
**Note**: PCA improves performance on high-dimensional feature space

### 2. Generative Model Family
Probabilistic classifiers modeling joint distribution P(X,Y):
- **Gaussian Naive Bayes**: Assumes feature independence (fast, robust)
- **LDA** (Linear Discriminant Analysis): Assumes shared covariance across classes
- **QDA** (Quadratic Discriminant Analysis): Allows class-specific covariance (with regularization)
- **KNN** (k-Nearest Neighbors, k=5): Non-parametric, local structure-focused

**Best use case**: Probability calibration, interpretable decision boundaries
**Note**: These models typically outperform logistic models on this task

### 3. Tree-Based Family
Ensemble and boosting methods capturing non-linear relationships:
- **Bagging**: Bootstrap aggregating with decision trees (base: max_depth=4)
- **Random Forest**: 100 trees with balanced class weights
- **Gradient Boosting**: Sequential error correction (learning_rate=0.05)
- **XGBoost**: Extreme gradient boosting with regularization

**Best use case**: Non-linear patterns, feature interaction capture, highest accuracy
**Note**: No PCA applied; trees handle high dimensions natively

## Results Summary

> Baseline reference: random 3-class guess → Accuracy = 33.3%, Log Loss = log(3) = 1.099

### Model Accuracy by Horizon

| Model | 5m Acc | 5m LogLoss | 10m Acc | 10m LogLoss | 20m Acc | 20m LogLoss | 50m Acc | 50m LogLoss |
|-------|--------|-----------|---------|------------|---------|------------|---------|------------|
| **Random Forest** | **0.4118** | **1.0753** | **0.4089** | **1.0787** | **0.3858** | **1.0872** | 0.3678 | **1.1025** |
| Bagging | 0.4043 | 1.0828 | 0.4011 | 1.0906 | 0.3816 | 1.1233 | 0.3612 | 1.2244 |
| XGBoost | 0.3937 | 1.1091 | 0.3908 | 1.1275 | 0.3752 | 1.1729 | **0.3704** | 1.2752 |
| Gradient Boosting | 0.3891 | 1.1240 | 0.3914 | 1.1426 | 0.3730 | 1.1990 | 0.3666 | 1.3039 |
| Naive Bayes | **0.3987** | 5.3446 | 0.3954 | 5.6139 | 0.3886 | 5.9670 | 0.3710 | 6.8420 |
| LDA (with PCA) | 0.3908 | 1.1192 | 0.3843 | 1.1431 | 0.3733 | 1.1764 | 0.3683 | 1.2221 |
| LR_lasso (with PCA) | 0.3907 | 1.1160 | 0.3853 | 1.1403 | 0.3729 | 1.1722 | 0.3678 | 1.2212 |

**Overall out-of-sample accuracy (Random Forest, 5m, rolling validation): 40.51%** vs. 33.3% random baseline.

### Key Findings

1. **Model Family**: Tree-based > Generative ≈ Logistic on accuracy; Random Forest has the best log loss across all horizons. Naive Bayes achieves high accuracy but is poorly calibrated (log loss >> 1.1).

2. **Horizon**: Accuracy decreases monotonically with longer horizons (5m: ~41% → 50m: ~37%). Mean-reversion signals strengthen at 50m (MR IC_IR = -0.631) but overall predictability still degrades.

3. **PCA**: Helps logistic/generative models; not applied to trees (handle high dimensions natively).

### Strategy Performance (Random Forest on 5m, XAG/USD)

Two signal construction approaches were evaluated using `vectorbt`:

| Metric | Probability-based | Prediction-based |
|--------|------------------|-----------------|
| **Period** | 2025-01-23 ~ 2026-01-30 (248 days) | 2025-01-23 ~ 2026-01-30 (248 days) |
| **Total Return** | **43.46%** | **149.15%** |
| **Benchmark Return (Buy & Hold)** | 220.32% | 220.32% |
| **Max Drawdown** | 13.22% | 19.01% |
| **Max Drawdown Duration** | 53 days | 35 days |
| **Total Trades** | 1,251 | 17,540 |
| **Win Rate** | 57.60% | 54.21% |
| **Sharpe Ratio** | 2.12 | **3.84** |
| **Calmar Ratio** | 5.29 | **14.85** |
| **Profit Factor** | 1.20 | 1.10 |
| **Avg Winning Trade** | +0.28% | +0.11% |
| **Avg Losing Trade** | -0.31% | -0.12% |
| **Fees** | 0 | 0 |

- **Probability-based**: Enter long when `prob_positive > 0.5` or `prob_negative > 0.5`; fewer, larger trades; better per-trade quality.
- **Prediction-based**: Enter when `pred == 2 (up)` or previous period `pred == 0 (down)`; high-frequency mean-reversion style; very high Sharpe but fee-sensitive.
- Neither strategy beats buy-and-hold over the period, primarily due to XAG's exceptional 220% bull run in 2025.

### Baseline Comparison
| Benchmark | Value |
|-----------|-------|
| Random 3-class accuracy | 33.33% |
| Log Loss (uniform random) | 1.099 |
| Buy & Hold Return | 220.32% |

## Improvements Over Initial Methodology

| Aspect | Initial | Improved | Benefit |
|--------|---------|----------|---------|
| **Leakage Handling** | Implicit | Explicit embargo gaps | Realistic performance estimates |
| **Feature Count** | 20 | 162 | Captures temporal patterns & regimes |
| **Training Window** | 500 | 1800 | Stable parameter estimation |
| **Log Loss Focus** | Secondary | Primary | Probability calibration emphasis |
| **Feature Standardization** | Applied | Per-split, train-fit | Prevents validation leakage |
| **Model Coverage** | Limited | 12 models × 2 PCA × 4 horizons | Comprehensive evaluation |

## Future Work

1. **Probability Calibration**: Apply Platt scaling or Isotonic regression to improve probability estimates
2. **Threshold Optimization**: Tune decision thresholds to maximize Sharpe ratio (not just accuracy)
3. **Cost-Sensitive Learning**: Incorporate asymmetric trading costs (long vs short commissions)
4. **Cross-Asset Testing**: Validate factor performance across CRYPTO, CURRENCY, EQUITY assets
5. **Real-Time Implementation**: Deploy models with proper Monte Carlo backtesting and walk-forward validation
6. **Explainability**: SHAP values and local interpretability for model decisions
7. **Ensemble Methods**: Combine best models from each family via meta-learning

## Technical Details

### Environment Requirements
- Python 3.8+
- pandas, numpy for data manipulation
- scikit-learn for classical ML models
- xgboost for gradient boosting
- matplotlib, seaborn for visualization
- scipy (spearmanr for RankIC computation)

### Code Organization
- Feature engineering centralized in `add_factor_transforms()`, `add_time_features()`, `add_regime_features()`
- Rolling backtest implemented via `rolling_backtest()` with configurable window sizes and metrics
- Model definitions separate by family for clarity and maintainability
- Results printed by horizon and model family for easy comparison

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
- `training_size = 1800`
- `validation_size = 200`
- `step_size = 1800`
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
