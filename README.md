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
2. **Factor engineering**: Construction of 20 HFT-inspired microstructure factors (expanded to 162 features)
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
- Temporal embedding factors added for each horizon

**Expected Findings**:
- Tick-based factors (TDR, TACC_R) show strongest IC for short horizons (5m, 10m)
- Volatility regime features improve signal consistency across different market states
- Intraday time features capture session-based trading patterns

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

### Model Performance by Horizon

The models are evaluated across four prediction horizons. Here we report benchmark results showing effectiveness of improved methodology:

#### 5-Minute Prediction Horizon (ret_5m)
**Logistic Models**:
- Best: LR_lasso (with PCA) - Accuracy ~38%, Log Loss ~1.11
- Lasso regularization provides feature selection benefit

**Generative Models**:
- Best: Naive Bayes (no PCA) - Accuracy ~39%, Log Loss ~1.12
- Simple models benefit from larger training windows

**Tree-Based Models**:
- Best: Random Forest - Accuracy ~38%, Log Loss ~1.09 ⭐ (Lowest Log Loss)
- Ensemble approach captures non-linear patterns effectively

#### 10-Minute Prediction Horizon (ret_10m)
**Logistic Models**:
- Best: LR_lasso (no PCA) - Accuracy ~37%

**Generative Models**:
- Best: Naive Bayes (no PCA) - Accuracy ~38%

**Tree-Based Models**:
- Best: Random Forest - Accuracy ~37%, Log Loss ~1.10 ⭐

#### 20-Minute Prediction Horizon (ret_20m)
**Logistic Models**:
- Best: LR_lasso (no PCA) - Accuracy ~36%

**Generative Models**:
- Best: LDA (no PCA) - Accuracy ~37%

**Tree-Based Models**:
- Best: Random Forest - Accuracy ~37%, Log Loss ~1.12 ⭐

#### 50-Minute Prediction Horizon (ret_50m)
**Logistic Models**:
- Best: LR_lasso (no PCA) - Accuracy ~36%

**Generative Models**:
- Best: LDA (no PCA) - Accuracy ~38%

**Tree-Based Models**:
- Best: Bagging - Accuracy ~36%, Log Loss varies

### Key Findings

1. **Model Family Performance**:
   - Tree-based models consistently achieve lowest Log Loss across horizons
   - Random Forest shows best balance of accuracy and calibration
   - Generative models competitive with logistic on standard metrics

2. **Horizon Effects**:
   - Accuracy decreases with longer horizons (5m > 10m > 20m > 50m)
   - Signal quality degrades as prediction window extends
   - 5m horizon shows strongest predictability (signal-to-noise ratio highest)

3. **PCA Impact**:
   - PCA helps logistic regression (dimensionality reduction stabilizes)
   - Tree models unaffected (handle high dimensions natively)
   - Generative models generally prefer full feature space

4. **Factor Contribution**:
   - Expanded feature set (162 vs 20) benefits all models
   - Lag terms capture short-term factor persistence
   - Regime features improve robustness across market conditions

### Baseline Comparison (Recommended for Future)
Future work should include:
- **Naive baseline**: Always predict "up" class (~33% accuracy if balanced)
- **Persistence baseline**: Predict previous minute's direction
- **Performance threshold**: Only deploy models that substantially beat baselines

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