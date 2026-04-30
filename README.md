# Project-046: Minute-Level HFT Alpha Research on XAG/USD

## Overview
This project studies short-horizon return predictability for XAG/USD (Silver) using high-frequency market microstructure factors and rolling time-series backtests. A key innovation is the use of a **Claude LLM agent** to systematically survey academic literature — spanning Roll (1984), Kyle (1985), Amihud (2002), Garman-Klass (1980), and Parkinson (1980) — and translate theoretical microstructure signals into 20 computable HFT factors. These are expanded to 149 features and evaluated through rigorous walk-forward validation across four families of classifiers.

**Key Research Goals:**
- Leverage an LLM agent to automate literature-driven HFT factor discovery
- Identify reliable predictive factors for minute-level price movements in XAG/USD
- Develop robust classification models with proper time-series leakage prevention
- Evaluate model performance across multiple prediction horizons (5m, 10m, 20m, 50m)

## Workflow
1. **Agent-assisted factor research**: Claude agent surveys market microstructure literature and proposes factor formulas grounded in theory
2. **Data preparation**: Cleaning and preprocessing minute-level OHLC and tick data
3. **Factor engineering**: 20 HFT microstructure factors → expanded to 149 features
4. **Factor testing**: RankIC (Spearman) analysis across four prediction horizons
5. **Rolling backtesting**: Walk-forward validation with per-horizon embargo gaps
6. **Model comparison**: Four classifier families — Logistic, Generative, Tree-based, GRU

## Repository Structure
- [Quantitative strategy.ipynb](Quantitative%20strategy.ipynb): Main research notebook
- [Improvement ideas.txt](Improvement%20ideas.txt): Iterative improvement log
- `data/`: Raw datasets
  - `COMMODITY/`: XAG_USD.csv, XAU_USD.csv
  - `CRYPTO/`: BTC
  - `CURRENCY/`: EUR_USD, GBP_USD, AUD_USD, USD_JPY, USD_CAD
  - `EQUITY/`: AAPL, QQQ

## Data and Targets
**Primary Asset**: XAG/USD minute-level data

**Forward Return Horizons**: `ret_5m`, `ret_10m`, `ret_20m`, `ret_50m`

**Classification Targets**: Three-class labels via train-set terciles:
- **Down** (0): Lower tercile
- **Noise** (1): Middle tercile
- **Up** (2): Upper tercile

---

## Data Cleaning

**Source file**: `data/COMMODITY/XAG_USD.csv` — 14 raw columns, 379,053 rows

**Step-by-step pipeline:**

| Step | Operation | Result |
|------|-----------|--------|
| 1 | Compute mid-prices: `(bid + ask) / 2` for open, high, low, close | 4 mid-price columns added |
| 2 | Log-transform tick count: `volume = log1p(tick_cnt)` | Normalizes tick activity |
| 3 | Drop rows with missing `status` | −19,253 rows → **359,800 rows** |
| 4 | Drop non-essential columns: `symbol`, `file_time`, `status`, `time_frame` | |
| 5 | Construct forward returns: `ret_Xm = (close.shift(-X) - close) / close` for X ∈ {5, 10, 20, 50} | 4 target columns |
| 6 | Drop NaN rows from forward-return construction | **~359,750 final rows** |

**Per rolling split (preprocessing inside each fold):**
1. Replace inf/NaN with NaN
2. Clip features at training-set 1%/99% quantiles
3. Impute remaining NaN with training-set median
4. Standardize via training-set `StandardScaler`
5. Optional PCA (80% variance) — Logistic and Generative models only

---

## Agent-Assisted Factor Construction

A **Claude LLM agent** was used to systematically survey market microstructure literature and propose HFT factors backed by empirical theory. All 20 factors include explicit paper citations in code comments. Rolling window parameter: `n = 10` bars.

### 20 Base Factors

| # | Factor | Formula (simplified) | Academic Grounding |
|---|--------|---------------------|-------------------|
| 1 | **RS** (Relative Spread) | `(close_ask − close_bid) / close` | Kyle (1985) |
| 2 | **SCZ** (Std. Close Spread Zone) | `(RS − mean(RS,n)) / std(RS,n)` | Kyle (1985) |
| 3 | **ISA** (Intra-Bar Spread Asymmetry) | `((high_ask−high_bid) − (low_ask−low_bid)) / close` | Glosten-Milgrom (1985) |
| 4 | **MPR** (Mid-Price Log Return) | `log(close) − log(close.shift(n))` | Jegadeesh-Titman (1993) |
| 5 | **IPS** (Intrabar Price Skewness) | `(close − open) / (high − low)` | Williams %R analog |
| 6 | **OCG** (Open-to-Close Gap) | `(open − close.shift(1)) / close.shift(1)` | — |
| 7 | **RER** (Range Expansion Ratio) | `(high−low) / mean(high−low, n)` | — |
| 8 | **MS** (MA Spread / Momentum Signal) | `(MA5 − MA20) / MA20` | Jegadeesh-Titman (1993) |
| 9 | **MR** (Mean Reversion) | `(close − MA20) / MA20` | — |
| 10 | **VDR** (Volume Density Ratio) | `volume / mean(volume, n)` | Amihud (2002) |
| 11 | **PVC** (Price-Volume Correlation) | `rolling_corr(close.diff(), volume, n)` | Amihud (2002) |
| 12 | **VWMD** (VWAP Mid Deviation) | `(close − VWAP) / VWAP` | — |
| 13 | **VACC_R** (Volume Acceleration) | `(volume − volume.shift(1)) / volume.shift(1)` | Amihud (2002) |
| 14 | **PV** (Parkinson Volatility) | `sqrt(mean(log(high/low)², n) / (4·ln2))` | Parkinson (1980) |
| 15 | **GKV** (Garman-Klass Volatility) | `0.5·log(high/low)² − (2·ln2−1)·log(close/open)²` | Garman-Klass (1980) |
| 16 | **VRR** (Volatility Regime Ratio) | `mean(PV, 5) / mean(PV, 60)` | — |
| 17 | **BVA** (Bid-Ask Volatility Asymmetry) | `std(close_bid, n) / std(close_ask, n)` | — |
| 18 | **RIS** (Roll Implied Spread) | `2·sqrt(max(−cov(Δclose, Δclose.shift(1), n), 0))` | Roll (1984) |
| 19 | **KL** (Kyle's Lambda) | `cov(Δclose, OF, n) / var(OF, n)` where `OF = sign(Δclose)·volume` | Kyle (1985) |
| 20 | **AIR** (Amihud Illiquidity Ratio) | `mean(|Δclose| / volume, n)` | Amihud (2002) |

### Feature Expansion: 20 → 149 Features

| Component | Count | Description |
|-----------|-------|-------------|
| Base factors | 20 | Original 20 microstructure factors |
| Lag features | 60 | Each factor × lags {1, 5, 20}: `factor.shift(l)` |
| Change features | 60 | Each factor × periods {1, 5, 20}: `factor.pct_change(l)` |
| Time encodings | 4 | `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos` |
| Session flags | 3 | `is_asia_session`, `is_europe_session`, `is_us_session` |
| Regime indicators | 2 | `regime_low_vol`, `regime_high_vol` (vol terciles) |
| **Total** | **149** | Used in all rolling model backtests |

---

## Factor Testing Results

### RankIC Analysis — 5-minute Horizon (29 factors tested)

IC computed via 600-bar rolling windows; Spearman rank correlation against `ret_5m`.

**7 out of 29 factors achieve |RankIC| > 3%.** The strongest signal, MR (Mid-Return), is the only one exceeding the 5% threshold.

| Factor | IC | IC_IR | RankIC | RankIR | Pos IC% | Signal |
|--------|----|-------|--------|--------|---------|--------|
| **MR** | -0.0486 | -0.443 | **-0.0547** | **-0.605** | 33.1% | Strong mean-reversion |
| **MS** | -0.0418 | -0.389 | -0.0467 | -0.522 | 33.4% | Mean-reversion |
| **MPR** | -0.0403 | -0.371 | -0.0460 | -0.523 | 33.7% | Mean-reversion |
| **VWMD** | -0.0365 | -0.343 | -0.0437 | -0.499 | 35.7% | Mean-reversion |
| **PV** | +0.0313 | +0.309 | +0.0343 | +0.410 | 61.1% | Volatility signal |
| **GKV** | +0.0304 | +0.392 | +0.0323 | +0.527 | 66.4% | Most stable IC-IR |
| **KL** | +0.0251 | +0.267 | +0.0304 | +0.376 | 62.4% | Price impact |
| AIR | +0.0254 | +0.270 | +0.0295 | +0.367 | 62.3% | Illiquidity |
| IPS | -0.0087 | -0.187 | -0.0187 | -0.417 | 40.9% | Directional bias |
| RS | +0.0101 | +0.138 | +0.0151 | +0.220 | 55.1% | Spread |
| regime_high_vol | +0.0108 | +0.119 | +0.0195 | +0.234 | 52.8% | Regime |
| is_us_session | +0.0068 | +0.078 | +0.0138 | +0.173 | 44.9% | Session |
| regime_low_vol | -0.0099 | -0.139 | -0.0178 | -0.239 | 45.2% | Regime |
| PVC | -0.0083 | -0.103 | -0.0120 | -0.152 | 45.2% | Vol-price corr |
| OCG | -0.0049 | -0.098 | -0.0049 | -0.118 | 45.9% | Gap |
| VRR | +0.0081 | +0.081 | +0.0068 | +0.081 | 54.3% | Vol ratio |
| RIS | +0.0097 | +0.097 | +0.0070 | +0.088 | 51.9% | Roll spread |
| RER, ISA, VDR, VACC_R, SCZ, hour/min/session | — | — | < ±0.005 | — | ~50% | Noise |

**Key Findings**:
- Mean-reversion dominates HFT signals; MR IC-IR strengthens from -0.605 (5m) to **-0.631 (50m)**
- GKV is the most stable factor (Pos IC% = 66.4%, highest of all)
- Most factors show |RankIC| < 3% at short horizons, consistent with near-efficient HFT markets
- Volatility and microstructure factors (PV, GKV, KL, AIR) have consistent positive IC

---

## Key Improvements Implemented

### 1. Time-Series Leakage Prevention
- Added per-horizon embargo gaps between training and validation sets
- Eliminates look-ahead bias from forward-return label construction

### 2. Feature Expansion (20 → 149 features)
- **Lag terms** (1, 5, 20 periods): factor persistence — 60 features
- **Rate-of-change** (1, 5, 20 periods): factor momentum — 60 features
- **Cyclical time encodings**: hour/minute sin-cos, session flags (Asia/Europe/US) — 7 features
- **Volatility regime indicators**: low/high vol state features — 2 features

### 3. Rolling Window Configuration
| Parameter | Previous | Improved |
|-----------|----------|----------|
| Training window | 500 | 1800 samples |
| Validation window | 100 | 200 samples |
| Step size | 500 | 1800 samples |
| Embargo | None | Per-horizon (5/10/20/50) |

---

## Results Summary

> **Random baseline**: 33.33% accuracy, Log Loss = 1.099, Brier Score = 0.667 (uniform 3-class)

### Full Model Comparison — 5-Minute Horizon

| Model | Accuracy | vs. Baseline | Log Loss | Brier Score | AUC-ROC |
|-------|----------|-------------|----------|-------------|---------|
| **Ensemble (Lasso+LDA+Bagging)** | **51.19%** | **+17.86 pp** | **0.9905** | **0.5898** | 0.6534 |
| **Bagging** | 50.92% | +17.59 pp | 0.9947 | 0.5924 | 0.5508 |
| **GRU** | 50.07% | +16.74 pp | 1.0024 | 0.5981 | **0.5633** |
| LR_lasso (with PCA) | 49.91% | +16.58 pp | 1.0339 | 0.6132 | 0.5401 |
| LDA (with PCA) | 49.94% | +16.61 pp | 1.0336 | 0.6133 | 0.5405 |
| XGBoost | 49.38% | +16.05 pp | 1.0701 | 0.6300 | 0.5416 |
| LightGBM | 49.13% | +15.80 pp | 1.1089 | 0.6450 | 0.5385 |
| Random Forest | 45.62% | +12.29 pp | 1.0480 | 0.6308 | 0.5507 |
| KNN (with PCA) | 44.86% | +11.53 pp | 6.3337 | 0.7271 | 0.5122 |
| Naive Bayes (with PCA) | 43.85% | +10.52 pp | 1.5099 | 0.7499 | 0.5286 |

**Best accuracy**: Ensemble at **51.19%** (+17.86 pp over baseline), also best log loss (0.9905) and Brier (0.5898).
**Best AUC-ROC**: GRU at **0.5633** — best class discrimination among individual models.
**Best single-model Brier Score**: Bagging at **0.5924** — strongest calibration without ensembling.

### Model Performance by Horizon

| Model | 5m Acc | 5m Brier | 5m AUC | 10m Acc | 10m Brier | 20m Acc | 20m Brier | 50m Acc | 50m Brier |
|-------|--------|---------|--------|---------|----------|---------|----------|---------|----------|
| **GRU** | 0.5007 | 0.5981 | **0.5633** | — | — | — | — | — | — |
| **Bagging** | **0.5092** | **0.5924** | 0.5508 | **0.4995** | **0.6038** | 0.4814 | 0.6293 | — | — |
| LR_lasso (PCA) | 0.4991 | 0.6132 | 0.5401 | 0.4871 | 0.6299 | 0.4788 | 0.6469 | 0.4378 | 0.6964 |
| LDA (PCA) | 0.4994 | 0.6133 | 0.5405 | 0.4879 | 0.6298 | 0.4801 | 0.6475 | 0.4404 | 0.6951 |
| XGBoost | 0.4938 | 0.6300 | 0.5416 | 0.4835 | 0.6533 | 0.4658 | 0.6911 | — | — |
| LightGBM | 0.4913 | 0.6450 | 0.5385 | 0.4759 | 0.6737 | 0.4607 | 0.7166 | — | — |
| Random Forest | 0.4562 | 0.6308 | 0.5507 | 0.4466 | 0.6335 | 0.4310 | **0.6411** | — | — |

**Key Findings**:
1. **Ensemble (Lasso+LDA+Bagging)** achieves highest 5m accuracy (51.19%), log loss (0.9905), and Brier (0.5898) via probability averaging
2. **Bagging** leads on Brier Score and accuracy among single models at 5m; calibration is strong
3. **GRU** leads on AUC-ROC (0.5633) — strongest class discrimination, only evaluated at 5m
4. **LDA with PCA** is competitive with LR_lasso (~49.9%), matching it on all metrics with full interpretability
5. Accuracy degrades with horizon (5m ~51% → 20m ~48%), consistent with weaker signal at longer lags
6. **Tree models degrade faster**: LightGBM Brier 0.645→0.717 (5m→20m); linear models are more stable

---

## Strategy Backtest Results (XAG/USD, Jan 2025 – Jan 2026)

Five models were backtested using `vectorbt` on the 5-minute trend prediction task. Each uses a probability-filtered long/short signal with fees (0.03%) and slippage (0.01%) applied.

**Benchmark (Buy & Hold): +221.13%** — XAG/USD experienced an exceptional bull run in 2025.

**Signal construction (probability-gated entries)**:
- **Lasso LR / LDA**: `prob_up >= 0.65` to enter long; `prob_down >= 0.65` to exit
- **Bagging**: `prob_up >= 0.50` to enter; `prob_down >= 0.50` to exit
- **GRU**: `prob_up >= 0.40` to enter; `prob_down >= 0.40` to exit
- **Ensemble**: Average probabilities of Lasso + LDA + Bagging; threshold `>= 0.50`

| Metric | Lasso LR | LDA | Bagging | GRU | Ensemble |
|--------|----------|-----|---------|-----|----------|
| **Total Return** | 84.21% | 60.90% | **145.83%** | 33.56% | 86.40% |
| **Max Drawdown** | 15.27% | 21.43% | 24.89% | 23.31% | 21.12% |
| **Win Rate** | 53.56% | 52.42% | 55.02% | 55.58% | **58.78%** |
| **Sharpe Ratio** | 5.49 | 4.34 | **7.54** | 2.84 | 5.27 |
| **Sortino Ratio** | 7.43 | 5.80 | **10.21** | 3.80 | 7.07 |
| **Calmar Ratio** | 575.34 | 148.82 | **2968.86** | 31.64 | 454.09 |
| **Profit Factor** | 1.51 | 1.34 | **1.66** | 1.31 | 1.53 |
| **Total Trades** | 240 | 249 | 210 | **458** | 149 |

**Key observations**:
- None of the strategies beats buy-and-hold due to XAG's 2025 bull run; the strategies reduce drawdown at the cost of total return
- **Bagging dominates all risk-adjusted metrics**: Sharpe 7.54, Sortino 10.21, Calmar 2969, highest total return (145.83%)
- **Ensemble achieves the best win rate (58.78%)** with the fewest trades (149), indicating highest signal quality per trade
- **GRU's low threshold (0.40)** generates 458 trades but lowest return — probability miscalibration at this level
- **Lasso LR** has the tightest drawdown (15.27%), useful if capital preservation is the priority

---

## Model Families Tested

### 1. Logistic Regression
LR_plain, LR_lasso (L1), LR_ridge (L2), LR_elasticnet — with and without PCA (80% variance). **PCA consistently adds ~3pp accuracy** and substantially improves log loss.

### 2. Generative Models
Gaussian Naive Bayes, LDA (`solver='lsqr', shrinkage='auto'`), QDA (`reg_param=0.5`), KNN (k=5). LDA with PCA matches logistic models in both accuracy and log loss.

### 3. Tree-Based Models
Bagging (DecisionTree, max_depth=4), Random Forest, LightGBM, XGBoost. No PCA applied. **Bagging achieves top accuracy at 5m** but tree models generally have higher log loss than linear models at longer horizons.

### 4. GRU Neural Network
Gated Recurrent Unit with rolling sequence inputs (seq_len=20, features=[return, volume, volatility, close, MA10]). **Highest AUC-ROC (0.5633) and log loss (1.0024)** on the 5m horizon — lowest log loss among all models, beating the random baseline (1.099). Accuracy: 50.07%. Training time: ~88 minutes per full backtest.

---

## Recommended Model Config

| Priority | Model | Accuracy (5m) | Log Loss (5m) | AUC (5m) | Trade-off |
|----------|-------|--------------|--------------|----------|-----------|
| Best AUC + log loss | **GRU** | 50.07% | **1.0024** | **0.5633** | Slow (~88 min/backtest) |
| Best accuracy + speed | **Bagging** | 50.92% | 0.9947 | 0.5508 | Fast, simple |
| Best interpretability | **LR_lasso + PCA** | 49.91% | 1.0339 | 0.5401 | Linear, transparent |
| Most stable across horizons | **Random Forest** | 45.62% | 1.0480 | 0.5507 | Consistent Brier Score |

**Recommended default**: Bagging — best risk-adjusted backtest performance (Sharpe 7.54, Calmar 2969), best Brier Score (0.5924), fast training

```python
BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=42),
    n_estimators=100,
    bootstrap=True,
    random_state=42
)
```

Backtest settings: `training_size=1800, validation_size=200, step_size=1800, embargo=horizon`

---

## Baseline Comparison
| Benchmark | Value |
|-----------|-------|
| Random 3-class accuracy | 33.33% |
| Log Loss (uniform random) | 1.099 |
| Buy & Hold Return (2025) | 221.13% |

---

## Future Work
1. **Probability Calibration**: Platt scaling / Isotonic regression (especially KNN, Naive Bayes)
2. **Threshold Optimization**: Tune signal thresholds to maximize Sharpe, not accuracy
3. **Transaction Cost Modeling**: Evaluate prediction-based strategy post-fee
4. **Cross-Asset Validation**: Test factor generalization on CRYPTO, CURRENCY, EQUITY
5. **GRU at Longer Horizons**: Currently only evaluated at 5m
6. ~~**Ensemble**: Combine Bagging + GRU via meta-learning~~ → **Implemented**: Equal-weight probability averaging (Lasso + LDA + Bagging); achieves highest accuracy (51.19%) and best win rate (58.78%) in backtest
7. **GRU Threshold Tuning**: GRU at threshold 0.40 overtrades (458 trades); optimize threshold to improve risk-adjusted returns

---

## Dependencies
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost lightgbm vectorbt ta-lib torch
```

## How to Run
1. Open `Quantitative strategy.ipynb`
2. Run all cells top to bottom
3. Review results under sections 1.3 (IC), 1.4 (models), 1.5 (strategy)

## Known Issues & Fixes

**XGBoost label encoding**: Requires labels `{0, 1, 2}`. Map `{-1, 0, 1}` → `{0, 1, 2}` before `fit`.

**LDA singular covariance**: Use `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`.

**QDA singular covariance**: Use `QuadraticDiscriminantAnalysis(reg_param=0.5)`.
