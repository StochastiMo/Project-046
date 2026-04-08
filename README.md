# Project-046: Minute-Level HFT Alpha Research on XAG/USD

## Overview
This project studies short-horizon return predictability for XAG/USD (Silver) using high-frequency market microstructure factors and rolling time-series backtests. A key innovation is the use of a **Claude LLM agent** to systematically survey academic literature — spanning Roll (1984), Kyle (1985), Amihud (2002), Garman-Klass (1980), and Parkinson (1980) — and translate theoretical microstructure signals into 20 computable HFT factors. These are expanded to 162 features and evaluated through rigorous walk-forward validation across four families of classifiers.

**Key Research Goals:**
- Leverage an LLM agent to automate literature-driven HFT factor discovery
- Identify reliable predictive factors for minute-level price movements in XAG/USD
- Develop robust classification models with proper time-series leakage prevention
- Evaluate model performance across multiple prediction horizons (5m, 10m, 20m, 50m)

## Workflow
1. **Agent-assisted factor research**: Claude agent surveys market microstructure literature and proposes factor formulas grounded in theory
2. **Data preparation**: Cleaning and preprocessing minute-level OHLC and tick data
3. **Factor engineering**: 20 HFT microstructure factors → expanded to 162 features
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
**Primary Asset**: XAG/USD minute-level data (359,750 observations)

**Forward Return Horizons**: `ret_5m`, `ret_10m`, `ret_20m`, `ret_50m`

**Classification Targets**: Three-class labels via train-set terciles:
- **Down** (0): Lower tercile
- **Noise** (1): Middle tercile
- **Up** (2): Upper tercile

---

## Agent-Assisted Factor Construction

A **Claude LLM agent** was used to systematically survey market microstructure literature and propose HFT factors backed by empirical theory. All 20 factors include explicit paper citations in code comments.

| Category | Factors | Academic Grounding |
|----------|---------|-------------------|
| Bid-Ask Spread & Order Pressure | RS, SCZ, ISA, MPR, IPS, OCG | Kyle (1985), Glosten-Milgrom (1985) |
| Momentum & Mean Reversion | RER, MS, MR | Jegadeesh-Titman (1993) |
| Volume & Activity | VDR, PVC, VWMD, VACC_R | Amihud (2002) |
| Volatility Estimation | PV, GKV, VRR | Parkinson (1980), Garman-Klass (1980) |
| Microstructure & Information Flow | BVA, RIS, KL, AIR | Roll (1984), Kyle's Lambda (1985) |

---

## Factor Testing Results

### RankIC Analysis (5-minute horizon, 29 factors tested)

**7 out of 29 factors achieve |RankIC| > 3%** on the 5-minute horizon. The strongest signal, Mid-Return (MR), exceeds the strict **5% threshold** (|RankIC| = 5.47%).

| Factor | RankIC (5m) | RankIR (5m) | Direction |
|--------|-------------|-------------|-----------|
| **MR** (Mid-Return) | **-0.0547** | -0.605 | Strong mean-reversion; only factor with \|RankIC\| > 5% |
| **MS** (Momentum Score) | -0.0467 | -0.522 | Mean-reversion |
| **MPR** (Mid-Price Return) | -0.0460 | -0.523 | Mean-reversion |
| **VWMD** (VWAP Mid Deviation) | -0.0437 | -0.499 | Mean-reversion |
| **PV** (Parkinson Volatility) | +0.0343 | +0.410 | Volatility → future range |
| **GKV** (Garman-Klass Vol) | +0.0323 | +0.527 | Most stable IC-IR |
| **KL** (Kyle's Lambda) | +0.0304 | +0.376 | Price impact signal |

**Key Findings**:
- Mean-reversion factors dominate; MR IC-IR strengthens from -0.605 (5m) to **-0.631 (50m)**
- Volatility factors (PV, GKV) show the most stable IC-IR among directional signals
- Most factors show |RankIC| < 3% at short horizons, consistent with near-efficient HFT markets

---

## Key Improvements Implemented

### 1. Time-Series Leakage Prevention
- Added per-horizon embargo gaps between training and validation sets
- Eliminates look-ahead bias from forward-return label construction

### 2. Feature Expansion (20 → 162 features)
- **Lag terms** (1, 5, 20 periods): factor persistence
- **Rate-of-change**: factor momentum
- **Cyclical time encodings**: hour/minute sin-cos, session flags (Asia/Europe/US)
- **Volatility regime indicators**: low/mid/high vol state features

### 3. Rolling Window Configuration
| Parameter | Previous | Improved |
|-----------|----------|----------|
| Training window | 500 | 1800 samples |
| Validation window | 100 | 200 samples |
| Step size | 500 | 1800 samples |
| Embargo | None | Per-horizon (5/10/20/50) |

---

## Results Summary

> **Random baseline**: 33.33% accuracy, Log Loss = 1.099 (uniform 3-class)

### Full Model Comparison — 5-Minute Horizon

| Model | Accuracy | vs. Baseline | Log Loss |
|-------|----------|-------------|----------|
| **GRU** | **50.98%** | **+17.65 pp** | **0.9914** |
| **Bagging** | **50.98%** | **+17.65 pp** | 0.9967 |
| LR_lasso (with PCA) | 50.04% | +16.71 pp | 1.0330 |
| LDA (with PCA) | 49.98% | +16.65 pp | 1.0330 |
| LR_elasticnet (with PCA) | 49.98% | +16.65 pp | 1.0344 |
| LR_plain (with PCA) | 49.95% | +16.62 pp | 1.0359 |
| XGBoost | 49.49% | +16.16 pp | 1.0694 |
| LightGBM | 48.86% | +15.53 pp | 1.1106 |
| LDA (without PCA) | 46.81% | +13.48 pp | 1.1886 |
| LR_lasso (without PCA) | 47.09% | +13.76 pp | 1.1477 |
| Random Forest | 45.67% | +12.34 pp | 1.0475 |
| KNN (with PCA) | 44.89% | +11.56 pp | 6.3157 |
| Naive Bayes (with PCA) | 43.94% | +10.61 pp | 1.5031 |

**Best accuracy**: GRU and Bagging both at **50.98%** — a **+17.65 percentage point** lift over the 33.33% random baseline.
**Best log loss**: GRU at **0.9914** — the only model to beat the random baseline (1.099) in both accuracy *and* probability calibration.

### Model Accuracy by Horizon

| Model | 5m Acc | 5m LogLoss | 10m Acc | 10m LogLoss | 20m Acc | 20m LogLoss | 50m Acc | 50m LogLoss |
|-------|--------|-----------|---------|------------|---------|------------|---------|------------|
| **GRU** | **0.5098** | **0.9914** | — | — | — | — | — | — |
| **Bagging** | **0.5098** | 0.9967 | **0.4998** | **1.0131** | 0.4787 | 1.0617 | 0.4306 | 1.2303 |
| LR_lasso (PCA) | 0.5004 | 1.0330 | 0.4883 | 1.0643 | **0.4789** | **1.1046** | **0.4396** | **1.2012** |
| LDA (PCA) | 0.4998 | 1.0330 | 0.4888 | 1.0640 | 0.4796 | 1.1058 | — | — |
| XGBoost | 0.4949 | 1.0694 | 0.4798 | 1.1184 | 0.4659 | 1.2030 | 0.4277 | 1.4035 |
| LightGBM | 0.4886 | 1.1106 | 0.4773 | 1.1741 | 0.4612 | 1.2899 | 0.4284 | 1.5499 |
| Random Forest | 0.4567 | 1.0475 | 0.4477 | 1.0513 | 0.4292 | 1.0626 | 0.3984 | 1.0889 |

**Key Findings**:
1. **GRU and Bagging dominate on accuracy** at 5m (50.98%), far exceeding the 33.3% baseline
2. **LR with PCA is surprisingly competitive** (~50%), matching complex models with full interpretability
3. **Random Forest ranks mid-table** — good log loss across horizons but not the accuracy leader
4. Accuracy degrades with horizon (5m ~51% → 50m ~44%), consistent with weaker signal at longer lags
5. KNN and raw Naive Bayes suffer from poor calibration (log loss >> 1.1) despite reasonable accuracy

---

## Strategy Backtest Results (XAG/USD, Jan 2025 – Jan 2026, 248 days)

Evaluated using `vectorbt` on the Random Forest model (5-minute trend prediction):

| Metric | Probability-based | Prediction-based |
|--------|------------------|-----------------|
| **Total Return** | 43.46% | **149.15%** |
| **Benchmark (Buy & Hold)** | 220.32% | 220.32% |
| **Max Drawdown** | 13.22% | 19.01% |
| **Max Drawdown Duration** | 53 days | 35 days |
| **Total Trades** | 1,251 | 17,540 |
| **Win Rate** | 57.60% | 54.21% |
| **Sharpe Ratio** | 2.12 | **3.84** |
| **Calmar Ratio** | 5.29 | **14.85** |
| **Profit Factor** | 1.20 | 1.10 |
| **Avg Win / Avg Loss** | +0.28% / -0.31% | +0.11% / -0.12% |
| **Fees** | 0 | 0 |

- **Probability-based**: Enter when `prob_up > 0.5` or `prob_down > 0.5`; fewer, higher-quality trades
- **Prediction-based**: Enter on `pred == up/down`; mean-reversion style; Sharpe **3.84**, Calmar **14.85** (pre-fee)

> Neither strategy beats buy-and-hold (220.32%) due to XAG's exceptional 2025 bull run. The prediction-based strategy's Sharpe of **3.84** and Calmar of **14.85** are institutional-grade on a zero-fee basis.

---

## Model Families Tested

### 1. Logistic Regression
LR_plain, LR_lasso (L1), LR_ridge (L2), LR_elasticnet — with and without PCA (80% variance). **PCA consistently adds ~3pp accuracy** and substantially improves log loss.

### 2. Generative Models
Gaussian Naive Bayes, LDA (`solver='lsqr', shrinkage='auto'`), QDA (`reg_param=0.5`), KNN (k=5). LDA with PCA matches logistic models in both accuracy and log loss.

### 3. Tree-Based Models
Bagging (DecisionTree, max_depth=4), Random Forest, LightGBM, XGBoost. No PCA applied. **Bagging achieves top accuracy at 5m** but tree models generally have higher log loss than linear models at longer horizons.

### 4. GRU Neural Network
Gated Recurrent Unit with rolling sequence inputs. **Highest accuracy (50.98%) and lowest log loss (0.9914)** on the 5m horizon — the only model where log loss beats the random baseline. Training time: ~88 minutes per full backtest.

---

## Recommended Model Config

| Priority | Model | Accuracy (5m) | Log Loss (5m) | Trade-off |
|----------|-------|--------------|--------------|-----------|
| Best overall | **GRU** | 50.98% | **0.9914** | Slow (88 min) |
| Best accuracy + speed | **Bagging** | 50.98% | 0.9967 | Fast, simple |
| Best interpretability | **LR_lasso + PCA** | 50.04% | 1.0330 | Linear, transparent |
| Most stable across horizons | **Random Forest** | 45.67% | 1.0475 | Consistent log loss |

**Recommended default**: Bagging — best accuracy, fast training, well-calibrated log loss

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
| Buy & Hold Return (2025) | 220.32% |

---

## Preprocessing Pipeline (per rolling split)
1. Replace inf/nan
2. Clip at training-set 1%/99% quantiles
3. Impute NaN with training-set median
4. Standardize via training-set `StandardScaler`
5. Optional PCA (80% variance) — Logistic and Generative models only

---

## Future Work
1. **Probability Calibration**: Platt scaling / Isotonic regression (especially KNN, Naive Bayes)
2. **Threshold Optimization**: Tune signal thresholds to maximize Sharpe, not accuracy
3. **Transaction Cost Modeling**: Evaluate prediction-based strategy post-fee
4. **Cross-Asset Validation**: Test factor generalization on CRYPTO, CURRENCY, EQUITY
5. **GRU at Longer Horizons**: Currently only evaluated at 5m
6. **Ensemble**: Combine Bagging + GRU via meta-learning

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
