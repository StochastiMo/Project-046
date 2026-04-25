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

> **Random baseline**: 33.33% accuracy, Log Loss = 1.099, Brier Score = 0.667 (uniform 3-class)

### Full Model Comparison — 5-Minute Horizon

| Model | Accuracy | vs. Baseline | Log Loss | Brier Score | AUC-ROC |
|-------|----------|-------------|----------|-------------|---------|
| **Ensemble (Lasso+LDA+Bagging)** | **51.19%** | **+17.86 pp** | — | — | — |
| **Bagging** | 50.92% | +17.59 pp | 0.9947 | **0.5924** | 0.5508 |
| **GRU** | 50.07% | +16.74 pp | 1.0024 | 0.5981 | **0.5633** |
| LR_lasso (with PCA) | 49.61% | +16.28 pp | 1.0358 | 0.6143 | 0.5381 |
| LDA (with PCA) | 49.63% | +16.30 pp | 1.0359 | 0.6147 | 0.5385 |
| XGBoost | 49.38% | +16.05 pp | 1.0701 | 0.6300 | 0.5416 |
| LightGBM | 49.13% | +15.80 pp | 1.1089 | 0.6450 | 0.5385 |
| Random Forest | 45.62% | +12.29 pp | 1.0480 | 0.6308 | 0.5507 |
| KNN (with PCA) | 44.89% | +11.56 pp | 6.3157 | 0.7476 | 0.5092 |
| Naive Bayes (with PCA) | 43.94% | +10.61 pp | 1.5031 | 0.7543 | 0.5292 |

**Best accuracy**: Ensemble at **51.19%** (+17.86 pp over baseline).
**Best Brier Score**: Bagging at **0.5924** — best probability calibration among all models.
**Best AUC-ROC**: GRU at **0.5633** — best class discrimination ability.
**Best log loss**: GRU at **1.0024** — beats random baseline (1.099).

### Model Accuracy by Horizon

| Model | 5m Acc | 5m Brier | 5m AUC | 10m Acc | 10m Brier | 20m Acc | 20m Brier | 50m Acc | 50m Brier |
|-------|--------|---------|--------|---------|----------|---------|----------|---------|----------|
| **GRU** | 0.5007 | 0.5981 | **0.5633** | — | — | — | — | — | — |
| **Bagging** | **0.5092** | **0.5924** | 0.5508 | **0.4998** | **0.6038** | 0.4787 | 0.6293 | 0.4306 | 0.7009 |
| LR_lasso (PCA) | 0.4961 | 0.6143 | 0.5381 | 0.4883 | 0.6299 | **0.4789** | 0.6469 | **0.4396** | 0.6964 |
| LDA (PCA) | 0.4963 | 0.6147 | 0.5385 | 0.4888 | 0.6298 | 0.4796 | 0.6475 | — | 0.6951 |
| XGBoost | 0.4938 | 0.6300 | 0.5416 | 0.4798 | 0.6533 | 0.4659 | 0.6911 | 0.4277 | 0.7771 |
| LightGBM | 0.4913 | 0.6450 | 0.5385 | 0.4773 | 0.6737 | 0.4612 | 0.7166 | 0.4284 | 0.8088 |
| Random Forest | 0.4562 | 0.6308 | 0.5507 | 0.4477 | 0.6335 | 0.4292 | **0.6411** | 0.3984 | **0.6597** |

**Key Findings**:
1. **Ensemble (Lasso+LDA+Bagging)** achieves highest 5m accuracy (51.19%) via probability averaging
2. **Bagging** leads on Brier Score (best calibration) and accuracy among single models at 5m
3. **GRU** leads on AUC-ROC (0.5633) — strongest class discrimination despite fewer trades
4. **LR with PCA is competitive** (~49.6%), matching complex models with full interpretability
5. Accuracy degrades with horizon (5m ~51% → 50m ~44%), consistent with weaker signal at longer lags
6. **Random Forest** shows the most stable Brier Score across horizons (0.631→0.660)

---

## Strategy Backtest Results (XAG/USD, Jan 2025 – Jan 2026)

Five models were backtested using `vectorbt` on the 5-minute trend prediction task. Each uses a probability-filtered long/short signal with fees (0.03%) and slippage (0.01%) applied.

**Benchmark (Buy & Hold): +221.13%**

| Metric | Lasso LR | LDA | Bagging | GRU | Ensemble |
|--------|----------|-----|---------|-----|----------|
| **Total Return** | 84.21% | 60.90% | **145.83%** | 33.56% | 86.40% |
| **Max Drawdown** | 15.27% | 21.43% | 24.89% | 23.31% | 21.12% |
| **Win Rate** | 53.56% | 52.42% | **55.02%** | 55.58% | 58.78% |
| **Sharpe Ratio** | 5.49 | 4.34 | **7.54** | 2.84 | 5.27 |
| **Calmar Ratio** | 575.34 | 148.82 | **2968.86** | 31.64 | 454.09 |
| **Profit Factor** | 1.51 | 1.34 | **1.66** | 1.31 | 1.53 |
| **Total Trades** | 240 | 249 | 210 | **458** | 149 |

**Signal construction**:
- **Lasso LR / LDA**: Enter when `pred == up` and `prob_up >= 0.65`; exit when `pred == down` and `prob_down >= 0.65`
- **Bagging**: Enter when `pred == up` and `prob_up >= 0.50`; exit when `pred == down` and `prob_down >= 0.50`
- **GRU**: Enter when `pred == up` and `prob_up >= 0.40`; exit when `pred == down` and `prob_down >= 0.40`
- **Ensemble**: Average probabilities of Lasso + LDA + Bagging; threshold `>= 0.50`

> None of the strategies beats buy-and-hold (+221.13%) due to XAG's exceptional 2025 bull run. **Bagging dominates on all risk-adjusted metrics** — Sharpe 7.54, Calmar 2969, highest total return (145.83%). **Ensemble achieves the best win rate (58.78%)** with the fewest trades (149), suggesting higher signal quality. GRU's lower threshold (0.40) generates the most trades (458) but the weakest returns, indicating probability miscalibration at that level.

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
