# Strategy Alignment Document

## Strategy Description

We start with a broad, liquid European equity universe and let the model, not judgment, decide the winners. 

**Stage 1: Partial Least Squares (PLS)**
- Compresses highly correlated macro and style variables into a small set of components
- These components capture the forces actually driving European equity returns
- Defines what tends to win in Europe

**Stage 2: Elastic Net Regression**
- For each stock, evaluates how strongly and consistently it benefits from these return drivers
- Shrinks weak or unstable relationships to zero
- Retains only robust signals
- Every stock receives a model-implied expected return

**Stage 3: Portfolio Construction**
- Rank stocks by model-implied expected return
- Construct constrained long-only portfolio from top-scoring names

**Winners Definition**: Stocks with the most stable and positive exposure to the factors that genuinely drive European returns after noise and overfitting are removed.

---

## Implementation Alignment

### ✅ 1. Broad, Liquid European Equity Universe
**Location**: `data/equity_universe.py`
- **Implementation**: Top 500 market cap European stocks
- **Data Source**: Yahoo Finance
- **No Judgment**: Universe defined by market cap, not subjective selection
- **Status**: ✅ Aligned

### ✅ 2. PLS: Compress Factors into Return-Driving Components
**Location**: `models/pls_model.py`
- **Implementation**: 
  - Input: Macro + style factors (X) and European stock returns (Y)
  - PLS maximizes covariance between X and Y
  - Extracts components that capture forces driving European equity returns
  - Components define what tends to win in Europe
- **Key Code**: `PLSRegression.fit(X_scaled, Y_scaled)` - maximizes covariance
- **Status**: ✅ Aligned

### ✅ 3. Elastic Net: Evaluate Stock Exposures
**Location**: `models/elastic_net_model.py`
- **Implementation**:
  - For each stock, fits Elastic Net using PLS components as features
  - Uses time-series cross-validation to ensure consistency
  - Regularization (L1/L2) shrinks weak/unstable relationships to zero
  - Only robust signals survive (non-zero coefficients)
  - Each stock gets model-implied expected return
- **Key Code**: 
  - `ElasticNetCV` with time-series CV
  - Coefficients represent stock's exposure to return drivers
  - Weak exposures → shrunk to zero
- **Status**: ✅ Aligned

### ✅ 4. Rank Stocks by Expected Return
**Location**: `portfolio/ranking.py`
- **Implementation**:
  - Sorts stocks by model-implied expected return (descending)
  - Filters negative returns (long-only constraint)
  - Top-scoring stocks = winners
- **Status**: ✅ Aligned

### ✅ 5. Construct Constrained Long-Only Portfolio
**Location**: `portfolio/optimization.py`
- **Implementation**:
  - Selects top N stocks by expected return
  - Equal-weighted (can be extended to optimization)
  - Long-only constraint enforced
- **Status**: ✅ Aligned

---

## Key Design Principles

### 1. Model Decides, Not Judgment
- ✅ Universe: Market cap-based, not hand-picked
- ✅ Factor selection: All factors included, PLS finds what matters
- ✅ Stock selection: Elastic Net regularization, not manual filtering
- ✅ Portfolio: Top-ranked by model score, not subjective picks

### 2. Captures Genuine Return Drivers
- ✅ PLS maximizes covariance with actual European equity returns
- ✅ Components represent forces driving returns, not spurious correlations
- ✅ Time-series CV ensures consistency over time

### 3. Removes Noise and Overfitting
- ✅ PLS reduces dimensionality (removes redundant factors)
- ✅ Elastic Net regularization shrinks weak relationships to zero
- ✅ Cross-validation prevents overfitting
- ✅ Only robust, stable signals retained

### 4. Winners = Stable Positive Exposure
- ✅ Elastic Net coefficients measure exposure strength
- ✅ Regularization ensures stability (weak → zero)
- ✅ Expected return = function of robust exposures
- ✅ Top-ranked = most stable positive exposure to return drivers

---

## Model Flow

```
Broad European Equity Universe (Top 500)
    ↓
Macro + Style Factors (many, correlated)
    ↓
[PLS Model] → Components capturing European return drivers
    ↓
[Elastic Net] → Stock exposures to components
    ↓
[Regularization] → Shrink weak/unstable to zero
    ↓
Model-Implied Expected Returns (robust signals only)
    ↓
Rank by Expected Return
    ↓
Top-Scoring Stocks = Winners
    ↓
Constrained Long-Only Portfolio
```

---

## Verification Checklist

- [x] PLS uses European stock returns as Y (not aggregated indices)
- [x] PLS maximizes covariance (finds return drivers)
- [x] Elastic Net uses PLS components as features
- [x] Elastic Net evaluates each stock individually
- [x] Regularization shrinks weak relationships
- [x] Cross-validation ensures consistency
- [x] Expected returns based on robust exposures
- [x] Ranking by expected return (not other criteria)
- [x] Portfolio from top-scoring names only
- [x] Long-only constraint enforced
- [x] No subjective judgment in selection

---

## Summary

The implementation is **fully aligned** with the strategy description:

1. ✅ **Broad universe**: Top 500 European stocks (market cap-based)
2. ✅ **PLS**: Compresses factors into components capturing European return drivers
3. ✅ **Elastic Net**: Evaluates each stock's robust exposure to return drivers
4. ✅ **Regularization**: Removes noise and overfitting (weak → zero)
5. ✅ **Ranking**: By model-implied expected return
6. ✅ **Portfolio**: Constrained long-only from top-scoring names
7. ✅ **Winners**: Stocks with most stable positive exposure to return drivers

**The model decides, not judgment. Winners are determined by stable exposure to factors genuinely driving European returns.**
