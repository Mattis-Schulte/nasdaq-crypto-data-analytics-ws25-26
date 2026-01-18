## Task 1 — Data Findings & Quality Analysis

### Splits correction summary
A total of **284** unadjusted split events were detected and corrected in the NASDAQ feed, representing **~11%** of the **2458** splits recorded in the `raw/splits_2000_2025.csv` file.

### NASDAQ (Base after split-check, before yfinance)

- **rows:** 9,390,260  
- **tickers:** 3,342  
- **date range:** 2000-01-03 -> 2025-12-12  
- **duplicates dropped:** 0
- **missing open/close:** ~0%
- **open distribution:** p01=0.5, p50=14.67, p99=8081.35
- **close distribution:** p01=0.5, p50=14.67, p99=8079.34
- **observed entries per ticker:** min 1, median 1838, max 6497
- **gap behavior:** median max gap ≈ 49 days, global max gap 890 days

Interpretation:

- The base feed seems largely in check with what is expected for the NASDAQ universe, with realistic price distributions etc.
- Some tickers have large gaps (up to 890 days), which may reflect delistings, ticker changes, mergers, or point to partial, to be corrected, histories
> Note: This is just measuring the quality of the existing rows; missing rows are not meant by missing open/close here.
---

### NASDAQ (Final after yfinance fill + outlier replacement)

- **rows:** 10,080,162  
- **tickers:** 3,343  
- **date range:** 2000-01-03 → 2025-12-31  
- **base missing rows:** 689,974 (**6.845%**)  
- **yfinance available rows:** 10,016,883 (**99.372%**)  
- **outliers (any):** 1,262,459 (**12.524%**)  

With `IQR_K = 1.5` and a return-based Tukey rule, **~12.5% of all NASDAQ rows** were flagged as outliers and (when possible) replaced by yfinance.
This might point to the from us choosen global IQR rule potentially being inappropiate and flagging legitimate volatility clusters as "outliers", as seen by the overlap of the corrections over time graph with known high-volatility periods (e.g., 2008 financial crisis, 2020 COVID crash); tho fine-tuning `IQR_K` could mitigate this to some extent.

---

### "Top 50 tickers by corrections" plot (fills vs replacements)

The stacked bar plot is a strong sanity check:

- Most corrected tickers are dominated by replacements, not fills
- This supports the conclusion that the potenially flawed outlier detector is the main driver of corrections

---

### Close (final/base/yf + corrections)

The time series line plots for individual tickers (e.g., AAPL, TSLA) show:
- base, yfinance and final series are mostly aligned
- the outlier replacements are unable to correct large errors in the base feed (e.g. stocks being off by a constant factor for larger periods; see PSTV in the early 2000s)
- treating yfinance as a trusted source might not always be optimal as shown by the SMX example

---

### Corrections over time (weekly)

The weekly time series shows:

- replacements occur across the full sample, with visible spikes in certain periods of high volatitility as noted above
- fills spike heavily at two seperate instances:
    - at the end (late 2025), consistent with extending to `END=2025-12-31` and relying on yfinance to fill the final days
    - between december 2025 and january 2026, pointing to a missing period in the base feed that yfinance covers

---

### Crypto (Base before yfinance)

- **rows:** 228,553  
- **tickers:** 105  
- **range:** 2010-07-17 → 2025-12-12  
- **missing open/close:** 0%  
- **max gap:** up to 806 days

Crypto is structurally different:

- Many assets did not exist in 2010–2015
- Some have long missing periods
- Symbol mapping to yfinance (`BTC -> BTC-USD`) is necessary and might not cover all coins

---

### Crypto (Final after yfinance fill + outlier replacement)

- **rows:** 265,180  
- **tickers:** 106  
- **base missing rows:** 36,627 (**13.812%**)  
- **yfinance available rows:** 242,160 (**91.319%**)  
- **outliers (any):** 30,567 (**11.527%**)  

Crypto shows the same pattern as equities: ~11.5% outliers flagged.

Additionally, the QA reports show non-positive values in the final crypto dataset, that strongly suggests that for some stablecoins or illiquid coins, yfinance may output zeros (or the base feed contains zeros and survived due to merge behavior).

---

## Task 3 — Model Building (Linear Regression for NVDA Next-Day Close)

## Feature design

Features are built at day **t**:

1. **Lagged closes:**  
   `c0 = close_t`, `c1 = close_{t-1}`, ..., `c(L-1)`

2. **Simple momentum proxies:**  
   - `ret_1` (1-day return)
   - `ret_5` (5-day return)

3. **Local level & volatility:**  
   - `ma_3` (short moving average)
   - `ma_L` (lookback moving average)
   - `std_L` (lookback volatility proxy)

The lookback `L` is chosen by validation MAPE on 2024 from:
- `[5, 10, 20, 30]`

**Result:** best lookback was **L = 5**, validation MAPE ≈ **2.46%**.

Interpretation:
- For this baseline, short memory works best (recent information dominates).

---

## Test results (2025)

- **Test MAPE (2025): ~2.14%**

This is a strong baseline for next-day close prediction, but it is also expected given the nature of the target:

- Next-day close is highly autocorrelated with today's close
- A linear model with close lags should be able to approximate a "persistence + small correction" behavior

The test plot (true vs predicted close) shows:
- predictions track the general trend closely being slightly delayed
- the model slightly smooths abrupt jumps (news-driven moves etc.), which is typical for purely technical linear models

---

## Forward forecasting (two modes)

### (a) Ex-post prediction
Uses *real* price history up to the last trading day before the target, via yfinance extension.

- used date: 2026-01-16
- predicted: **186.51**
- true: **186.23**
- relative error: **0.148%**

Interpretation:
- With true history available, the model behaves like a one-step predictor and does well.
- Not really a realistic forecast setting, but might serve as a decent sanity check.

---

### (b) Autoregressive multi-step forecast
From a cutoff date, predict one day forward, append prediction, repeat.

In the notebook's shown run:
- cutoff: **2025-12-01**
- target: **2026-01-16**
- predicted: **181.31**
- true: **186.23**
- relative error: **2.64%**

Interpretation:
- Error compounds when predictions feed future features
- Return and volatility features tend to collapse toward small values
- The forecast path becomes overly smooth (the plotted red line drifts slowly)

> Linear regression with close-based features can produce decent next-day predictions for NVDA (2.14% MAPE on test), but multi-step forecasts degrade quickly due to error compounding and feature collapse.

---

## Critical limitations

### Linear regression is a baseline at best
Even at ~2% MAPE, this does not imply profitability:
- transaction costs, slippage, and risk dominate
- MAPE does not measure directional accuracy directly
- the model mostly learns that tomorrow ≈ today

### No exogenous variables
The model ignores:
- earnings events
- macro news
- sector moves
- volume/flow signals
- regime shifts

So it cannot react to many real drivers of NVDA moves.
