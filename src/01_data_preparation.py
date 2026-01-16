from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ------------------ config ------------------

DATA_DIR = Path("data")
RAW, OUT = DATA_DIR / "raw", DATA_DIR / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2025-12-31")
EFFECTIVE_END = min(END, pd.Timestamp.today().normalize())

# Relative tolerance for confirming a split jump:
# confirmed if |ratio - split_factor| / |split_factor| <= TOL
TOL = 0.05


# ------------------ helpers ------------------

def to_date(s: pd.Series) -> pd.Series:
    """Parse a date-like Series to tz-naive pandas timestamps (NaT on errors)."""
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names for robust downstream processing."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def quality(df: pd.DataFrame, name: str) -> None:
    """
    Print a compact quality snapshot of a price-like dataset.

    Expected structure for price datasets:
      - ticker, date, open, close
    """
    print(f"\n=== {name} ===")
    print("rows:", len(df), "| tickers:", df["ticker"].nunique() if "ticker" in df else "n/a")
    if "date" in df:
        print("range:", df["date"].min(), "->", df["date"].max())
    if {"open", "close"}.issubset(df.columns):
        print("missing open:", df["open"].isna().mean(), "| missing close:", df["close"].isna().mean())
    if {"ticker", "date"}.issubset(df.columns):
        print("dup(ticker,date):", df.duplicated(["ticker", "date"]).sum())


# ------------------ loaders/transformers ------------------

def load_price_dir(price_file: Path) -> pd.DataFrame:
    """
    Load all CSVs in a directory and return one long DataFrame with columns:
      ticker, date, open, close

    Assumptions:
      - Each CSV has at least a 'date' column
      - If no 'ticker' column exists, the file stem is used as ticker
      - open/close are converted to numeric (coerce invalids to NaN)
      - Data is restricted to [START, END]
      - (ticker, date) duplicates are dropped
    """
    frames = []
    for f in sorted(price_file.glob("*.csv")):
        d = clean_cols(pd.read_csv(f, keep_default_na=False))
        if "date" not in d.columns:
            continue

        if "ticker" not in d.columns:
            d["ticker"] = f.stem

        d["ticker"] = d["ticker"].astype(str).str.strip()
        d["date"] = to_date(d["date"])

        for c in ("open", "close"):
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

        d = (
            d.dropna(subset=["ticker", "date"])
            .loc[lambda x: x["date"].between(START, END), ["ticker", "date", "open", "close"]]
            .drop_duplicates(["ticker", "date"])
        )
        frames.append(d)

    if not frames:
        raise RuntimeError(f"No usable CSVs in {price_file}")

    return pd.concat(frames).sort_values(["ticker", "date"]).reset_index(drop=True)


def daily_to_weekly(df: pd.DataFrame, week_ending: str = "FRI") -> pd.DataFrame:
    """
    Aggregate daily prices to weekly prices.

    Weekly definition is controlled via `week_ending` (e.g. 'FRI' for equities, 'SUN' for crypto).

    Output columns:
      - date is the period start timestamp for that week (aligned to `W-{week_ending}`)
      - open is the first observed open within that week
      - close is the last observed close within that week
    """
    x = df.sort_values(["ticker", "date"]).copy()
    x["date"] = x["date"].dt.to_period(f"W-{week_ending}").dt.to_timestamp()
    w = x.groupby(["ticker", "date"], as_index=False).agg(open=("open", "first"), close=("close", "last"))

    return w.dropna(subset=["open", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_splits(split_file: Path) -> pd.DataFrame:
    """
    Load split events and return standardized columns:
      ticker, date, split_factor

    Notes:
      - 'date' is normalized to midnight (tz-naive)
      - split_factor is numeric and non-zero
      - restricted to [START, END]
    """
    sp = clean_cols(pd.read_csv(split_file, keep_default_na=False))
    sp = sp.rename(
        columns={
            "symbol": "ticker",
            "stock splits": "split_factor",
        }
    )
    sp["date"] = to_date(sp["date"]).dt.normalize()
    sp["split_factor"] = pd.to_numeric(sp["split_factor"], errors="coerce")

    sp = sp.dropna(subset=["ticker", "date", "split_factor"])
    sp = sp[sp["date"].between(START, END) & (sp["split_factor"] != 0)]
    return sp[["ticker", "date", "split_factor"]]


def adjust_splits_if_needed(prices: pd.DataFrame, splits: pd.DataFrame, tol: float = TOL) -> tuple[pd.DataFrame, int]:
    """
    Check all provided split events against daily prices and apply split-adjustments
    only when the split is detected as an unadjusted price jump in the dataset.

    Core logic:
      1) For each split event, find the next available trading day in the price data
         (important: split date may be weekend/holiday).
      2) Confirm split presence by checking:
           ratio = close_{t-1} / close_{t}
         where t is the matched trading day, and compare ratio to split_factor.
      3) If confirmed, adjust all earlier historical prices by dividing open/close by factor.

    Returns:
      (adjusted_prices_df, number_of_applied_split_events)

    Implementation detail for (1): uses `merge_asof(..., direction="forward")` to map
    a split date to the next trading date available in the price series.
    """
    out = []
    applied = 0

    prices = prices[["ticker", "date", "open", "close"]].copy()
    splits = splits[["ticker", "date", "split_factor"]].copy()

    for t, dft in prices.groupby("ticker", sort=False):
        spt = splits[splits["ticker"] == t].sort_values("date")
        if spt.empty or len(dft) < 2:
            out.append(dft.sort_values("date"))
            continue

        dft = dft.sort_values("date").copy()
        dft["prev_close"] = dft["close"].shift(1)

        # Keep split date separate from matched trading date
        left = spt.rename(columns={"date": "split_date"}).copy()
        right = dft[["date", "close", "prev_close"]].rename(columns={"date": "trade_date"}).copy()

        # merge_asof requires sorted keys
        left = left.sort_values("split_date")
        right = right.sort_values("trade_date")

        chk = pd.merge_asof(
            left,
            right,
            left_on="split_date",
            right_on="trade_date",
            direction="forward",
            allow_exact_matches=True,
        )

        # If the split_date is after the last available trading date -> trade_date becomes NaT
        chk = chk.dropna(subset=["trade_date", "close", "prev_close", "split_factor"])
        if chk.empty:
            out.append(dft.drop(columns="prev_close"))
            continue

        ratio = chk["prev_close"] / chk["close"]
        ok = ((ratio - chk["split_factor"]).abs() / chk["split_factor"].abs()) <= tol

        # Confirmed splits, aggregated per matched trading day (in case of multiple events)
        confirmed = (
            chk.loc[ok, ["trade_date", "split_factor"]]
            .groupby("trade_date", as_index=False)["split_factor"]
            .prod()
            .sort_values("trade_date", ascending=False)
        )

        if confirmed.empty:
            out.append(dft.drop(columns="prev_close"))
            continue

        # Apply in reverse chronological order: adjust all data strictly before trade_date
        for dt, f in confirmed.itertuples(index=False):
            mask = dft["date"] < dt
            dft.loc[mask, ["open", "close"]] = dft.loc[mask, ["open", "close"]] / float(f)

        applied += len(confirmed)
        out.append(dft.drop(columns="prev_close"))

    prices_adj = pd.concat(out, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices_adj, applied


def yf_fetch_open_close(tickers: list[str], start: pd.Timestamp = START, end: pd.Timestamp = EFFECTIVE_END, retries: int = 4) -> pd.DataFrame:
    """
    Download daily Open/Close data via yfinance for a list of tickers.

    Returns:
      ticker, date, yf_open, yf_close

    Notes:
      - Uses auto_adjust=False (raw prices).
      - `end` is treated as inclusive in our pipeline, but yfinance uses end-exclusive,
        so we request end+1 day.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "date", "yf_open", "yf_close"])
    
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # end exclusive
        auto_adjust=False,
        group_by="ticker",
        progress=True,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame(columns=["ticker", "date", "yf_open", "yf_close"])

    syms = sorted(set(data.columns.get_level_values(0)))
    frames = [
        data[sym][["Open", "Close"]]
        .rename(columns={"Open": "yf_open", "Close": "yf_close"})
        .reset_index()
        .assign(ticker=sym)
        for sym in syms
    ]

    y = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date"})
    y["date"] = to_date(y["date"])
    y["yf_open"] = pd.to_numeric(y["yf_open"], errors="coerce").round(5)
    y["yf_close"] = pd.to_numeric(y["yf_close"], errors="coerce").round(5)

    return y.dropna(subset=["ticker", "date", "yf_open", "yf_close"])[["ticker", "date", "yf_open", "yf_close"]]


def outlier_mask_iqr(
    df: pd.DataFrame,
    value_col: str = "close"
) -> pd.Series:
    """
    Flag outliers per `by` group using Tukey's IQR rule on *levels*:

        outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR

    Returns a boolean Series aligned to `df.index`.
    """
    x = df[["ticker", value_col]].copy()

    g = x.groupby("ticker")[value_col]
    q1 = g.transform(lambda s: s.quantile(0.25))
    q3 = g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1

    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr

    mask = (x[value_col] < lo) | (x[value_col] > hi)
    return mask.fillna(True)


def merge_prices_with_yf_replace_outliers(
    base: pd.DataFrame,
    yf: pd.DataFrame,
    replace_outliers_yf: bool = True
) -> pd.DataFrame:
    """
    Outer-join base with yfinance, then:
      - fill missing open/close from yfinance
      - detect outliers on BASE (per ticker, IQR rule on levels)
      - replace outlier rows with yfinance open/close where available
      - drop rows still missing open/close
    """
    m = (base.merge(yf, on=["ticker", "date"], how="outer")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True))

    ob, cb = m["open"], m["close"]
    oy, cy = m["yf_open"], m["yf_close"]

    # fill missing from yfinance
    m["open"], m["close"] = ob.combine_first(oy), cb.combine_first(cy)

    # outliers on base only
    base_ok = ob.notna() & cb.notna()
    out = pd.Series(False, index=m.index)
    if base_ok.any():
        b = pd.DataFrame({"ticker": m.loc[base_ok, "ticker"], "open": ob[base_ok], "close": cb[base_ok]})
        out.loc[base_ok] = outlier_mask_iqr(b, "open") | outlier_mask_iqr(b, "close")

    if replace_outliers_yf:
        # replace with yfinance where available
        repl = out & oy.notna() & cy.notna()
        m.loc[repl, ["open", "close"]] = np.c_[oy[repl], cy[repl]]
    else:
        # simply drop outliers outright
        m = m.loc[~out]

    m = (m.dropna(subset=["ticker", "date"])
           .loc[m["date"].between(START, END)]
           .dropna(subset=["open", "close"])
           .drop_duplicates(["ticker", "date"]))

    m[["open", "close"]] = m[["open", "close"]].round(5)
    return m[["ticker", "date", "open", "close"]].sort_values(["ticker", "date"]).reset_index(drop=True)


def load_meta(meta_file: Path, addr_file: Path) -> pd.DataFrame:
    """
    Load NASDAQ metadata and merge with a company address file (if available).

    Returns a DataFrame keyed by ticker with all available descriptive columns.
    """
    meta = clean_cols(pd.read_csv(meta_file, keep_default_na=False)).rename(columns={"symbol": "ticker"})
    meta = meta.drop_duplicates("ticker")

    addr = clean_cols(pd.read_csv(addr_file, keep_default_na=False))
    if "ticker" not in addr.columns:
        raise RuntimeError("Address file must have column 'ticker'.")
    if "address" not in addr.columns:
        addr["address"] = np.nan

    addr = addr.drop_duplicates("ticker")[["ticker", "address"]]

    return meta.merge(addr, on="ticker", how="left").reset_index(drop=True)


def yf_symbol_crypto(t: str) -> str:
    """Convert a plain crypto ticker (e.g. BTC) to yfinance format (e.g. BTC-USD) if needed."""
    return t if "-" in t else f"{t}-USD"


# ------------------ run ------------------

# Metadata (required output)
df_nasdaq_meta = load_meta(RAW / "nasdaq_screener.csv", RAW / "nasdaq_company_addresses.csv")

# ------------------ NASDAQ ------------------

nasdaq = load_price_dir(RAW / "nasdaq-daily")
splits = load_splits(RAW / "splits_2000_2025.csv")

# Split-check + conditional adjustment (robust to non-trading split dates)
nasdaq, n_adj = adjust_splits_if_needed(nasdaq, splits, tol=TOL)
quality(nasdaq, "NASDAQ daily (after split-check, before yfinance)")

# Fetch yfinance for ALL tickers
all_n = sorted(nasdaq["ticker"].unique())
yf_n = yf_fetch_open_close(all_n, START, EFFECTIVE_END)

# Fill missing + replace outliers using yfinance
df_nasdaq_daily = merge_prices_with_yf_replace_outliers(nasdaq, yf_n)
df_nasdaq_weekly = daily_to_weekly(df_nasdaq_daily, week_ending="FRI")

# ------------------ Crypto ------------------

crypto = load_price_dir(RAW / "crypto-daily")
quality(crypto, "Crypto daily (before yfinance)")

# Fetch yfinance for ALL crypto tickers
all_c = sorted(crypto["ticker"].unique())

# yfinance crypto symbols are often {TICKER}-USD (e.g., BTC-USD)
sym_map = {t: yf_symbol_crypto(t) for t in all_c}
inv_map = {v: k for k, v in sym_map.items()}

yf_c = yf_fetch_open_close(list(sym_map.values()), START, EFFECTIVE_END)
if not yf_c.empty:
    yf_c["ticker"] = yf_c["ticker"].map(inv_map).fillna(yf_c["ticker"])

# Fill missing + replace outliers using yfinance
df_crypto_daily = merge_prices_with_yf_replace_outliers(crypto, yf_c)
df_crypto_weekly = daily_to_weekly(df_crypto_daily, week_ending="SUN")

# ------------------ (FX for 2.5 DCA) EURUSD from yfinance ------------------

eurusd_raw = yf_fetch_open_close(["EURUSD=X"], START, EFFECTIVE_END)
df_eurusd_daily = (
    eurusd_raw.rename(columns={"yf_open": "open", "yf_close": "close"})
    .assign(ticker="EURUSD")[["ticker", "date", "open", "close"]]
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)
df_eurusd_weekly = daily_to_weekly(df_eurusd_daily, week_ending="FRI")

# ------------------ quality + save ------------------

quality(df_nasdaq_daily, "df_nasdaq_daily (final)")
quality(df_nasdaq_weekly, "df_nasdaq_weekly (final)")
quality(df_crypto_daily, "df_crypto_daily (final)")
quality(df_crypto_weekly, "df_crypto_weekly (final)")
quality(df_eurusd_daily, "df_eurusd_daily (final)")
quality(df_eurusd_weekly, "df_eurusd_weekly (final)")
quality(df_nasdaq_meta, "df_nasdaq_meta (final)")

# Required file names
df_nasdaq_daily.to_csv(OUT / "df_nasdaq_daily.csv", index=False)
df_nasdaq_weekly.to_csv(OUT / "df_nasdaq_weekly.csv", index=False)
df_crypto_daily.to_csv(OUT / "df_crypto_daily.csv", index=False)
df_crypto_weekly.to_csv(OUT / "df_crypto_weekly.csv", index=False)
df_nasdaq_meta.to_csv(OUT / "df_nasdaq_meta.csv", index=False)

# Optional (FX)
df_eurusd_daily.to_csv(OUT / "df_eurusd_daily.csv", index=False)
df_eurusd_weekly.to_csv(OUT / "df_eurusd_weekly.csv", index=False)

print("\nSaved to:", OUT.resolve())
print("Splits adjusted (unadjusted jumps detected):", n_adj)
print("EFFECTIVE_END used for yfinance:", EFFECTIVE_END.date())